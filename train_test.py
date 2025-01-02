import os
import argparse
import logging
import warnings
import pprint
import numpy as np
import random
from datetime import datetime

import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from tinyvit.dpt import TinyVitDpt
from midas_loss import ScaleAndShiftInvariantLoss

from dataset.nyu2 import get_nyud_loader
from dataset.kitti import get_kitti_loader
from dataset.diode import get_diode_loader
from dataset.ddad import get_ddad_loader
from dataset.eth3d import get_eth3d_loader
from dataset.dataloader import get_train_loader

from util.utils import init_log
from util.metric import recover_metric_depth, compute_errors, RunningAverageDict


parser = argparse.ArgumentParser(description='Train TinyVit for Relative Depth Estimation')


parser.add_argument('--input_height', default=224, type=int)
parser.add_argument('--input_width', default=224, type=int) # 输入到网络中的size
# train
parser.add_argument('--distributed',default=False, type=bool)
parser.add_argument('--batch_size', default=96, type=int)
parser.add_argument('--workers', default=16, type=int)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--lr', default=0.0001, type=float) # 调一调学习率是否可以加快收敛呢？ 0.000005 0.000008  00008
parser.add_argument('--save-path', default='exp/tinyvit',type=str)
parser.add_argument('--filelist_path', default='/home/chenwu/DisDepth/dataset/splits/train/relative_depth_train.txt',type=str)
# data augmentation

# dataset val
parser.add_argument('--nyu_val', default='/home/chenwu/DisDepth/dataset/splits/val/nyu_val.txt', type=str)
parser.add_argument('--kitti_val', default='/home/chenwu/DisDepth/dataset/splits/val/kitti_val.txt', type=str)
parser.add_argument('--ddad_val', default='/home/chenwu/DisDepth/dataset/splits/val/ddad_val.txt', type=str)
parser.add_argument('--diode_outdoor_val', default='/home/chenwu/DisDepth/dataset/splits/val/diode_outdoor_val.txt', type=str)
parser.add_argument('--diode_indoor_val', default='/home/chenwu/DisDepth/dataset/splits/val/diode_indoor_val.txt', type=str) 
parser.add_argument('--eth3d_indoor_val', default='/home/chenwu/DisDepth/dataset/splits/val/eth3d_indoor_val.txt', type=str) 
parser.add_argument('--eth3d_outdoor_val', default='/home/chenwu/DisDepth/dataset/splits/val/eth3d_outdoor_val.txt', type=str) 

parser.add_argument('--seed', '--rs', default=12, type=int,help='random seed (default: 0)') # default=12
parser.add_argument('--pretrained_from', default='/home/chenwu/TinyVit/checkpoints/tiny_vit_5m_22kto1k_distill.pth', type=str,help='the trained model weight') #加载相对深度权重


def main():
    args = parser.parse_args()

    # 确保训练过程的随机性可控，实验结果具有可复现性。
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    warnings.simplefilter('ignore', np.RankWarning)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    all_args = vars(args)
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True  

    ############################################## Dataset Load ########################################################
     # dataloader
    size = (args.input_width, args.input_height)  
    trainloader  = get_train_loader(args, 'train')
    valloader = get_nyud_loader(data_dir_root=args.filenames_file_eval,size =size)

    ###########################################################################################################################


    ############################################## Student Model Load ########################################################

    model_configs = {
        '5m_224':  {'embed_dims': [64, 128, 160, 320], 'features': 64, 'in_channels':[128, 160, 320,320],'out_channels': [48, 96, 192, 384],'num_heads':[2, 4, 5, 10],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.0},
        '11m_224': {'embed_dims': [64, 128, 256, 448], 'features': 128, 'in_channels':[128, 256, 448,448],'out_channels': [96, 192, 384, 768],'num_heads':[2, 4, 8, 14],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.1},
        '21m_224': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.2},
        '21m_384': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[12, 12, 24, 12],'drop_path_rate':0.1},
        '21m_512': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[16, 16, 32, 16],'drop_path_rate':0.1},
    }

    model = TinyVitDpt(config=args, **model_configs['5m_224'],use_bn=True)
    # model = torch.nn.DataParallel(model)
    
    if args.pretrained_from:
        #1.加载TinyVit的预训练权重
        state_dict = torch.load(args.pretrained_from,map_location='cuda')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.pretrained.load_state_dict(state_dict)
        
 
    model.cuda() # 将模型移动到 GPU
    ###########################################################################################################################


    ############################################## Loss &&  Optimizer  ########################################################
   
    criterion = ScaleAndShiftInvariantLoss().cuda()
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    ###########################################################################################################################
   

    total_iters = args.epochs * len(trainloader)

    for epoch in range(args.epochs):
        logger.info('epoch: {}/{}'.format(epoch, args.epochs))
        model.train()
        total_loss = 0
        ################################# Train loop ##########################################################
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            img, depth  = sample['image'].cuda(), sample['depth'].cuda() #valid_mask和 depth是对应的，depth>0的地方是1，否则是0
            
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                
            pred = model(img) 
            pred = pred.squeeze(1)
            valid_mask = depth >= 0 # 如果预测出来等于0的地方呢？

            loss = criterion(pred, depth, valid_mask)
           
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iters = epoch * len(trainloader) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            writer.add_scalar('train/loss', loss.item(), iters)

            if i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))

        ############################################################################################################


        ################################# Eval loop ##########################################################
        model.eval()
        metrics =RunningAverageDict()
        with torch.no_grad():
            for batch_idx, sample in tqdm(enumerate(valloader),total=len(valloader)):
                img, depth = sample['image'].cuda().float(), sample['depth'].cuda()[0]  # torch.Size([1, 3, 224, 224]) depth torch.Size([1, 1, 480, 640])
                
                pred = model(img) # torch.Size([1,1, 224, 224])     
                pred = F.interpolate(pred, depth.shape[-2:], mode='bilinear',align_corners=True)  # nearest  bilinear | bicubic torch.Size([1, 1, 480, 640])
                pred = pred.squeeze().cpu().numpy()   # torch.Size([480, 640]) 
                depth = depth.squeeze().cpu().numpy() # torch.Size([480, 640]) 
                
                valid_mask = (depth > args.min_depth) & (depth <= args.max_depth)
                eval_mask = np.zeros(valid_mask.shape)
                eval_mask[45:471, 41:601] = 1 # eigen crop for nyu
                valid_mask = np.logical_and(valid_mask, eval_mask)

                # 对其pred和depth  整理一个API出来
                pred = recover_metric_depth(pred,depth,valid_mask)
            
                metrics.update(compute_errors(depth[valid_mask], pred[valid_mask])) # 第二步1.0 / pred[valid_mask]
        
            metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
            print(f"Metrics: {metrics}")

            # 将指标记录到txt文件中
            with open(os.path.join(args.save_path, 'metrics.txt'), 'a') as f:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"Epoch {epoch} [{current_time}]: {metrics}\n")

        ############################################################################################################
        checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
        filename = os.path.join(args.save_path, f'latest_epoch_{epoch}.pth')
        torch.save(checkpoint, filename)
        ############################################################################################################



if __name__ == '__main__':
    main()


    # valloader = get_nyud_loader(data_dir_root="/home/chenwu/DisDepth/dataset/splits/val/nyu_val.txt")
    # valloader = get_kitti_loader(data_dir_root="/home/chenwu/DisDepth/dataset/splits/val/kitti_val.txt")
    # valloader = get_diode_loader(data_dir_root="/home/chenwu/DisDepth/dataset/splits/val/diode_outdoor_val.txt") # indoor
    # valloader = get_ddad_loader(data_dir_root="/home/chenwu/DisDepth/dataset/splits/val/ddad_val.txt")
    # valloader = get_eth3d_loader(data_dir_root="/home/chenwu/DisDepth/dataset/splits/val/eth3d_indoor_val.txt") # outdoor

    # for idx, data in enumerate(valloader):
    #     # torch.Size([1, 3, 224, 288])  torch.Size([1, 480, 640]) torch.Size([1, 480, 640])
    #     image,depth,valid_mask = data['image'],data['depth'],data['valid_mask']  # image torch.Size([1, 3, 224, 224]) depth torch.Size([1, 1, 480, 640])
    #     pass # debug point

# nyu:torch.Size([1, 3, 224, 288]) 
# kitti:torch.Size([1, 3, 224, 736])
# diode_indoor:torch.Size([1, 3, 224, 288]) 
# diode_outdoor:torch.Size([1, 3, 224, 288])
# ddad:torch.Size([1, 3, 224, 352])
# eth3d_indoor: torch.Size([1, 3, 224, 352])
# eth3d_outdoor: torch.Size([1, 3, 224, 320])

