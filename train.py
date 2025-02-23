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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from tinyvit.dpt import TinyVitDpt

from dataset.nyu2 import get_nyud_loader
from dataset.kitti import get_kitti_loader
from dataset.dataloader import get_train_loader

from util.utils import init_log
from util.metric import compute_errors, RunningAverageDict, align_depth_least_square,submap_align,blend_overlap

from loss_function import VNL_Loss
from midas_loss import ScaleAndShiftInvariantLoss

parser = argparse.ArgumentParser(description='Train TinyVit for Relative Depth Estimation')


parser.add_argument('--input_height', default=224, type=int)
parser.add_argument('--input_width', default=224, type=int) # 输入到网络中的size
parser.add_argument('--min_depth', default=0.001, type=float)
parser.add_argument('--max_depth', default=10, type=float)
# train
parser.add_argument('--distributed',default=False, type=bool)
parser.add_argument('--batch_size', default=192, type=int)
parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--lr', default=0.0001, type=float) # 调一调学习率是否可以加快收敛呢？ 0.000005 0.000008  00008
parser.add_argument('--save-path', default='exp/tinyvit',type=str)
# parser.add_argument('--filelist_path', default='/home/chenwu/DisDepth/dataset/splits/train/train_uint8_5.txt',type=str)
parser.add_argument('--filelist_path', default='/home/chenwu/DisDepth/dataset/splits/train',type=str) # 数据集splits的根目录
# 使用数据集
parser.add_argument('--datasets', nargs='+', default=['LSUN','Object365','ImageNet21K','GoogleLandmark','DIML_indoor','KITTI','ApolloScapeExtra','Cityscapes','NYU'], type=str, help='List of datasets') # 

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
parser.add_argument('--pretrained_from', default='/home/chenwu/DisDepth/checkpoints/tiny_vit_5m_22kto1k_distill.pth', type=str,help='the trained model weight') #加载相对深度权重
# parser.add_argument('--pretrained_from', default='/home/chenwu/DisDepth/pth_save/latest_epoch_39_0.952_0.853.pth', type=str,help='the trained model weight') #加载相对深度权重

# 函数：记录指标到 txt 文件中
def log_metrics(epoch, dataset_name, metrics, save_path):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics_str = {k: round(v, 3) for k, v in metrics.items()}
    with open(save_path, 'a') as f:
        f.write(f"Epoch {epoch} [{current_time}] - {dataset_name} Metrics: {metrics_str}\n")

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

    # 记录参数到txt文件
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(args.save_path, f'log_{current_time}.txt')
    with open(log_file, 'w') as f:
        for key, value in all_args.items():
            f.write(f'{key}: {value}\n')


    ############################################## Dataset Load ########################################################
     # dataloader
    size = (args.input_width, args.input_height)  
    trainloader  = get_train_loader(args, 'train')
    nyu_valloader = get_nyud_loader(data_dir_root=args.nyu_val,size =size) # filenames_file_eval
    kitti_valloader = get_kitti_loader(data_dir_root=args.kitti_val,size =size)
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
        # model.load_state_dict(state_dict) # 加载全部权重

        # 只加载包含 'pretrained' 的键的部分
        # pretrained_state_dict = {k: v for k, v in state_dict.items() if 'pretrained' in k}
        # 加载到 model.pretrained
        # missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)
        
        # 打印成功加载的键
        # loaded_keys = pretrained_state_dict.keys()
        # print(f"Successfully loaded keys: {list(loaded_keys)}")
        # 打印缺失的键和意外的键
        # if missing_keys:
        #     print(f"Missing keys: {missing_keys}")
        # if unexpected_keys:
        #     print(f"Unexpected keys: {unexpected_keys}")

       
    model.cuda() # 将模型移动到 GPU
    ###########################################################################################################################


    ############################################## Loss &&  Optimizer  ########################################################
   
    scale_shift_invariant_loss = ScaleAndShiftInvariantLoss().cuda()
    # scale_shift_invariant_loss = SSIL_Loss().cuda()
    virtual_normal_loss = VNL_Loss(focal_x=200.0, focal_y=200.0,
                                            input_size=(224, 224) , sample_ratio=0.15).cuda()


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

            # 1.法向量损失
            depth = depth.unsqueeze(1)  # 将 depth 从 B H W 转换为 B 1 H W
            vnl_loss = virtual_normal_loss(depth, pred) 
            

            pred = pred.squeeze(1)
            depth = depth.squeeze(1)  # 将 depth 从 B 1 H W  转换为 B H W
            valid_mask = depth > 0 # 如果预测出来等于0的地方呢？
            ssil_loss = scale_shift_invariant_loss(pred, depth, valid_mask)

            loss = vnl_loss + ssil_loss
        

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iters = epoch * len(trainloader) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            writer.add_scalar('loss_vnl', vnl_loss.item(), iters) 
            writer.add_scalar('loss_ssil', ssil_loss.item(), iters) 


            if i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))

        ############################################################################################################


        ################################# Eval loop ##########################################################
        model.eval()
        metrics =RunningAverageDict()
        with torch.no_grad():
            for batch_idx, sample in tqdm(enumerate(nyu_valloader),total=len(nyu_valloader)):
                img, depth = sample['image'].cuda().float(), sample['depth'].cuda()[0]  # torch.Size([1, 3, 224, 224]) depth torch.Size([1, 1, 480, 640])
                
                pred = model(img) # torch.Size([1,1, 224, 224])     
                pred = F.interpolate(pred, depth.shape[-2:], mode='bilinear',align_corners=True)  # nearest  bilinear | bicubic torch.Size([1, 1, 480, 640])
                pred = pred.squeeze().cpu().numpy()   # torch.Size([480, 640]) 
                depth = depth.squeeze().cpu().numpy() # torch.Size([480, 640]) 
                
                valid_mask = (depth > 0.01) & (depth < 10)
                pred = align_depth_least_square(pred,depth,valid_mask,10) # 10最大深度            
                metrics.update(compute_errors(depth[valid_mask], pred[valid_mask])) 
        
            metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
            print(f"Metrics: {metrics}")
            log_metrics(epoch, "NYU", metrics, log_file)  # 记录NYU数据集的评估结果

        metrics =RunningAverageDict()
        with torch.no_grad():
            for batch_idx, sample in tqdm(enumerate(kitti_valloader),total=len(kitti_valloader)):
                sub_image1 = sample['sub_image1'].cuda().float()
                sub_image2 = sample['sub_image2'].cuda().float()
                sub_image3 = sample['sub_image3'].cuda().float()
                sub_image4 = sample['sub_image4'].cuda().float()
                sub_image5 = sample['sub_image5'].cuda().float()
                sub_image6 = sample['sub_image6'].cuda().float()
                depth = sample['depth'].cuda()[0] 
                depth = depth.squeeze().cpu().numpy() 


                pred1 = model(sub_image1) # torch.Size([1,1, 224, 224])     
                pred2 = model(sub_image2) # torch.Size([1,1, 224, 224])
                pred3 = model(sub_image3) # torch.Size([1,1, 224, 224])
                pred4 = model(sub_image4) # torch.Size([1,1, 224, 224])
                pred5 = model(sub_image5) # torch.Size([1,1, 224, 224])
                pred6 = model(sub_image6) # torch.Size([1,1, 224, 224])
                pred1 = pred1.squeeze().cpu().numpy()
                pred2 = pred2.squeeze().cpu().numpy()
                pred3 = pred3.squeeze().cpu().numpy()
                pred4 = pred4.squeeze().cpu().numpy()
                pred5 = pred5.squeeze().cpu().numpy()
                pred6 = pred6.squeeze().cpu().numpy()

                # 右边子图对齐左边子图
                pred2 = submap_align(pred2, pred1)
                pred3 = submap_align(pred3, pred2)
                pred4 = submap_align(pred4, pred3)
                pred5 = submap_align(pred5, pred4)
                pred6 = submap_align(pred6, pred5)

                pred = np.zeros((224, 1184))
                pred = blend_overlap(pred, pred1, pred2, pred3, pred4, pred5, pred6) # d1': 0.896, 'd2': 0.975, 'd3': 0.993,
           
                valid_mask = (depth > 0.01) & (depth < 80)
                pred = align_depth_least_square(pred, depth, valid_mask, 80)
                metrics.update(compute_errors(depth[valid_mask], pred[valid_mask])) 
            metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
            print(f"Metrics: {metrics}")
            log_metrics(epoch, "KITTI", metrics, log_file) 

        # with torch.no_grad():
        #     for batch_idx, sample in tqdm(enumerate(kitti_valloader),total=len(kitti_valloader)):
        #         img, depth = sample['image'].cuda().float(), sample['depth'].cuda()[0]  # torch.Size([1, 3, 224, 224]) depth torch.Size([1, 1, 480, 640])
                
        #         pred = model(img) # torch.Size([1,1, 224, 224])     
        #         pred = F.interpolate(pred, depth.shape[-2:], mode='bilinear',align_corners=True)  # nearest  bilinear | bicubic torch.Size([1, 1, 480, 640])
        #         pred = pred.squeeze().cpu().numpy()   # torch.Size([480, 640]) 
        #         depth = depth.squeeze().cpu().numpy() # torch.Size([480, 640]) 
                
        #         valid_mask = (depth > 0.01) & (depth < 80)
        #         pred = align_depth_least_square(pred,depth,valid_mask,80) # 10最大深度            
        #         metrics.update(compute_errors(depth[valid_mask], pred[valid_mask])) 
        
        #     metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
        #     print(f"Metrics: {metrics}")
        #     log_metrics(epoch, "KITTI", metrics, log_file) 

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


