import os
import argparse
import logging
import warnings
import pprint
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from tinyvit.dpt import TinyVitDpt
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from midas_loss import ScaleAndShiftInvariantLoss,SiLogLoss
from dataset.DIML import DIML
from dataset.nyu2 import NYU
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.metric import *
from util.utils import init_log

parser = argparse.ArgumentParser(description='Train TinyVit for Relative Depth Estimation')


parser.add_argument('--input_height', default=224, type=int)
parser.add_argument('--input_width', default=224, type=int) # 输入到网络中的size
parser.add_argument('--min_depth', default=0.001, type=float)
parser.add_argument('--max_depth', default=10, type=float)
# train
parser.add_argument('--distributed',default=False, type=bool)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--workers', default=16, type=int)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--lr', default=0.0001, type=float) # 调一调学习率是否可以加快收敛呢？ 0.000005 0.000008  00008
parser.add_argument('--save-path', default='exp/tinyvit',type=str)
parser.add_argument('--w_silog', default=1.0, type=float)
parser.add_argument('--w_grad', default=0.0, type=float)
# data augmentation
parser.add_argument('--aug', default=True, type=bool)
parser.add_argument('--do_random_rotate', default=False, type=bool)
parser.add_argument('--random_crop', default=False, type=bool)
parser.add_argument('--random_translate', default=False, type=bool)
parser.add_argument('--degree', default=1.5, type=float) # 随机旋转正负1.5度
parser.add_argument('--max_translation', default=100, type=int)
parser.add_argument('--translate_prob', default=0.2, type=float)

parser.add_argument('--do_kb_crop', default=False, type=bool)
parser.add_argument('--avoid_boundary', default=False, type=bool) # 评估时eval_mask[45:471, 41:601] = 1

# dataset
parser.add_argument('--dataset', default='nyu', type=str)
parser.add_argument('--filenames_file', default='/home/chenwu/TinyVit/data_split/nyu2_train.txt', type=str)
parser.add_argument('--filenames_file_eval', default='/home/chenwu/TinyVit/data_split/nyu2_test.txt', type=str)
parser.add_argument('--data_path', default='/data2/cw/sync/', type=str) # /data2/cw/sync
parser.add_argument('--gt_path', default='/data2/cw/sync/', type=str) 
parser.add_argument('--data_path_eval', default='/data2/cw/nyu2_test/', type=str)
parser.add_argument('--gt_path_eval', default='/data2/cw/nyu2_test/', type=str)
parser.add_argument('--use_shared_dict', default=False, type=bool)


parser.add_argument('--seed', '--rs', default=12, type=int,help='random seed (default: 0)') # default=12
# parser.add_argument('--pretrained_from', default='/home/chenwu/TinyVit/checkpoints/latest.pth', type=str,help='the trained model weight') #加载相对深度权重
parser.add_argument('--pretrained_from', default='/home/chenwu/TinyVit/checkpoints/tiny_vit_5m_22kto1k_distill.pth', type=str,help='the trained model weight') #加载相对深度权重

# DepthAnythingV2
parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

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
     # dataset, dataloader
    size = (args.input_width, args.input_height)  
    trainset = DIML('dataset/splits/DIML/filtered_train_files.txt', 'train', size=size)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, num_workers=16,shuffle=True)

    valset = NYU('dataset/splits/nyu/val.txt', 'val', size=size)    
    Valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,shuffle=False)


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
    silogloss = SiLogLoss().cuda()
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
            img, depth ,valid_mask = sample['image'].cuda(), sample['depth'].cuda(),sample['valid_mask'].cuda() #valid_mask和 depth是对应的，depth>0的地方是1，否则是0
            
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)

           
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
        with torch.no_grad():
            metrics =RunningAverageDict()
            for i, sample in enumerate(Valloader):
                img, depth ,valid_mask = sample['image'].cuda(), sample['depth'].cuda(),sample['valid_mask'].cuda()
                pred = model(img)   
                # 先插值到原图大小
                # 再对齐pred和depth

                #
                loss = silogloss(pred, depth, valid_mask=valid_mask) # valid_mask 原版

        ############################################################################################################
        checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        ############################################################################################################


       

if __name__ == '__main__':
    main()









# import os
# import argparse
# import logging
# import warnings
# import pprint
# import numpy as np
# import random
# import torch
# import torch.backends.cudnn as cudnn
# from torch.optim import AdamW
# from torch.utils.tensorboard import SummaryWriter

# from tinyvit.dpt import TinyVitDpt
# from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
# from midas_loss import ScaleAndShiftInvariantLoss,SiLogLoss
# from dataset.DIML import DIML
# from dataset.nyu2 import NYU
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from util.metric import *
# from util.utils import init_log

# parser = argparse.ArgumentParser(description='Train TinyVit for Relative Depth Estimation')


# parser.add_argument('--input_height', default=224, type=int)
# parser.add_argument('--input_width', default=224, type=int) # 输入到网络中的size
# parser.add_argument('--min_depth', default=0.001, type=float)
# parser.add_argument('--max_depth', default=10, type=float)
# # train
# parser.add_argument('--distributed',default=False, type=bool)
# parser.add_argument('--batch_size', default=64, type=int)
# parser.add_argument('--workers', default=16, type=int)
# parser.add_argument('--epochs', default=40, type=int)
# parser.add_argument('--lr', default=0.0001, type=float) # 调一调学习率是否可以加快收敛呢？ 0.000005 0.000008  00008
# parser.add_argument('--save-path', default='exp/tinyvit',type=str)
# parser.add_argument('--w_silog', default=1.0, type=float)
# parser.add_argument('--w_grad', default=0.0, type=float)
# # data augmentation
# parser.add_argument('--aug', default=True, type=bool)
# parser.add_argument('--do_random_rotate', default=False, type=bool)
# parser.add_argument('--random_crop', default=False, type=bool)
# parser.add_argument('--random_translate', default=False, type=bool)
# parser.add_argument('--degree', default=1.5, type=float) # 随机旋转正负1.5度
# parser.add_argument('--max_translation', default=100, type=int)
# parser.add_argument('--translate_prob', default=0.2, type=float)

# parser.add_argument('--do_kb_crop', default=False, type=bool)
# parser.add_argument('--avoid_boundary', default=False, type=bool) # 评估时eval_mask[45:471, 41:601] = 1

# # dataset
# parser.add_argument('--dataset', default='nyu', type=str)
# parser.add_argument('--filenames_file', default='/home/chenwu/TinyVit/data_split/nyu2_train.txt', type=str)
# parser.add_argument('--filenames_file_eval', default='/home/chenwu/TinyVit/data_split/nyu2_test.txt', type=str)
# parser.add_argument('--data_path', default='/data2/cw/sync/', type=str) # /data2/cw/sync
# parser.add_argument('--gt_path', default='/data2/cw/sync/', type=str) 
# parser.add_argument('--data_path_eval', default='/data2/cw/nyu2_test/', type=str)
# parser.add_argument('--gt_path_eval', default='/data2/cw/nyu2_test/', type=str)
# parser.add_argument('--use_shared_dict', default=False, type=bool)


# parser.add_argument('--seed', '--rs', default=12, type=int,help='random seed (default: 0)') # default=12
# # parser.add_argument('--pretrained_from', default='/home/chenwu/TinyVit/checkpoints/latest.pth', type=str,help='the trained model weight') #加载相对深度权重
# parser.add_argument('--pretrained_from', default='/home/chenwu/TinyVit/checkpoints/tiny_vit_5m_22kto1k_distill.pth', type=str,help='the trained model weight') #加载相对深度权重

# # DepthAnythingV2
# parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

# def main():
#     args = parser.parse_args()

#     # 确保训练过程的随机性可控，实验结果具有可复现性。
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     np.random.seed(args.seed)  # Numpy module.
#     random.seed(args.seed)  # Python random module.
#     torch.manual_seed(args.seed)
#     warnings.simplefilter('ignore', np.RankWarning)

#     logger = init_log('global', logging.INFO)
#     logger.propagate = 0

#     all_args = vars(args)
#     logger.info('{}\n'.format(pprint.pformat(all_args)))
#     writer = SummaryWriter(args.save_path)

#     cudnn.enabled = True
#     cudnn.benchmark = True  

#     ############################################## Dataset Load ########################################################
#      # dataset, dataloader
#     size = (args.input_width, args.input_height)  
#     trainset = DIML('dataset/splits/DIML/filtered_train_files.txt', 'train', size=size)
#     trainloader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, num_workers=16,shuffle=True)

#     valset = NYU('dataset/splits/nyu/val.txt', 'val', size=size)    
#     Valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,shuffle=False)


#     ###########################################################################################################################

#     ############################################## Teacher Model Load ########################################################
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
#     model_configs = {
#         'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#         'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#         'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
#         'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
#     }
    
#     depth_anything = DepthAnythingV2(**model_configs[args.encoder])
#     depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
#     depth_anything = depth_anything.to(DEVICE).eval()
#     ###########################################################################################################################

#     ############################################## Student Model Load ########################################################

#     model_configs = {
#         '5m_224':  {'embed_dims': [64, 128, 160, 320], 'features': 64, 'in_channels':[128, 160, 320,320],'out_channels': [48, 96, 192, 384],'num_heads':[2, 4, 5, 10],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.0},
#         '11m_224': {'embed_dims': [64, 128, 256, 448], 'features': 128, 'in_channels':[128, 256, 448,448],'out_channels': [96, 192, 384, 768],'num_heads':[2, 4, 8, 14],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.1},
#         '21m_224': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.2},
#         '21m_384': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[12, 12, 24, 12],'drop_path_rate':0.1},
#         '21m_512': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[16, 16, 32, 16],'drop_path_rate':0.1},
#     }

#     model = TinyVitDpt(config=args, **model_configs['5m_224'],use_bn=True)
#     # model = torch.nn.DataParallel(model)
    
#     if args.pretrained_from:
#         #1.加载TinyVit的预训练权重
#         state_dict = torch.load(args.pretrained_from,map_location='cuda')
#         if 'model' in state_dict:
#             state_dict = state_dict['model']
#         model.pretrained.load_state_dict(state_dict)
 
#     model.cuda() # 将模型移动到 GPU
#     ###########################################################################################################################


#     ############################################## Loss &&  Optimizer  ########################################################
#     silogloss = SiLogLoss().cuda()
#     criterion = ScaleAndShiftInvariantLoss().cuda()
#     optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
#                        {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
#                       lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
#     ###########################################################################################################################
   

#     total_iters = args.epochs * len(trainloader)

#     for epoch in range(args.epochs):
#         logger.info('epoch: {}/{}'.format(epoch, args.epochs))
#         model.train()
#         total_loss = 0

#         ################################# Train loop ##########################################################
#         for i, sample in enumerate(trainloader):
#             optimizer.zero_grad()
#             img, depth ,valid_mask = sample['image'].cuda(), sample['depth'].cuda(),sample['valid_mask'].cuda() #valid_mask和 depth是对应的，depth>0的地方是1，否则是0
            
#             if random.random() < 0.5:
#                 img = img.flip(-1)
#                 depth = depth.flip(-1)
#                 valid_mask = valid_mask.flip(-1)

#             pred = model(img)

#             with torch.no_grad():
#                 depth = depth_anything(img)
#             loss = criterion(pred, depth, valid_mask)
           

#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             iters = epoch * len(trainloader) + i
#             lr = args.lr * (1 - iters / total_iters) ** 0.9

#             optimizer.param_groups[0]["lr"] = lr
#             optimizer.param_groups[1]["lr"] = lr * 10.0

#             writer.add_scalar('train/loss', loss.item(), iters)

#             if i % 100 == 0:
#                 logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))

#         ############################################################################################################


#         ################################# Eval loop ##########################################################
#         model.eval()
#         with torch.no_grad():
#             metrics =RunningAverageDict()
#             for i, sample in enumerate(Valloader):
#                 img, depth ,valid_mask = sample['image'].cuda(), sample['depth'].cuda(),sample['valid_mask'].cuda()
#                 pred = model(img)   
#                 # 先插值到原图大小
#                 # 再对齐pred和depth

#                 #
#                 loss = silogloss(pred, depth, valid_mask=valid_mask) # valid_mask 原版

#         ############################################################################################################
#         checkpoint = {
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#             }
#         torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
#         ############################################################################################################


       

# if __name__ == '__main__':
#     main()


