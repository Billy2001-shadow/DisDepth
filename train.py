import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.utils.data
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss

from dataloader.dataloader import DepthDataLoader

import model_io  
from util.utils import RunningAverage
from util import utils



parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
# dataset
parser.add_argument('--dataset', default='nyu', choices=['nyu', 'kitti'])
parser.add_argument('--avoid_boundary', default=False, type=bool)
parser.add_argument('--data_path', default='/data2/cw/sync/', type=str)
parser.add_argument('--gt_path', default='/data2/cw/sync/', type=str)
parser.add_argument('--data_path_val', default='/data2/cw/nyu2_test/', type=str)
parser.add_argument('--gt_path_val', default='/data2/cw/nyu2_test/', type=str)
parser.add_argument('--filenames_file', default='/home/chenwu/DisDepth/dataloader/splits/nyudepthv2_train_files_with_gt.txt', type=str)
parser.add_argument('--filenames_file_eval', default='/home/chenwu/DisDepth/dataloader/splits/nyudepthv2_test_files_with_gt.txt', type=str)
parser.add_argument('--input_height', default=480, type=int)
parser.add_argument('--input_width', default=640, type=int)
parser.add_argument('--aug', default=False, type=bool)
parser.add_argument('--random_crop', default=True, type=bool)
parser.add_argument('--random_translate', default=True, type=bool)
parser.add_argument('--do_random_rotate', default=True, type=bool)

# Training related arguments
parser.add_argument('--img_size', default=518, type=int)
parser.add_argument('--min_depth', default=0.001, type=float)
parser.add_argument('--max_depth', default=10, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=32, type=int)
parser.add_argument('--lr', default=0.000005, type=float) #  0.0002 0.000005
parser.add_argument('--save_path', default="/data2/cw/dinov2_nyu/", type=str)


def main():
    args = parser.parse_args()


    ###################################### Load model ##############################################

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    model.cuda()

    ################################################################################################


     ###################################### Dataloader  ##############################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    ################################################################################################

    ###################################### losses ##############################################
    criterion_ueff = SiLogLoss().cuda
    ################################################################################################
   
     ###################################### Optimizer ################################################
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    ################################################################################################
    total_iters = args.epochs * len(train_loader)
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}

    

if __name__ == '__main__':
    main()