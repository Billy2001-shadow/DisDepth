import os
import argparse
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))) # 添加当前目录到 PYTHONPATH

import torch
import torch.nn.functional as F

from tinyvit.dpt import TinyVitDpt
from util.metric import recover_metric_depth, compute_errors, RunningAverageDict
from dataset.nyu2 import get_nyud_loader

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate for Relative Depth Estimation')
    
    parser.add_argument('--input_height', default=224, type=int)
    parser.add_argument('--input_width', default=224, type=int) # 输入到网络中的size
    parser.add_argument('--min_depth', default=0.001, type=float)
    parser.add_argument('--max_depth', default=10, type=float)
    
    parser.add_argument('--filenames_file_eval', default='dataset/splits/val/nyu_val.txt', type=str) 
    parser.add_argument('--pretrained-from', type=str, default='exp/tinyvit/latest_epoch_1.pth')  # TinyVit_relative_12_24_08_55

    args = parser.parse_args()
    
    torch.cuda.set_device(1)

    model_configs = {
        '5m_224':  {'embed_dims': [64, 128, 160, 320], 'features': 64, 'in_channels':[128, 160, 320,320],'out_channels': [48, 96, 192, 384],'num_heads':[2, 4, 5, 10],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.0},
        '11m_224': {'embed_dims': [64, 128, 256, 448], 'features': 128, 'in_channels':[128, 256, 448,448],'out_channels': [96, 192, 384, 768],'num_heads':[2, 4, 8, 14],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.1},
        '21m_224': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.2},
        '21m_384': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[12, 12, 24, 12],'drop_path_rate':0.1},
        '21m_512': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[16, 16, 32, 16],'drop_path_rate':0.1},
    }

    model = TinyVitDpt(config=args, **model_configs['5m_224'],use_bn=True)

    if args.pretrained_from:
        model.load_state_dict(torch.load(args.pretrained_from, map_location='cpu')['model'])

    valloader = get_nyud_loader(data_dir_root=args.filenames_file_eval,size =(args.input_height, args.input_width))

   
    print("loading model {}".format(args.pretrained_from))
    print("epoch = ",torch.load(args.pretrained_from)["epoch"])
   
    state_dict=torch.load(args.pretrained_from)["model"]
    model.load_state_dict(state_dict)
    metrics =RunningAverageDict()
    model.cuda().eval() # 在加载完参数之后再设置为eval模式
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
