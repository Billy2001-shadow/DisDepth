import os
import argparse
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))) # 添加当前目录到 PYTHONPATH

import torch
import torch.nn.functional as F

from tinyvit.dpt import TinyVitDpt
from util.metric import recover_metric_depth, compute_errors, RunningAverageDict,align_depth_least_square,submap_align,blend_overlap
from dataset.eth3d import get_eth3d_loader

from scipy.ndimage import zoom
import matplotlib
import cv2
from tqdm import tqdm
from PIL import Image

cmap = matplotlib.colormaps.get_cmap('Spectral_r')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate for Relative Depth Estimation')
    
    parser.add_argument('--input_height', default=224, type=int)
    parser.add_argument('--input_width', default=224, type=int) # 输入到网络中的size
    parser.add_argument('--min_depth', default=0.001, type=float)
    parser.add_argument('--max_depth', default=10, type=float)
    
    parser.add_argument('--eth3d_eval_fileslist', default='dataset/splits/val/eth3d_indoor_val.txt', type=str) 
    # parser.add_argument('--eth3d_eval_fileslist', default='dataset/splits/val/eth3d_outdoor_val.txt', type=str) 

    parser.add_argument('--pretrained-from', type=str, default='exp/tinyvit/latest_epoch_39.pth')  # TinyVit_relative_12_24_08_55
    parser.add_argument('--eval_datasets', nargs='+', default=['ETH3D'], type=str, help='List of datasets') #  ['KITTI','NYU','DIODE','ETH3D']


    args = parser.parse_args()
    

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

    if 'ETH3D' in args.eval_datasets:
        eth3dvalloader = get_eth3d_loader(data_dir_root=args.eth3d_eval_fileslist,size =(args.input_height, args.input_width))
    
    print("loading model {}".format(args.pretrained_from))
    print("epoch = ",torch.load(args.pretrained_from)["epoch"])
   
    state_dict=torch.load(args.pretrained_from)["model"]
    model.load_state_dict(state_dict)
    
    model.cuda().eval() # 在加载完参数之后再设置为eval模式

    # # 室内50米 室外300米
    with torch.no_grad():
        if 'ETH3D' in args.eval_datasets:
            metrics =RunningAverageDict()
            with open('d1_results_eth3d.txt', 'a') as f:
                for batch_idx, sample in tqdm(enumerate(eth3dvalloader),total=len(eth3dvalloader)):
                    img, depth = sample['image'].cuda().float(), sample['depth'].cuda()[0]  # torch.Size([1, 3, 224, 224]) 
                    
                    pred = model(img) # torch.Size([1,1, 224, 224])     
                    pred = F.interpolate(pred, depth.shape[-2:], mode='bilinear',align_corners=True)  # nearest  bilinear | bicubic 
                    pred = pred.squeeze().cpu().numpy()   
                    depth = depth.squeeze().cpu().numpy()  #(4032, 6048) image和depth的分辨率竟然不一样？
                  
              
                    valid_mask = (depth > 0.01) & (depth < 50) # 室内50米 室外300米 
                
                    pred = align_depth_least_square(pred,depth,valid_mask,50) # 10最大深度 

                    # save_pred = pred.reshape(1,pred.shape[0],pred.shape[1])
                    # save_pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255.0 
                    # save_pred = save_pred.astype(np.uint8)
                    # save_pred = (cmap(save_pred)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                    # image_path = sample['image_path'][0]
                    # save_path = image_path.replace(".JPG","_pred.png")
                    # cv2.imwrite(save_path, save_pred)   

                    epsilon = 1e-8
                    depth = depth + epsilon
                    pred = pred + epsilon   
                    thresh = np.maximum((depth / pred), (pred / depth)) 
                    d1_mask = thresh < 1.25

                    valid_sum = valid_mask.sum()
                    d1 = (d1_mask & valid_mask).sum() / valid_sum

                    error_mask = ((thresh >= 1.25) & valid_mask)
                    image_path = sample['image_path'][0]

                    original_img  = np.array(Image.open(image_path))
                    color_layer = np.zeros_like(original_img)
                    color_layer[error_mask] = (0, 255, 0) # 绿色
                    alpha = 0.4
                    visualized  = (original_img * (1 - alpha) + color_layer * alpha).astype(np.uint8)

                    save_path = image_path.replace(".JPG","_error.png")
                    Image.fromarray(visualized).save(save_path)
                    
                    # # 将 image_path 和 d1 写入到文件
                    f.write(f"{image_path} {d1:.4f}\n")  # 写入格式: 路径 d1值

                    metrics.update(compute_errors(depth[valid_mask], pred[valid_mask])) # 第二步1.0 / pred[valid_mask]
        
                metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
                print(f"Metrics: {metrics}")