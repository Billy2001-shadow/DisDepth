import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

# 添加当前目录到 PYTHONPATH
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tinyvit.dpt import TinyVitDpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer TinyVit for Relative Depth Estimation')
    
    parser.add_argument('--img-path', default='rgb_00017.jpg',type=str)
    parser.add_argument('--input_height', default=224, type=int)
    parser.add_argument('--input_width', default=224, type=int) # 输入到网络中的size
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    parser.add_argument('--pretrained-from', type=str, default='exp/tinyvit/TinyVit_relative_12_25_00_43.pth')  # TinyVit_relative_12_24_08_55

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
    
    model_suffix = os.path.splitext(os.path.basename(args.pretrained_from))[0].split('_')[-4:]
    model_suffix = '_'.join(model_suffix) # 将日期部分拼接成字符串

    model.cuda().eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = model.infer_image(raw_image, args.input_size) # (H, W)
        
        depth = np.squeeze(depth,axis=0) # 直接保存为npy文件  224*224  # Max=3529.0732  Min=233.85077

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0 # 归一化到0-255
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            save_filename = f"{base_filename}_{model_suffix}.png"
            cv2.imwrite(os.path.join(args.outdir, save_filename), depth)
            # cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)


     