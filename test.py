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
    parser = argparse.ArgumentParser(description='TinyVit Relative Depth Estimation')
    
    parser.add_argument('--img-path', default='rgb_00017.jpg',type=str)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    parser.add_argument('--pretrained-from', type=str, default='exp/tinyvit/latest.pth')

    args = parser.parse_args()
    
    model = TinyVitDpt()
    if args.pretrained_from:
        model.load_state_dict(torch.load(args.pretrained_from, map_location='cpu')['model'])
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
        
        depth = np.squeeze(depth,axis=0) # 直接保存为npy文件  224*224

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)


     