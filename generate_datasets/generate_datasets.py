import argparse
import cv2
import glob
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

'''
python run.py \
  --encoder <vits | vitb | vitl | vitg> \
  --img-path <path> --outdir <outdir> \
  [--input-size <size>] [--pred-only] [--grayscale]
'''

# 如何使用该脚本
# 1.请替换--img-path参数为你的数据集路径

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img_path', type=str,default='./data_splits/DIML_indoor.txt') # 这里放一个列表
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
     
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))

    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    

    output_txt_path = "./DIML_indoor_pseudo_depth.txt"
    with open(output_txt_path, 'w') as output_file:

        for k, filename in enumerate(filenames):
            print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            
            if raw_image is None:
                print(f'Warning: Unable to read image {filename}. Skipping.')
                continue

            depth = depth_anything.infer_image(raw_image, args.input_size) # (1,1, 224, 224)
            depth = depth.squeeze()
            save_name = os.path.splitext(os.path.basename(filename))[0] + '.npy'
            outdir =  os.path.join(os.path.dirname(os.path.dirname(filename)) ,'pseudo_depth')
            save_path = os.path.join(outdir, save_name)

            os.makedirs(outdir, exist_ok=True)

            np.save(save_path, depth)

            output_file.write(f'{filename} {save_path}\n')
    print('Processing complete.')
        # 直接使用npy保存深度图
        
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0 # 归一化到0-255
        # depth = depth.astype(np.uint8)
        
       