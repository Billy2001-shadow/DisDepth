import argparse
import cv2
import matplotlib
import numpy as np
import os


if __name__ == '__main__':
    npy_path = '/data2/cw/Relative_depth/NYU/pseudo_depth/living_room_0038_out_71.npy'
    outdir = './vis_depth'

    os.makedirs(outdir, exist_ok=True)
   
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth = np.load(npy_path)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    
    save_name = os.path.splitext(os.path.basename(npy_path))[0] + '.png'

    cv2.imwrite(os.path.join(outdir, save_name), depth)