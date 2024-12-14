# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

# This file is partly inspired from BTS (https://github.com/cleinc/bts/blob/master/pytorch/bts_dataloader.py); author: Jin Han Lee

import os
import random

import numpy as np
import cv2
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .preprocess import get_white_border
from .transform import Resize,NormalizeImage, PrepareForNet, Crop

import cv2


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s



class NYU(Dataset):
    def __init__(self, config, mode, size=(518, 518)):
        self.config = config

        self.mode = mode
        self.size = size
        
        if mode == "train":
            with open(config.filenames_file, 'r') as f:
                self.filenames = f.readlines()
        elif mode == "eval":
            with open(config.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()

        net_w, net_h = size
     
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
        
    def __getitem__(self, index):
        sample_path = self.filenames[index]

        if self.mode == 'train':
            image_path = os.path.join(
                        self.config.data_path, remove_leading_slash(sample_path.split()[0]))
            depth_path = os.path.join(
                        self.config.gt_path, remove_leading_slash(sample_path.split()[1]))
            
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
    
            w, h = image.size

            # Avoid blank boundaries due to pixel registration?
            # Train images have white border. Test images have black border.
            if self.config.avoid_boundary:
                # We just crop and pad again with reflect padding to original size
                # original_size = image.size
                crop_params = get_white_border(np.array(image, dtype=np.uint8))
                image = image.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
                depth_gt = depth_gt.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))

                # Use reflect padding to fill the blank
                image = np.array(image)
                image = np.pad(image, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right), (0, 0)), mode='reflect')
                image = Image.fromarray(image)

                depth_gt = np.array(depth_gt)
                depth_gt = np.pad(depth_gt, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
                depth_gt = Image.fromarray(depth_gt)

            if self.config.do_random_rotate and (self.config.aug):
                    random_angle = (random.random() - 0.5) * 2 * self.config.degree
                    image = self.rotate_image(image, random_angle)
                    depth_gt = self.rotate_image(
                        depth_gt, random_angle, flag=Image.NEAREST)
            # 
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 1000.0 # nyu dataset depth in mm

            # 打印处理后的图像数据
            # print(f"Image data (after processing): {image}")
            # print(f"Depth data (after processing): {depth_gt}")

             # 训练集数据增强
            if self.config.aug and (self.config.random_crop):
                    image, depth_gt = self.random_crop(
                        image, depth_gt, self.config.input_height, self.config.input_width)

            if self.config.aug and self.config.random_translate:
                    # print("Random Translation!")
                    image, depth_gt = self.random_translate(image, depth_gt, self.config.max_translation)
            
            image, depth_gt = self.train_preprocess(image, depth_gt)

            

        elif self.mode == 'eval':
            image_path = os.path.join(
                        self.config.data_path_val, remove_leading_slash(sample_path.split()[0]))
            depth_path = os.path.join(
                        self.config.gt_path_val, remove_leading_slash(sample_path.split()[1]))
            
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            w, h = image.size
            # Avoid blank boundaries due to pixel registration?
            # Train images have white border. Test images have black border.
            if self.config.avoid_boundary:
                # We just crop and pad again with reflect padding to original size
                # original_size = image.size
                crop_params = get_white_border(np.array(image, dtype=np.uint8))
                image = image.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
                depth_gt = depth_gt.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))

                # Use reflect padding to fill the blank
                image = np.array(image)
                image = np.pad(image, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right), (0, 0)), mode='reflect')
                image = Image.fromarray(image)

                depth_gt = np.array(depth_gt)
                depth_gt = np.pad(depth_gt, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
                depth_gt = Image.fromarray(depth_gt)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 1000.0 # nyu dataset depth in mm

        sample = self.transform({'image': image, 'depth': depth_gt})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        sample['valid_mask'] = (torch.isnan(sample['depth']) == 0)
        sample['depth'][sample['valid_mask'] == 0] = 0
        
        sample['image_path'] = image_path
        
        
        return sample

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth
    
    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth

    def train_preprocess(self, image, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug



    def __len__(self):
        return len(self.filenames)





















