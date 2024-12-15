import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet,Crop
import random
from PIL import Image
import numpy as np

class NYU(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        
        # 自定义训练集的数据增强是否开启
        self.do_random_rotate = True
        self.train_process = True
        self.degree = 2.5

        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False, # False时 only resize image
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        image = Image.open(img_path)
        depth_gt = Image.open(depth_path)

        image = image.crop((43, 45, 608, 472))  # crop后size=(416,544) 网络输入518×518
        depth_gt = depth_gt.crop((43, 45, 608, 472))
        
       
        # 训练集进行数据增强
        if self.mode == 'train':
            if self.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)

            if self.train_process:
                image, depth_gt = self.train_preprocess(image, depth_gt)

        elif self.mode == 'val':
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)

        sample = self.transform({'image': image, 'depth': depth_gt})                 # (3, 518, 686)  (518, 686)
        
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['depth'] = sample['depth'] / 1000.0  # convert in meters
        
        sample['valid_mask'] = sample['depth'] > 0
        
        sample['image_path'] = self.filelist[item].split(' ')[0] # img_path
        
        return sample

    def __len__(self):
        return len(self.filelist)
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result
    

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            # depth_gt = (depth_gt[:, ::-1, :]).copy()
            depth_gt = depth_gt[:, ::-1].copy()  # 对二维数组进行水平翻转

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