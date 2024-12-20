import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class NYU(Dataset):
    def __init__(self, filelist_path, mode, size=(224, 224)):
        
        self.mode = mode
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    
    def __getitem__(self, item):
        img_path = self.filelist[item].split(',')[0]
        depth_path = self.filelist[item].split(',')[1]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        if self.mode == 'train':
            depth = np.load(depth_path)
        else:
            depth = None
        
        
        sample = self.transform({'image': image})
        sample['depth'] = depth

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        # sample['valid_mask'] = (sample['depth'] <= 80) # 这里读的是numpy，相对深度，无法比较真实尺度下的深度
        
        sample['image_path'] = self.filelist[item].split(',')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)