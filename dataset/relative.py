import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, DepthToDisparity

from PIL import Image

# ['BlendMVS', 'Holopix50k', 'HRWSI','MegaDepth','ReDWeb']
class Relative(Dataset):
    def __init__(self, filelist_path, mode, size=(224, 224)):
        
        self.mode = mode
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = [line.strip() for line in f if line.strip()]  # 过滤掉空行
            
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):

        img_path = self.filelist[item].split()[0]
        depth_path = self.filelist[item].split()[1]

        sample = {}
        

        image = Image.open(img_path)    
        image = np.asarray(image, dtype=np.float32) / 255.0
        sample = self.transform({'image': image})
        
        depth = np.load(depth_path).astype('float32')
    
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(depth)
        
      
        sample['image_path'] = img_path

    
        return sample

    def __len__(self):
        return len(self.filelist)
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    trainset = Relative('dataset/splits/relative_depth_train.txt', 'train',(224,224))
    print(len(trainset))
    sample = trainset[0]
    print(sample['image'].shape, sample['depth'].shape, sample['image_path'])

    trainloader = DataLoader(trainset, 32, pin_memory=True, num_workers=16,shuffle=True)

    for i, sample in enumerate(trainloader):
        img, depth  = sample['image'].cuda(), sample['depth'].cuda() 
        img_path = sample['image_path']