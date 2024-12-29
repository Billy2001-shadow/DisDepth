import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ColorJitter, GaussianBlur,Resize,Normalize,ToTensor

# from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, DepthToDisparity
import torchvision.transforms as T

from PIL import Image

# ['BlendMVS', 'Holopix50k', 'HRWSI','MegaDepth','ReDWeb']
class Relative(Dataset):
    def __init__(self, filelist_path, mode, size=(224, 224)):
        
        self.mode = mode
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = [line.strip() for line in f if line.strip()]  # 过滤掉空行
            
        
       
        self.transform = Compose([
            # 色彩抖动：随机调整亮度、对比度、饱和度和色相
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # 高斯模糊：使用3x3的卷积核
            GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            Resize(size),
            ToTensor(),  # 将图像转换为张量(0-255归一化为0-1,并转换为C*H*W)
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __getitem__(self, item):

        img_path = self.filelist[item].split()[0]
        depth_path = self.filelist[item].split()[1]

        sample = {}
        
         # 使用 PIL 直接读取图像
        image = Image.open(img_path).convert('RGB')  # 使用PIL读取并转换为RGB格式
        
        # 应用数据增强和预处理
        image = self.transform(image)
        sample['image'] = image

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load depth image: {depth_path}")
        depth = depth.astype(np.float32)
    
        
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