import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
import torchvision.transforms as T


# ['BlendMVS', 'Holopix50k', 'HRWSI','MegaDepth','ReDWeb']
class DepthDataLoader(Dataset):
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
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    
    def __getitem__(self, item):

        img_path = self.filelist[item].split()[0]
        depth_path = self.filelist[item].split()[1]

        sample = {}
        
         # 使用 PIL 直接读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load depth image: {depth_path}")
        depth = depth.astype(np.float32)

        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])

        sample['image_path'] = img_path

    
        return sample

    def __len__(self):
        return len(self.filelist)

def get_train_loader(config , mode ):
    size = (config.input_height,config.input_width)
    dataset = DepthDataLoader(config.filelist_path,mode, size)
    return DataLoader(dataset, config.batch_size ,shuffle=True,num_workers= config.workers,pin_memory=True)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    trainset = DepthDataLoader('dataset/splits/relative_depth_train.txt', 'train',(224,224))
    print(len(trainset))
    sample = trainset[0]
    print(sample['image'].shape, sample['depth'].shape, sample['image_path'])

    trainloader = DataLoader(trainset, 32, pin_memory=True, num_workers=16,shuffle=True)

    for i, sample in enumerate(trainloader):
        img, depth  = sample['image'].cuda(), sample['depth'].cuda() 
        img_path = sample['image_path']