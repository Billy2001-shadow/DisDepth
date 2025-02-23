import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self,size=(224,224)):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.normalize = lambda x : x
        # 和Resize的分辨率有很大的关系
        self.resize = transforms.Resize(size)
        

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        sub_image1 = image[:, 0:224, :]
        sub_image2 = image[:, 192:416, :]
        sub_image3 = image[:, 384:608, :]
        sub_image4 = image[:, 576:800, :]
        sub_image5 = image[:, 768:992, :]
        sub_image6 = image[:, 960:1184, :]
       
        sub_image1 = self.to_tensor(sub_image1)
        sub_image1 = self.normalize(sub_image1)
        sub_image2 = self.to_tensor(sub_image2)
        sub_image2 = self.normalize(sub_image2)
        sub_image3 = self.to_tensor(sub_image3)
        sub_image3 = self.normalize(sub_image3)
        sub_image4 = self.to_tensor(sub_image4)
        sub_image4 = self.normalize(sub_image4)
        sub_image5 = self.to_tensor(sub_image5)
        sub_image5 = self.normalize(sub_image5)
        sub_image6 = self.to_tensor(sub_image6)
        sub_image6 = self.normalize(sub_image6)
        
        # image = self.to_tensor(image)
        # image = self.normalize(image)
        depth = self.to_tensor(depth)

        # image = self.resize(image)
        # return {'image': image, 'depth': depth, 'dataset': "kitti"}
        return {'sub_image1':sub_image1,
                'sub_image2':sub_image2,
                'sub_image3':sub_image3,
                'sub_image4':sub_image4,
                'sub_image5':sub_image5,
                'sub_image6':sub_image6,
                 'depth': depth, 'dataset': "kitti"}

        

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

# 修改，之前的crop是((44, 153, 1197, 371))   现在修改为((44, 152, 1197, 370)) 
class KITTI(Dataset):
    def __init__(self, filenames_file,size=(224,224)):
       
        with open(filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.transform = ToTensor(size)

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        
        image_path = sample_path.split()[0]
        depth_path = sample_path.split()[1]

        # Crop the RGB image (Garg Crop)
        image = Image.open(image_path).convert('RGB')  # 确保是 RGB 格式
        image_height = image.height
        image_width = image.width
        
        crop_size = (0.03594771 * image_width,0.40810811 * image_height,0.96405229 * image_width,0.99189189 * image_height)
        # image = image.crop(crop_size)
        image = image.crop((44, 153, 1197, 371))  # (left, upper, right, lower) # 218, 1153 
        image = image.resize((1184, 224), Image.BILINEAR) # (1185, 224)
        image = np.asarray(image, dtype=np.float32) / 255.0 # (224, 1184, 3) # -65

        # 对depth也进行resize处理，进行最近邻插值
        depth = Image.open(depth_path)
        depth = depth.crop((44, 153, 1197, 371))  # (left, upper, right, lower) # 218, 1153   1242,375
        # depth = depth.crop(crop_size)
        # depth = depth.resize((1184, 224), Image.NEAREST) # (1185, 224) 评估的时候直接插值到218, 1153
        depth = np.asarray(depth, dtype=np.float32) / 256.0 # in meters
        depth = np.expand_dims(depth, axis=2)
        
        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)
        sample['image_path'] = image_path

        return sample

    def __len__(self):
        return len(self.filenames)


def get_kitti_loader(data_dir_root, size=(224, 224)):
    dataset = KITTI(data_dir_root, size)
    return DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True)