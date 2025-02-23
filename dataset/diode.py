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
        self.resize = transforms.Resize(size)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        image = self.resize(image)

        return {'image': image, 'depth': depth, 'dataset': "diode"}

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


class DIODE(Dataset):
    def __init__(self, filenames_file,size=(224,224)):
       
        with open(filenames_file, 'r') as f:
            self.filenames = f.readlines()
        self.transform = ToTensor(size)

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        
        image_path = sample_path.split()[0]
        depth_path = sample_path.split()[1]
        depth_mask_path = sample_path.split()[2]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        # (768, 1024, 1)
        depth = np.load(depth_path)  # in meters
        eval_mask = np.load(depth_mask_path)

        # (768, 1024, 1, 1)
        # depth = np.expand_dims(depth, axis=2)
        
        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)
        sample['image_path'] = image_path
        sample['eval_mask'] = eval_mask
        return sample

    def __len__(self):
        return len(self.filenames)


def get_diode_loader(data_dir_root, size=(224, 224)):
    dataset = DIODE(data_dir_root, size)
    return DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True)



