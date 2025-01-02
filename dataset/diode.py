import numpy as np
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose


from dataset.transform import Resize, NormalizeImage, PrepareForNet


class DIODE(Dataset):
    def __init__(self, filelist_path, size=(224, 224)):
        
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=False, # ZeroShot,NYUD数据集不参与训练
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = np.load(depth_path)
        
        sample = self.transform({'image': image, 'depth': depth})
        
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        # sample['depth'] = sample['depth'] / 1000.0  # convert in meters
        
        if 'indoor' in img_path:
            sample['valid_mask'] = (sample['depth'] > 0) & (sample['depth'] <= 10)
        elif 'outdoor' in img_path:
            sample['valid_mask'] = (sample['depth'] > 0) & (sample['depth'] <= 80)


        sample['image_path'] = img_path
        
        return sample

    def __len__(self):
        return len(self.filelist)




def get_diode_loader(data_dir_root, size=(224, 224)):
    dataset = DIODE(data_dir_root, size)
    return DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True)



# import numpy as np
# import torch
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms


# class ToTensor(object):
#     def __init__(self):
#         self.normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         # self.normalize = lambda x : x
#         self.resize = transforms.Resize((224,224))

#     def __call__(self, sample):
#         image, depth = sample['image'], sample['depth']
#         image = self.to_tensor(image)
#         image = self.normalize(image)
#         depth = self.to_tensor(depth)

#         image = self.resize(image)

#         return {'image': image, 'depth': depth, 'dataset': "diode"}

#     def to_tensor(self, pic):

#         if isinstance(pic, np.ndarray):
#             img = torch.from_numpy(pic.transpose((2, 0, 1)))
#             return img

#         #         # handle PIL Image
#         if pic.mode == 'I':
#             img = torch.from_numpy(np.array(pic, np.int32, copy=False))
#         elif pic.mode == 'I;16':
#             img = torch.from_numpy(np.array(pic, np.int16, copy=False))
#         else:
#             img = torch.ByteTensor(
#                 torch.ByteStorage.from_buffer(pic.tobytes()))
#         # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
#         if pic.mode == 'YCbCr':
#             nchannel = 3
#         elif pic.mode == 'I;16':
#             nchannel = 1
#         else:
#             nchannel = len(pic.mode)
#         img = img.view(pic.size[1], pic.size[0], nchannel)

#         img = img.transpose(0, 1).transpose(0, 2).contiguous()

#         if isinstance(img, torch.ByteTensor):
#             return img.float()
#         else:
#             return img


# class DIODE(Dataset):
#     def __init__(self, filenames_file):
#         # import glob
#         # image paths are of the form <data_dir_root>/scene_#/scan_#/*.png
#         # self.image_files = glob.glob(
#         #     os.path.join(data_dir_root, '*', '*', '*.png'))
#         # self.depth_files = [r.replace(".png", "_depth.npy")
#         #                     for r in self.image_files]
#         # self.depth_mask_files = [
#         #     r.replace(".png", "_depth_mask.npy") for r in self.image_files]

#         with open(filenames_file, 'r') as f:
#                 self.filenames = f.readlines()

#         self.transform = ToTensor()

#     def __getitem__(self, idx):
#         sample_path = self.filenames[idx]
        
#         image_path = sample_path.split()[0]
#         depth_path = sample_path.split()[1]

        
#         image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
#         depth = np.load(depth_path)  # in meters
        
#         # depth[depth > 8] = -1
#         # depth = depth[..., None]
#         sample = dict(image=image, depth=depth)

#         # return sample
#         sample = self.transform(sample)

#         if idx == 0:
#             print(sample["image"].shape)

#         return sample

#     def __len__(self):
#         return len(self.filenames)


# def get_diode_loader(data_dir_root, batch_size=1, **kwargs):
#     dataset = DIODE(data_dir_root)
#     return DataLoader(dataset, batch_size, **kwargs)

# get_diode_loader(data_dir_root="datasets/diode/val/outdoor")

