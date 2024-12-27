import argparse
import logging
import os
import random
import warnings
import pprint
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tinyvit.dpt import TinyVitDpt

from dataset.nyu2 import NYU
from dataset.DIML import DIML
from midas_loss import ScaleAndShiftInvariantLoss
from util.utils import init_log
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')


parser.add_argument('--img-size', default=224, type=int)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--batch_size', default=32,type=int)
parser.add_argument('--lr', default=0.000008, type=float) # 调一调学习率是否可以加快收敛呢？ 0.000005 0.000008

parser.add_argument('--save-path', default='exp/tinyvit',type=str)
parser.add_argument('--pretrained-from', default='/home/chenwu/DisDepth/checkpoints/tiny_vit_5m_22kto1k_distill.pth', type=str)

parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
def main():
    args = parser.parse_args()

    
    # 确保训练过程的随机性可控，实验结果具有可复现性。
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)  # Numpy module.
    # random.seed(args.seed)  # Python random module.
    # torch.manual_seed(args.seed)

    warnings.simplefilter('ignore', np.RankWarning)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    all_args = vars(args)
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    # dataset, dataloader
    size = (args.img_size, args.img_size)
    trainset = DIML('dataset/splits/DIML/filtered_train_files.txt', 'train', size=size)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, num_workers=16,shuffle=True)

    # valset = NYU('dataset/splits/nyu/val.txt', 'val', size=size)
    # valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4)


    model = TinyVitDpt().cuda()
    # model = torch.nn.DataParallel(model)


    # if args.pretrained_from:
    #     model.pretrained.load_state_dict(torch.load(args.pretrained_from)["model"])
        # model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    

    criterion = ScaleAndShiftInvariantLoss().cuda()
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    total_iters = args.epochs * len(trainloader)

    for epoch in range(args.epochs):
        
        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            img, depth ,valid_mask = sample['image'].cuda(), sample['depth'].cuda(),sample['valid_mask'].cuda() #valid_mask和 depth是对应的，depth>0的地方是1，否则是0

            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)
            # depth = depth.squeeze(1)
            with torch.no_grad():
                depth = depth_anything(img)
            loss = criterion(pred, depth, valid_mask)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iters = epoch * len(trainloader) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            writer.add_scalar('train/loss', loss.item(), iters)

            if i % 100 == 0:
                logger.info('epoch: {}/{}'.format(epoch, args.epochs))
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))

            
        checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))


if __name__ == '__main__':
    main()


