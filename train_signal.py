import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader.nyu import NYU
from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log

parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
# dataset
parser.add_argument('--dataset', default='nyu', choices=['nyu', 'kitti'])
parser.add_argument('--avoid_boundary', default=False, type=bool)
parser.add_argument('--data_path', default='/data2/cw/sync/', type=str)
parser.add_argument('--gt_path', default='/data2/cw/sync/', type=str)
parser.add_argument('--data_path_val', default='/data2/cw/nyu_test/', type=str)
parser.add_argument('--gt_path_val', default='/data2/cw/nyu_test/', type=str)
parser.add_argument('--filenames_file', default='/home/chenwu/DisDepth/dataloader/splits/nyudepthv2_train_files_with_gt.txt', type=str)
parser.add_argument('--filenames_file_eval', default='/home/chenwu/DisDepth/dataloader/splits/nyudepthv2_test_files_with_gt.txt', type=str)
parser.add_argument('--input_height', default=480, type=int)
parser.add_argument('--input_width', default=640, type=int)
parser.add_argument('--aug', default=False, type=bool)
parser.add_argument('--random_crop', default=True, type=bool)
parser.add_argument('--random_translate', default=True, type=bool)
parser.add_argument('--do_random_rotate', default=True, type=bool)

# Training related arguments
parser.add_argument('--img_size', default=518, type=int)
parser.add_argument('--min_depth', default=0.001, type=float)
parser.add_argument('--max_depth', default=10, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=16, type=int)
parser.add_argument('--lr', default=0.000005, type=float) #  0.0002
parser.add_argument('--save_path', default="/data2/cw/dinov2_nyu/", type=str)

def main():
    args = parser.parse_args()

    warnings.simplefilter('ignore', np.RankWarning)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    all_args = vars(args)
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    size = (args.img_size, args.img_size)
    if args.dataset == 'nyu':
        trainset = NYU(args, 'train', size=size)
    else:
        raise NotImplementedError

    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, shuffle=True)

    if args.dataset == 'nyu':
        valset = NYU(args, 'eval', size=size)
    else:
        raise NotImplementedError

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, shuffle=False)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    model.cuda()

    criterion = SiLogLoss().cuda()

    optimizer = AdamW([
        {'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
        {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}
    ], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    total_iters = args.epochs * len(trainloader)

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}

    for epoch in range(args.epochs):
        logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
        logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                    'log10: {:.3f}, silog: {:.3f}'.format(
                        epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
                        previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))

        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()

            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)

            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i

            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            writer.add_scalar('train/loss', loss.item(), iters)

            if i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))

        model.eval()

        results = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0, 'log10': 0, 'silog': 0}
        nsamples = 0

        for i, sample in enumerate(valloader):

            img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]

            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)

            if valid_mask.sum() < 10:
                continue

            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1

        for k in results.keys():
            results[k] /= nsamples

        logger.info('==========================================================================================')
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple(results.values())))
        logger.info('==========================================================================================')

        for name, metric in results.items():
            writer.add_scalar(f'eval/{name}', metric, epoch)

        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], results[k])
            else:
                previous_best[k] = min(previous_best[k], results[k])

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))

if __name__ == '__main__':
    main()