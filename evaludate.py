import os
import argparse
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))) # 添加当前目录到 PYTHONPATH

import torch
import torch.nn.functional as F

from tinyvit.dpt import TinyVitDpt
from util.metric import recover_metric_depth, compute_errors, RunningAverageDict,align_depth_least_square,submap_align,blend_overlap
from dataset.nyu2 import get_nyud_loader

from dataset.kitti import get_kitti_loader
from dataset.ddad import get_ddad_loader
from dataset.diode import get_diode_loader
from dataset.eth3d import get_eth3d_loader

from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import matplotlib
import cv2
from tqdm import tqdm
from PIL import Image

cmap = matplotlib.colormaps.get_cmap('Spectral_r')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate for Relative Depth Estimation')
    
    parser.add_argument('--input_height', default=224, type=int)
    parser.add_argument('--input_width', default=224, type=int) # 输入到网络中的size
    parser.add_argument('--min_depth', default=0.001, type=float)
    parser.add_argument('--max_depth', default=10, type=float)
    
    parser.add_argument('--nyu_eval_fileslist', default='dataset/splits/val/nyu_val.txt', type=str) 
    parser.add_argument('--kitti_eval_fileslist', default='dataset/splits/val/kitti_val.txt', type=str) 
    # parser.add_argument('--diode_eval_fileslist', default='dataset/splits/val/diode_indoor_val.txt', type=str) 
    parser.add_argument('--diode_eval_fileslist', default='dataset/splits/val/diode_outdoor_val.txt', type=str) 
    # parser.add_argument('--eth3d_eval_fileslist', default='dataset/splits/val/eth3d_indoor_val.txt', type=str) 
    parser.add_argument('--eth3d_eval_fileslist', default='dataset/splits/val/eth3d_outdoor_val.txt', type=str) 




    parser.add_argument('--pretrained-from', type=str, default='exp/tinyvit/latest_epoch_39.pth')  # TinyVit_relative_12_24_08_55

    parser.add_argument('--eval_datasets', nargs='+', default=['KITTI'], type=str, help='List of datasets') #  ['KITTI','NYU','DIODE','ETH3D']


    args = parser.parse_args()
    

    model_configs = {
        '5m_224':  {'embed_dims': [64, 128, 160, 320], 'features': 64, 'in_channels':[128, 160, 320,320],'out_channels': [48, 96, 192, 384],'num_heads':[2, 4, 5, 10],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.0},
        '11m_224': {'embed_dims': [64, 128, 256, 448], 'features': 128, 'in_channels':[128, 256, 448,448],'out_channels': [96, 192, 384, 768],'num_heads':[2, 4, 8, 14],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.1},
        '21m_224': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[7, 7, 14, 7],'drop_path_rate':0.2},
        '21m_384': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[12, 12, 24, 12],'drop_path_rate':0.1},
        '21m_512': {'embed_dims': [96, 192, 384, 576], 'features': 128, 'in_channels':[192, 384, 576,576],'out_channels': [96, 192, 384, 768],'num_heads':[3, 6, 12, 18],'window_sizes':[16, 16, 32, 16],'drop_path_rate':0.1},
    }

    model = TinyVitDpt(config=args, **model_configs['5m_224'],use_bn=True)

    if args.pretrained_from:
        model.load_state_dict(torch.load(args.pretrained_from, map_location='cpu')['model'])

    if 'KITTI' in args.eval_datasets:
        kittiloader = get_kitti_loader(data_dir_root=args.kitti_eval_fileslist, size =(args.input_height, args.input_width))
    if 'NYU' in args.eval_datasets:
        nyuvalloader = get_nyud_loader(data_dir_root=args.nyu_eval_fileslist,size =(args.input_height, args.input_width))
    if 'DIODE' in args.eval_datasets:
        diodevalloader = get_diode_loader(data_dir_root=args.diode_eval_fileslist,size =(args.input_height, args.input_width))
    if 'ETH3D' in args.eval_datasets:
        eth3dvalloader = get_eth3d_loader(data_dir_root=args.eth3d_eval_fileslist,size =(args.input_height, args.input_width))

   
    print("loading model {}".format(args.pretrained_from))
    print("epoch = ",torch.load(args.pretrained_from)["epoch"])
   
    state_dict=torch.load(args.pretrained_from)["model"]
    model.load_state_dict(state_dict)
    
    model.cuda().eval() # 在加载完参数之后再设置为eval模式
    with torch.no_grad():
        if 'NYU' in args.eval_datasets:
            metrics =RunningAverageDict()
            with open('d1_results.txt', 'a') as f:
                for batch_idx, sample in tqdm(enumerate(nyuvalloader),total=len(nyuvalloader)):
                    img, depth = sample['image'].cuda().float(), sample['depth'].cuda()[0]  # torch.Size([1, 3, 224, 224]) depth torch.Size([1, 1, 480, 640])
                    
                    pred = model(img) # torch.Size([1,1, 224, 224])     
                    pred = F.interpolate(pred, depth.shape[-2:], mode='bilinear',align_corners=True)  # nearest  bilinear | bicubic torch.Size([1, 1, 480, 640])
                    pred = pred.squeeze().cpu().numpy()   # torch.Size([480, 640]) 
                    depth = depth.squeeze().cpu().numpy() # torch.Size([480, 640]) 
                    
                    # save_pred = pred.reshape(1,pred.shape[0],pred.shape[1])
                    # save_pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255.0 
                    # save_pred = save_pred.astype(np.uint8)
                    # save_pred = (cmap(save_pred)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                    # image_path = sample['image_path'][0]
                    # save_path = image_path.replace(".jpg",".png")
                    # cv2.imwrite(save_path, save_pred)

                    valid_mask = (depth > 0.01) & (depth < 10)

                    pred = align_depth_least_square(pred,depth,valid_mask,10) # 10最大深度    

                    # 对(426,520)的区域进行测试  
                    epsilon = 1e-8
                    depth = depth + epsilon
                    pred = pred + epsilon   
                    thresh = np.maximum((depth / pred), (pred / depth)) 
                    d1_mask = thresh < 1.25

                    valid_sum = valid_mask.sum()
                    d1 = (d1_mask & valid_mask).sum() / valid_sum

                    error_mask = ((thresh >= 1.25) & valid_mask)
                    image_path = sample['image_path'][0]

                    original_img  = np.array(Image.open(image_path))
                    cropped_img = original_img[45:471, 41:601, :]

                    # 应用半透明红色高亮（alpha=0.5）
                    color_layer = np.zeros_like(cropped_img)
                    color_layer[error_mask] = (0, 255, 0) # 红色
                    alpha = 0.7
                    visualized  = (cropped_img * (1 - alpha) + color_layer * alpha).astype(np.uint8)

                    save_path = image_path.replace("rgb","error_combined")
                    combined = Image.new("RGB", (560*2, 426))
                    # root_path = "/home/chenwu/DisDepth/vis/nyu/error"
                    # save_name = (os.path.splitext(os.path.basename(image_path))[0]).replace("rgb","error_rgb") + ".png"
                    # save_path = os.path.join(root_path,save_name)
                    # Image.fromarray(visualized).save(save_path)
                    # print(f"可视化结果已保存至：{save_path}")
                    # 将 image_path 和 d1 写入到文件
                    # f.write(f"{image_path} {d1:.4f}\n")  # 写入格式: 路径 d1值
                    

                    metrics.update(compute_errors(depth[valid_mask], pred[valid_mask])) # 第二步1.0 / pred[valid_mask]
        
                metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
                print(f"Metrics: {metrics}")

        if 'KITTI' in args.eval_datasets:
            metrics =RunningAverageDict()
            with open('d1_results_kitti.txt', 'a') as f:
                for batch_idx, sample in tqdm(enumerate(kittiloader),total=len(kittiloader)):
                    sub_image1 = sample['sub_image1'].cuda().float()
                    sub_image2 = sample['sub_image2'].cuda().float()
                    sub_image3 = sample['sub_image3'].cuda().float()
                    sub_image4 = sample['sub_image4'].cuda().float()
                    sub_image5 = sample['sub_image5'].cuda().float()
                    sub_image6 = sample['sub_image6'].cuda().float()
                    depth = sample['depth'].cuda()[0] 
                    # (218, 1153)
                    depth = depth.squeeze().cpu().numpy()

                    pred1 = model(sub_image1) # torch.Size([1,1, 224, 224])     
                    pred2 = model(sub_image2) # torch.Size([1,1, 224, 224])
                    pred3 = model(sub_image3) # torch.Size([1,1, 224, 224])
                    pred4 = model(sub_image4) # torch.Size([1,1, 224, 224])
                    pred5 = model(sub_image5) # torch.Size([1,1, 224, 224])
                    pred6 = model(sub_image6) # torch.Size([1,1, 224, 224])
                    pred1 = pred1.squeeze().cpu().numpy()
                    pred2 = pred2.squeeze().cpu().numpy()
                    pred3 = pred3.squeeze().cpu().numpy()
                    pred4 = pred4.squeeze().cpu().numpy()
                    pred5 = pred5.squeeze().cpu().numpy()
                    pred6 = pred6.squeeze().cpu().numpy()

                    # 右边子图对齐左边子图
                    pred2 = submap_align(pred2, pred1)
                    pred3 = submap_align(pred3, pred2)
                    pred4 = submap_align(pred4, pred3)
                    pred5 = submap_align(pred5, pred4)
                    pred6 = submap_align(pred6, pred5)

                    # 创建一个224*1184的全零np数组
                    pred = np.zeros((224, 1184))
                    pred = blend_overlap(pred, pred1, pred2, pred3, pred4, pred5, pred6)
                    # save_depth_as_image(pred, f"pred_{batch_idx}.png")
                    valid_mask = (depth > 0.01) & (depth < 80)
                    zoom_factors = (depth.shape[0] / pred.shape[0], depth.shape[1] / pred.shape[1])
                    pred = zoom(pred, zoom_factors, order=1)
                    
                    # (218, 1153)
                    pred = align_depth_least_square(pred, depth, valid_mask, 80)

                    # save_pred = pred.reshape(1,pred.shape[0],pred.shape[1])
                    # save_pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255.0 
                    # save_pred = save_pred.astype(np.uint8)
                    # save_pred = (cmap(save_pred)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                    # image_path = sample['image_path'][0]
                    # save_path = image_path.replace(".png","_pred.png")
                    # cv2.imwrite(save_path, save_pred)

                    epsilon = 1e-8
                    depth = depth + epsilon
                    pred = pred + epsilon   
                    thresh = np.maximum((depth / pred), (pred / depth)) 
                    d1_mask = thresh < 1.25

                    valid_sum = valid_mask.sum()
                    d1 = (d1_mask & valid_mask).sum() / valid_sum

                    error_mask = ((thresh >= 1.25) & valid_mask)
                    image_path = sample['image_path'][0]
                    original_img  = np.array(Image.open(image_path))
                    
                   
                     # 应用半透明红色高亮（alpha=0.5）
                    color_layer = np.zeros_like(cropped_img)
                    color_layer[error_mask] = (255, 0, 0) # 红色色
                    alpha = 0.3
                    visualized  = (cropped_img * (1 - alpha) + color_layer * alpha).astype(np.uint8)
                    save_path = image_path.replace(".png","_error.jpg")
                    Image.fromarray(visualized).save(save_path)
                    # # 将 image_path 和 d1 写入到文件
                    # f.write(f"{image_path} {d1:.4f}\n")  # 写入格式: 路径 d1值

                    metrics.update(compute_errors(depth[valid_mask], pred[valid_mask])) # 第二步1.0 / pred[valid_mask]
                metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
                print(f"Metrics: {metrics}")

        if 'DIODE' in args.eval_datasets:
            metrics =RunningAverageDict()
            with open('d1_results_diode.txt', 'a') as f:
                for batch_idx, sample in tqdm(enumerate(diodevalloader),total=len(diodevalloader)):
                    img, depth = sample['image'].cuda().float(), sample['depth'].cuda()[0]  # torch.Size([1, 3, 224, 224]) depth torch.Size([1, 1, 480, 640])
                    eval_mask = sample['eval_mask']

                    pred = model(img) # torch.Size([1,1, 224, 224])     
                    pred = F.interpolate(pred, depth.shape[-2:], mode='bilinear',align_corners=True)  # nearest  bilinear | bicubic torch.Size([1, 1, 480, 640])
                    pred = pred.squeeze().cpu().numpy()   # (768, 1024) 
                    depth = depth.squeeze().cpu().numpy() # (768, 1024)
                    eval_mask = eval_mask.squeeze().cpu().numpy()
              
                    valid_mask = (depth > 0.6) & (depth < 50) # 室内50米 室外300米 # torch.Size([768, 1024])
                    valid_mask = np.logical_and(eval_mask,valid_mask)

                    pred = align_depth_least_square(pred,depth,valid_mask,50) # 10最大深度    

                    epsilon = 1e-8
                    depth = depth + epsilon
                    pred = pred + epsilon   
                    thresh = np.maximum((depth / pred), (pred / depth)) 
                    d1_mask = thresh < 1.25

                    valid_sum = valid_mask.sum()
                    d1 = (d1_mask & valid_mask).sum() / valid_sum
                    image_path = sample['image_path']
                    # # 将 image_path 和 d1 写入到文件
                    f.write(f"{image_path} {d1:.4f}\n")  # 写入格式: 路径 d1值

                    metrics.update(compute_errors(depth[valid_mask], pred[valid_mask])) # 第二步1.0 / pred[valid_mask]
        
                metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
                print(f"Metrics: {metrics}")


        if 'ETH3D' in args.eval_datasets:
            metrics =RunningAverageDict()
            with open('d1_results_eth3d.txt', 'a') as f:
                for batch_idx, sample in tqdm(enumerate(eth3dvalloader),total=len(eth3dvalloader)):
                    img, depth = sample['image'].cuda().float(), sample['depth'].cuda()[0]  # torch.Size([1, 3, 224, 224]) depth torch.Size([1, 1, 480, 640])
                    
                    pred = model(img) # torch.Size([1,1, 224, 224])     
                    pred = F.interpolate(pred, depth.shape[-2:], mode='bilinear',align_corners=True)  # nearest  bilinear | bicubic torch.Size([1, 1, 480, 640])
                    pred = pred.squeeze().cpu().numpy()   # (768, 1024) 
                    depth = depth.squeeze().cpu().numpy() # (768, 1024)
                    
                    valid_mask = (depth > 0.0001) & (depth < 80) # 室内50米 室外300米 # torch.Size([768, 1024])


                    # 计算 valid_mask 为 True 的数量
                    valid_pixel_count = np.sum(valid_mask)

                    # 获取 depth 数组的总像素数
                    total_pixel_count = depth.size  # 或者使用 valid_mask.size，结果应该是一样的

                    # 计算 valid_mask 占 depth 总数的比例
                    valid_ratio = valid_pixel_count / total_pixel_count

                    print(f"valid_mask 占 depth 总数的比例: {valid_ratio:.4f}")


                    pred = align_depth_least_square(pred,depth,valid_mask,80) # 10最大深度    


                    # epsilon = 1e-8
                    # depth = depth + epsilon
                    # pred = pred + epsilon   
                    # thresh = np.maximum((depth / pred), (pred / depth)) 
                    # d1_mask = thresh < 1.25

                    # valid_sum = valid_mask.sum()
                    # d1 = (d1_mask & valid_mask).sum() / valid_sum
                    # image_path = sample['image_path']
                    # # 将 image_path 和 d1 写入到文件
                    # f.write(f"{image_path} {d1:.4f}\n")  # 写入格式: 路径 d1值

                    metrics.update(compute_errors(depth[valid_mask], pred[valid_mask])) # 第二步1.0 / pred[valid_mask]
        
                metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
                print(f"Metrics: {metrics}")
        