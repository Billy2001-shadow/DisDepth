import os
import numpy as np
import cv2

def npy_to_png(txt_file_path):
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 拆分路径并过滤掉多余的空格
        parts = line.strip().split()
        
        # 确保每行有两个路径
        if len(parts) != 2:
            print(f"Invalid line format: {line.strip()}")
            continue
        
        img_path, npy_path = parts
        
        # 构建输出目录
        base_dir = os.path.dirname(npy_path)
        dataset_dir = os.path.basename(os.path.dirname(base_dir))
        output_dir = os.path.join(os.path.dirname(base_dir), 'depth')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建 PNG 文件路径
        png_filename = os.path.splitext(os.path.basename(npy_path))[0] + '.png'
        png_path = os.path.join(output_dir, png_filename)

        # 读取 .npy 文件中的深度图数据
        depth = np.load(npy_path)

        # 将深度图数据转换为 16 位无符号整数类型
        depth_uint16 = depth.astype(np.uint16)

        # 保存为 PNG 文件
        cv2.imwrite(png_path, depth_uint16)

        print(f"Converted {npy_path} to {png_path}")

if __name__ == "__main__":
    # 替换为你的txt文件路径
    txt_file_path = "/home/chenwu/DisDepth/dataset/splits/train/indoor_train.txt"
    npy_to_png(txt_file_path)