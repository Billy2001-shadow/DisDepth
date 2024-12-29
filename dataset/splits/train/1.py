# def truncate_file(file_path, line_number):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()

#     # 保留前 line_number 行
#     lines = lines[:line_number]

#     # 将剩余的行写回到文件中
#     with open(file_path, 'w') as f:
#         f.writelines(lines)

# if __name__ == "__main__":
#     # 替换为你的txt文件路径
#     txt_file_path = "/home/chenwu/DisDepth/dataset/splits/train/indoor_train.txt"
#     # 设置要保留的行数
#     line_number = 48345
#     truncate_file(txt_file_path, line_number)


import os
import numpy as np

def find_min_max(file_path):
    min_value = float('inf')
    max_value = float('-inf')

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 拆分路径并过滤掉多余的空格
        parts = line.strip().split()
        
        # 确保每行有两个路径
        if len(parts) != 2:
            print(f"Invalid line format: {line.strip()}")
            continue
        
        img_path, depth_path = parts
        
        # 读取 .npy 文件中的深度图数据
        depth = np.load(depth_path)
        
        # 更新最小值和最大值
        min_value = min(min_value, depth.min())
        max_value = max(max_value, depth.max())

    return min_value, max_value

if __name__ == "__main__":
    # 替换为你的txt文件路径
    txt_file_path = "/home/chenwu/DisDepth/dataset/splits/train/indoor_train.txt"
    min_value, max_value = find_min_max(txt_file_path)
    print(f"Minimum value: {min_value}")
    print(f"Maximum value: {max_value}")