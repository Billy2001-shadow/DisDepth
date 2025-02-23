# import numpy as np

# def analyze_npy(file_path):
#     try:
#         # 加载.npy文件
#         data = np.load(file_path)
        
#         # 检查是否是NumPy数组
#         if not isinstance(data, np.ndarray):
#             raise ValueError("文件内容不是NumPy数组")
        
#         # 计算统计值
#         max_val = np.max(data)
#         min_val = np.min(data)
#         median_val = np.median(data)
        
#         # 统计小于50的值的数量
#         count_less_than_50 = np.sum(data < 80)
#         total_elements = data.size
#         proportion_less_than_50 = count_less_than_50 / total_elements
        
#         print(f"最大值: {max_val:.4f}")
#         print(f"最小值: {min_val:.4f}")
#         print(f"中位数: {median_val:.4f}")
#         print(f"小于50的数量: {count_less_than_50}")
#         print(f"小于50的比例: {proportion_less_than_50:.4%}")
        
#     except FileNotFoundError:
#         print(f"错误：文件 '{file_path}' 不存在")
#     except Exception as e:
#         print(f"发生错误: {str(e)}")

# # 替换为你的.npy文件路径
# file_path = "/data2/cw/Zero_shot_Datasets/diode_val/outdoors/scene_00022/scan_00193/00022_00193_outdoor_070_010_depth.npy"
# analyze_npy(file_path)


# def add_mask_path_to_txt(input_txt_path, output_txt_path):
#     try:
#         with open(input_txt_path, 'r') as infile, open(output_txt_path, 'w') as outfile:
#             for line in infile:
#                 # 去掉行尾的换行符
#                 line = line.strip()
#                 # 分割路径，获取 depth 路径
#                 paths = line.split()
#                 if len(paths) == 2:  # 如果当前行包含图片路径和depth路径
#                     depth_path = paths[1]
#                     # 根据 depth 路径构造 mask 路径
#                     mask_path = depth_path.replace('_depth.npy', '_depth_mask.npy')
#                     # 将新的路径加到每行末尾
#                     outfile.write(f"{line} {mask_path}\n")
#                 else:
#                     # 如果格式不正确，可以跳过或者根据需要处理
#                     print(f"Skipping invalid line: {line}")
                    
#         print(f"处理完成，输出文件保存为: {output_txt_path}")
#     except Exception as e:
#         print(f"发生错误: {str(e)}")

# # 输入文本文件路径和输出文件路径
# input_txt_path = "/home/chenwu/DisDepth/dataset/splits/val/diode_outdoor_val.txt"  # 替换为实际的输入文件路径
# output_txt_path = "/home/chenwu/DisDepth/dataset/splits/val/diode_outdoor_mask_val.txt"  # 替换为实际的输出文件路径

# add_mask_path_to_txt(input_txt_path, output_txt_path)

# import cv2
# import numpy as np

# def analyze_depth_image(file_path):
#     # 读取深度图像，假设深度图是PNG格式，且每个像素的值表示深度
#     depth_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # 读取为原始图像

#     if depth_img is None:
#         print(f"错误：无法读取图像 '{file_path}'")
#         return

#     # 将图像转换为浮动格式（假设深度图是16位或者其他需要转换的格式）
#     depth_img = depth_img.astype(np.float32)

#     # 计算最大值、最小值、中位数
#     max_val = np.max(depth_img)
#     min_val = np.min(depth_img)
#     median_val = np.median(depth_img)

#     # 输出分析结果
#     print(f"深度图最大值: {max_val:.4f}")
#     print(f"深度图最小值: {min_val:.4f}")
#     print(f"深度图中位数: {median_val:.4f}")

#     # 统计深度图有效像素的范围（大于零的有效深度）
#     valid_depth_pixels = depth_img[depth_img > 0]
#     valid_max = np.max(valid_depth_pixels)
#     valid_min = np.min(valid_depth_pixels)
#     valid_median = np.median(valid_depth_pixels)

#     print(f"有效深度图最大值: {valid_max:.4f}")
#     print(f"有效深度图最小值: {valid_min:.4f}")
#     print(f"有效深度图中位数: {valid_median:.4f}")

# # 输入你的深度图路径
# file_path = "/data2/cw/Zero_shot_Datasets/ETH3D/fine_gt_depth/electro/DSC_9294.png"
# analyze_depth_image(file_path)


# from PIL import Image
# import os

# # 文件路径
# txt_file = '/home/chenwu/DisDepth/dataset/splits/val/kitti_val.txt'  # 替换为你的txt文件路径

# # 用于存储所有图片尺寸的集合
# image_sizes = {}

# # 读取txt文件
# with open(txt_file, 'r') as f:
#     lines = f.readlines()

# # 遍历每行，处理RGB图片路径
# for line in lines:
#     rgb_path, depth_path, label = line.strip().split()  # 每行按空格分割
#     try:
#         # 打开RGB图片
#         img = Image.open(rgb_path)
        
#         # 获取图片的尺寸
#         size = img.size  # (width, height)

#         # 统计每种尺寸出现的次数
#         if size in image_sizes:
#             image_sizes[size] += 1
#         else:
#             image_sizes[size] = 1

#     except Exception as e:
#         print(f"Error processing image {rgb_path}: {e}")

# # 打印所有不同尺寸的统计
# print("Image size statistics:")
# for size, count in image_sizes.items():
#     print(f"Size: {size}, Count: {count}")


# Size: (1242, 375), Count: 487   用garg_crop计算出来是[ 153  371   44 1197]
# Size: (1224, 370), Count: 23                        [ 151  366   43 1180]
# Size: (1238, 374), Count: 23                        [ 152  370   44 1193]
# Size: (1226, 370), Count: 71                        [ 151  366   44 1181]
# Size: (1241, 376), Count: 48                        [ 153  372   44 1196]

# import numpy as np


# gt_height = 376
# gt_width = 1241

# crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
#                                      0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

# print(crop)



