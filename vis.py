# import re

# input_file = "d1_below_0_96_sorted_nyu.txt"
# output_file = "d1_below_0_96_sorted_nyu_vis.txt"

# with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
#     for line in f_in:
#         # 使用正则匹配路径部分
#         match = re.search(r"\['(.*\.)jpg'\] (.*)", line)
#         if match:
#             # 构造新路径并保持指标不变
#             new_path = f"{match.group(1)}png"
#             metric = match.group(2)
#             f_out.write(f"{new_path} {metric}\n")
#         else:
#             f_out.write(line)  # 保留不符合格式的行（可选）


input_file = "/home/chenwu/DisDepth/dataset/splits/val/kitti_val.txt"
output_file = "/home/chenwu/DisDepth/vis/kitti_vis.txt"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        # 保留前两列（从右边分割一次）
        parts = line.strip().rsplit(' ', 2)  # 从右边分割一次
        if len(parts) >= 2:
            new_line = parts[0] + "\n"       # 取前两部分
            fout.write(new_line)
        else:
            fout.write(line)  # 保留格式异常的行（可选）



# import os
# from PIL import Image

# input_txt = "/home/chenwu/DisDepth/vis/kitti_vis.txt"
# # output_dir = "/home/chenwu/DisDepth/vis/combined"
# crop_area = (41, 45, 601, 471)

# # os.makedirs(output_dir, exist_ok=True)

# with open(input_txt, "r") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
        
#         try:
#             rgb_path, depth_path = line.split()
#             base_name = os.path.basename(rgb_path).split(".")[0]
            
#             rgb_image = Image.open(rgb_path)
#             cropped_rgb = rgb_image.crop(crop_area)

#             depth_image = Image.open(depth_path)

#             combined = Image.new("RGB", (560*2, 426))
#             combined.paste(cropped_rgb, (0, 0))
#             combined.paste(depth_image, (560, 0))
#             output_path = rgb_path.replace("rgb","combined")
#             # output_path = os.path.join(output_dir, f"{base_name}_combined.jpg")
#             combined.save(output_path)
#             print(f"Saved: {output_path}")
           
            
#         except Exception as e:
#             print(f"ERROR in {line}: {type(e).__name__} - {str(e)}")


# import os
# from PIL import Image

# # 输入文件路径
# input_txt = "/home/chenwu/DisDepth/vis/nyu/nyu_vis.txt"

# # 有缺陷：rgb_path中必须包含"rgb"这个字符串
# def commbined(rgb_path,depth_path):
#     # 打开图像
#     rgb_img = Image.open(rgb_path)
#     depth_img = Image.open(depth_path)
    
#     # 转换为RGB模式（处理深度图为单通道的情况）
#     if depth_img.mode != "RGB":
#         depth_img = depth_img.convert("RGB")
    
#     # 验证尺寸
#     if rgb_img.size != depth_img.size:
#         print(f"行 {line_num}: 尺寸不匹配（RGB：{rgb_img.size}，深度图：{depth_img.size}），自动调整深度图尺寸")
#         depth_img = depth_img.resize(rgb_img.size)
    
#     # 计算拼接后尺寸
#     width = rgb_img.width + depth_img.width
#     height = max(rgb_img.height, depth_img.height)
    
#     # 创建新画布
#     combined = Image.new("RGB", (width, height))
    
#     # 拼接图像
#     combined.paste(rgb_img, (0, 0))  # 左侧贴RGB图
#     combined.paste(depth_img, (rgb_img.width, 0))  # 右侧贴深度图
    
#     # 保存结果
#     # output_path = rgb_path.replace("rgb","combined")
#     save_path = rgb_path.replace("rgb","combined")
#     combined.save(save_path)
#     print(f"行 {line_num}: 已保存 {save_path}")

# # 处理每行数据
# with open(input_txt, "r") as f:
#     for line_num, line in enumerate(f, 1):
#         line = line.strip()
#         rgb_path, depth_path = line.split()
#         commbined(rgb_path,depth_path)
            