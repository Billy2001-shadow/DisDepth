# import os

# def write_jpg_paths_to_txt(folder_path, output_file):
#     with open(output_file, 'w') as file:
#         for root, dirs, files in os.walk(folder_path):
#             for name in files:
#                 if name.endswith('.jpg'):
#                     file.write(os.path.join(root, name) + '\n')

# # 使用示例
# folder_path = '/mnt/chenwu/Relative_depth/NYU/imgs'  # 替换为你的文件夹路径
# output_file = '/home/chenwu/DisDepth/dataset/splits/train/NYU.txt'  # 替换为你的输出文件路径
# write_jpg_paths_to_txt(folder_path, output_file)

input_file = '/home/chenwu/DisDepth/dataset/splits/train/NYU.txt'
output_file = '/home/chenwu/DisDepth/dataset/splits/train/NYU_updated.txt'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        line = line.strip()
        if line:
            depth_path = line.replace('imgs', 'pseudo_label_uint8').replace('.jpg', '.png')
            outfile.write(f"{line} {depth_path}\n")