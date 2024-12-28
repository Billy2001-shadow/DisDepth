import os
import shutil

def copy_and_rename_jpg_files(txt_file, dst_dir, output_txt_file):
    # 确保目标文件夹存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with open(txt_file, 'r') as file:
        lines = file.readlines()

    with open(output_txt_file, 'w') as output_file:
        for line in lines:
            line = line.strip()
            if line.endswith('.jpg'):
                src_file = line
                # 获取最后一级目录名和文件名
                parts = src_file.split('/')
                dir_name = parts[-2]
                file_name = parts[-1]
                new_file_name = f"{dir_name}_{file_name}"
                dst_file = os.path.join(dst_dir, new_file_name)
                shutil.copy2(src_file, dst_file)
                output_file.write(dst_file + '\n')
                print(f'Copied {src_file} to {dst_file}')

# 示例用法
txt_file = '/home/chenwu/DisDepth/generate_datasets/data_splits/NYUD.txt'  # 替换为你的txt文件路径
dst_dir = '/data2/cw/Relative_depth/NYU/imgs'  # 替换为你的目标文件夹路径
output_txt_file = '/home/chenwu/DisDepth/generate_datasets/data_splits/copied_files.txt'  # 替换为你的输出txt文件路径
copy_and_rename_jpg_files(txt_file, dst_dir, output_txt_file)