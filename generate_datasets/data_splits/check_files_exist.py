import os

def check_file_existence(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if os.path.isfile(line):
            print(f'File exists: {line}')
        else:
            print(f'File does not exist: {line}')
            exit()

# 示例用法
txt_file = '/home/chenwu/DisDepth/generate_datasets/data_splits/DIML_indoor.txt'  # 替换为你的txt文件路径
check_file_existence(txt_file)