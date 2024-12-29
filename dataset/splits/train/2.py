def truncate_file(file_path, line_number):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 保留前 line_number 行
    lines = lines[line_number:]

    # 将剩余的行写回到文件中
    with open(file_path, 'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    # 替换为你的txt文件路径
    txt_file_path = "/home/chenwu/DisDepth/dataset/splits/train/indoor_train.txt"
    # 设置要保留的行数
    line_number = 238233
    truncate_file(txt_file_path, line_number)