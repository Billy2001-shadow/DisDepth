import os

def check_paths(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        img_path, depth_path = line.strip().split()
        
        img_exists = os.path.exists(img_path)
        depth_exists = os.path.exists(depth_path)
        
        if not img_exists:
            print(f"Image path does not exist: {img_path}")
        if not depth_exists:
            print(f"Depth path does not exist: {depth_path}")
        
        if img_exists and depth_exists:
            print(f"Both paths exist: {img_path}, {depth_path}")

if __name__ == "__main__":
    # 替换为你的txt文件路径
    txt_file_path = "relative_depth_train.txt"
    check_paths(txt_file_path)




# 应该会剩238233个文件，这些文件都存在
# import os

# def remove_invalid_paths(file_path):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()

#     # 过滤掉包含 'Turquie' 的行
#     valid_lines = [line for line in lines if 'Turquie' not in line]  # Turquie无效路径

#     # 将剩余的行写回到文件中
#     with open(file_path, 'w') as f:
#         f.writelines(valid_lines)

# if __name__ == "__main__":
#     # 替换为你的txt文件路径
#     txt_file_path = "relative_depth_train.txt"
#     remove_invalid_paths(txt_file_path)

# /data2/cw/Relative_depth/HRWSI/imgs/72629_pbz98_3D_MPO_70pc Turquie Aphrodisias.jpg /data2/cw/Relative_depth/HRWSI/pseudo_depth/72629_pbz98_3D_MPO_70pc Turquie Aphrodisias.npy
# /data2/cw/Relative_depth/HRWSI/imgs/73106_pbz98_3D_MPO_70pc Turquie Konya Musee de Mevlana.jpg /data2/cw/Relative_depth/HRWSI/pseudo_depth/73106_pbz98_3D_MPO_70pc Turquie Konya Musee de Mevlana.npy
# /data2/cw/Relative_depth/HRWSI/imgs/72624_pbz98_3D_MPO_70pc Turquie Aphrodisias.jpg /data2/cw/Relative_depth/HRWSI/pseudo_depth/72624_pbz98_3D_MPO_70pc Turquie Aphrodisias.npy
# /data2/cw/Relative_depth/HRWSI/imgs/72662_pbz98_3D_MPO_70pc Turquie Aphrodisias Tetrapylon.jpg /data2/cw/Relative_depth/HRWSI/pseudo_depth/72662_pbz98_3D_MPO_70pc Turquie Aphrodisias Tetrapylon.npy
# /data2/cw/Relative_depth/HRWSI/imgs/73050_pbz98_3D_MPO_70pc Turquie Pamukkale.jpg /data2/cw/Relative_depth/HRWSI/pseudo_depth/73050_pbz98_3D_MPO_70pc Turquie Pamukkale.npy
# /data2/cw/Relative_depth/HRWSI/imgs/72758_pbz98_3D_MPO_70pc Turquie Aphrodisias Stade.jpg /data2/cw/Relative_depth/HRWSI/pseudo_depth/72758_pbz98_3D_MPO_70pc Turquie Aphrodisias Stade.npy



