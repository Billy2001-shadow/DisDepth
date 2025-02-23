# 读取原始文件并筛选出 d1 值低于 0.96 的行
input_file = 'd1_results_diode.txt'
output_file = 'd1_sorted_diode.txt'

# 先筛选出 d1 值低于 0.96 的行
filtered_lines = []
with open(input_file, 'r') as f_in:
    for line in f_in:
        # 分割路径和 d1 值
        parts = line.split()
        image_path = parts[0]
        d1_value = float(parts[1])

        # 筛选出 d1 值低于 0.96 的行
        filtered_lines.append(line)
        
            

# 按照 d1 值排序
filtered_lines.sort(key=lambda x: float(x.split()[1]))

# 将排序后的内容写入新的文件
with open(output_file, 'w') as f_out:
    for line in filtered_lines:
        f_out.write(line)


