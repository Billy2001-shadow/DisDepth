def filter_lines(input_file, output_file, keywords):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            if any(keyword in line for keyword in keywords):
                file.write(line)

if __name__ == '__main__':
    input_file = '/home/chenwu/DisDepth/dataset/splits/DIML/multi_train_files_with_gt_ablation_cls4+.txt'
    output_file = '/home/chenwu/DisDepth/dataset/splits/DIML/filtered_train_files.txt'
    keywords = ['DIML_indoor_1', 'DIML_indoor_2', 'ScanNet']
    filter_lines(input_file, output_file, keywords)