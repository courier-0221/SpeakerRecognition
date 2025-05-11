import collections
import shutil
import os

dst_dir = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/cn-celeb-test/test-small"
input_file_path = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/cn-celeb-test/trials_list_big.txt"
output_file_path = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/cn-celeb-test/trials_list.txt"

'''
step1.处理trials_list.txt，从完整测试数据集中抽取固定数量生成小的测试数据子集，生成trials_list_small.txt
step2.从trials_list_small.txt中，拷贝音频文件，生成文件目录
'''


def process_file(input_file, output_file, num_files_per_label=5):
    """
    从输入文件中提取每个label对应的最多num_files_per_label个文件，
    并将结果写入输出文件。

    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param num_files_per_label: 每个label保留的文件数量
    """
    label_to_files = collections.defaultdict(list)
    
    # 读取输入文件并按label分组
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 按 <test_audio_file> \t <label> 格式解析
            parts = line.strip().split('\t')
            if len(parts) == 2:
                test_audio_file, label = parts[0], parts[1]
                label_to_files[label].append(test_audio_file)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for label, files in label_to_files.items():
            # 对每个label取最多num_files_per_label个文件
            for test_audio_file in files[:num_files_per_label]:
                f.write(f"{test_audio_file}\t{label}\n")
                print(f"写入文件: {test_audio_file}\t{label}")


def copy_files_to_new_directory(dst_dir, trials_file):

    os.makedirs(dst_dir, exist_ok=True)

    with open(trials_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        filename = line.strip().split('\t')[0]
        filepath = os.path.join("/home/dear/code/VoiceprintRecognition-Pytorch", filename)
        if os.path.exists(filepath):
            shutil.copy(filepath, dst_dir)
            print(f"已拷贝: {filepath}")
        else:
            print(f"文件不存在: {filepath}")

if __name__ == '__main__':
    process_file(input_file_path, output_file_path, num_files_per_label=5)
    # copy_files_to_new_directory(dst_dir, output_file_path)