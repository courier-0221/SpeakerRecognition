import os
import shutil

src_dir = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/CN-Celeb_flac/data"
dst_dir = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/CN-Celeb_flac/data-small"
train_list = r"/home/dear/code/VoiceprintRecognition-Pytorch/dataset/train_list.txt"

'''
step1.处理训练数据集目录，从完整训练数据集中抽取固定数量作为小的训练数据子集，拷贝音频文件，生成文件目录
step2.从训练数据子集根据文件目录结构生成train_list.txt
'''

def copy_files_to_new_directory(src_dir, dest_dir, num_files=5):
    """
    从每个子目录中选取若干文件，并复制到新的目录下的对应子目录。

    :param src_dir: 源目录路径
    :param dest_dir: 目标目录路径
    :param num_files: 每个子目录中选取的文件数量
    """
    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 遍历源目录下的所有子目录
    for subdir in os.listdir(src_dir):
        subdir_path = os.path.join(src_dir, subdir)

        # 确保只处理子目录
        if os.path.isdir(subdir_path):
            # 获取子目录下的所有文件
            files = os.listdir(subdir_path)
            
            # 如果文件数少于需要的数量，全部选取；否则选前 num_files 个
            selected_files = files[:num_files]
            
            # 创建目标目录下的对应子目录
            dest_subdir_path = os.path.join(dest_dir, subdir)
            if not os.path.exists(dest_subdir_path):
                os.makedirs(dest_subdir_path)
            
            # 复制选取的文件到目标子目录
            for file in selected_files:
                src_file_path = os.path.join(subdir_path, file)
                dest_file_path = os.path.join(dest_subdir_path, file)

                if os.path.isfile(src_file_path):  # 确保是文件
                    shutil.copy(src_file_path, dest_file_path)
                    print(f"复制文件: {src_file_path} -> {dest_file_path}")

def create_small_train_list(list_path, data_path):
    f_train = open(list_path, 'w', encoding='utf-8')
    dirs = sorted(os.listdir(data_path))
    for label, d in enumerate(dirs):
        # 跳过测试集
        if label >= 800:continue
        for file in os.listdir(os.path.join(data_path, d)):
            sound_path = os.path.join(data_path, d, file).replace('\\', '/')
            f_train.write(f'{sound_path}\t{label}\n')
    f_train.close()

if __name__ == '__main__':
    # copy_files_to_new_directory(src_dir, dst_dir)
    create_small_train_list(train_list, dst_dir)