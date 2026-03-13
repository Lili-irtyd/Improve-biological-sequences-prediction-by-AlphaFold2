import os
import numpy as np
from Bio import SeqIO

def process_fasta_files(input_folder, output_folder, window_height=30, step=1):
    """
    对文件夹中的 .fasta 文件进行滑动窗口切割，生成包含所有窗口的矩阵，并保存为一个文件。

    参数:
        input_folder (str): 输入文件夹路径，包含 .fasta 文件。
        output_folder (str): 输出文件夹路径，用于保存切割后的窗口数据。
        window_height (int): 滑动窗口的高度（序列长度），默认为 30。
        step (int): 滑动窗口的步长，默认为 1。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有 .fasta 文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.fasta') or file_name.endswith('.fa'):
            # 加载 .fasta 文件
            file_path = os.path.join(input_folder, file_name)
            sequences = list(SeqIO.parse(file_path, "fasta"))

            # 用于存储所有窗口的列表
            all_windows = []

            # 遍历所有序列并对每个序列进行切割
            for record in sequences:
                sequence = str(record.seq)

                # 检查数据是否满足窗口要求
                if len(sequence) < window_height:
                    print(f"序列 {record.id} 长度小于窗口大小，跳过...")
                    continue

                # 切割文件
                for start_pos in range(0, len(sequence) - window_height + 1, step):
                    # 切割出当前窗口
                    window = sequence[start_pos:start_pos + window_height]
                    all_windows.append(window)

            # 转换为 NumPy 数组
            all_windows = np.array(all_windows)

            # 保存所有窗口为一个 .npy 文件
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_windows.npy")
            np.save(output_file_path, all_windows)

            print(f"Processed {file_name}: {all_windows.shape[0]} segments saved as one file.")

# 使用示例
input_folder = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/FASTA_files"  # 替换为实际输入文件夹路径
output_folder = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/FASTA_files-slide"  # 替换为实际输出文件夹路径
process_fasta_files(input_folder, output_folder, window_height=30, step=1)
