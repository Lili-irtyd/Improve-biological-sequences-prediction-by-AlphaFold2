import os
import pandas as pd
from Bio import SeqIO

def process_fasta_files_to_excel(input_folder, output_excel, window_height=30, step=1):
    """
    对文件夹中的 .fasta 文件进行滑动窗口切割，生成包含所有窗口的矩阵，并保存为一个 Excel 文件。
    按照输入文件夹中文件的顺序排列。

    参数:
        input_folder (str): 输入文件夹路径，包含 .fasta 文件。
        output_excel (str): 输出 Excel 文件路径。
        window_height (int): 滑动窗口的高度（序列长度），默认为 30。
        step (int): 滑动窗口的步长，默认为 1。
    """
    # 用于存储结果的列表
    data = []

    # 获取并排序输入文件夹中的所有 .fasta 文件
    fasta_files = [f for f in os.listdir(input_folder) if f.endswith('.fasta') or f.endswith('.fa')]
    fasta_files.sort()  # 按字母顺序排序文件

    # 遍历排序后的文件
    for file_name in fasta_files:
        file_path = os.path.join(input_folder, file_name)

        # 读取 .fasta 文件中的序列
        for record in SeqIO.parse(file_path, "fasta"):
            sequence_name = record.id
            sequence = str(record.seq).upper()

            # 确保序列长度大于窗口大小
            if len(sequence) < window_height:
                print(f"序列 {sequence_name} 长度小于窗口大小，跳过...")
                continue

            # 切割窗口
            window_index = 1  # 初始化窗口编号
            for start_pos in range(0, len(sequence) - window_height + 1, step):
                # 切割出当前窗口
                window = sequence[start_pos:start_pos + window_height]

                # 将文件名、序列、窗口编号添加到数据列表
                data.append({"File Name": file_name, "Sequence ID": sequence_name, "Window Index": window_index,
                             "Window Sequence": window})

                window_index += 1  # 增加窗口编号

    # 将结果存储到 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 保存为 Excel 文件
    df.to_excel(output_excel, index=False)
    print(f"Excel file saved to {output_excel}")


# 示例使用
input_folder = "/media/gpux1/Pro1/app/intermediate-fasta"  # 替换为实际输入文件夹路径
output_excel = "/media/gpux1/Pro1/app/intermediate_slide_fasta.xlsx"  # 替换为输出 Excel 文件路径

process_fasta_files_to_excel(input_folder, output_excel, window_height=30, step=1)
