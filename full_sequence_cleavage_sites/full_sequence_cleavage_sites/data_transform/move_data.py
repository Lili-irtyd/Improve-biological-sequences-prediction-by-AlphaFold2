import os
import pandas as pd
from Bio import SeqIO


def fasta_to_excel_sorted(input_folder, reference_folder, output_excel):
    """
    将文件夹中的 .fasta 文件转换为 Excel 文件，包含文件名和序列两列，
    按照 reference_folder 中的文件顺序排列，并按照氨基酸名后面的编码从 0 开始排序。

    参数:
        input_folder (str): 包含 .fasta 文件的文件夹路径 (intermediate_slide_fasta)。
        reference_folder (str): 用于提供文件顺序的参考文件夹路径 (intermediate)。
        output_excel (str): 输出 Excel 文件路径。
    """
    data = []

    # 获取 reference_folder 中的文件顺序
    reference_files = sorted(os.listdir(reference_folder))

    # 遍历 reference_folder 中的文件，确保按照该顺序处理 input_folder 中的 .fasta 文件
    for file_name in reference_files:
        if file_name.endswith(".fasta") or file_name.endswith(".fa"):
            file_path = os.path.join(input_folder, file_name)

            # 确保 input_folder 中存在对应的文件
            if os.path.exists(file_path):
                # 读取每个 .fasta 文件中的序列
                for record in SeqIO.parse(file_path, "fasta"):
                    sequence_name = record.id
                    sequence = str(record.seq).upper()

                    # 提取氨基酸后的编码并排序
                    if "_" in sequence_name:
                        try:
                            encoding = int(sequence_name.split("_")[-1])
                        except ValueError:
                            encoding = -1  # 如果无法解析编码，设置为 -1
                    else:
                        encoding = -1

                    # 添加到数据列表
                    data.append({"Acc": file_name, "Sequence": sequence, "Encoding": encoding})

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 按照文件夹顺序和编码从小到大排序
    df.sort_values(by=["Acc", "Encoding"], inplace=True, ignore_index=True)

    # 删除中间列，只保留文件名和序列两列
    df = df[["Acc", "Sequence"]]

    # 保存为 Excel 文件
    df.to_excel(output_excel, index=False)
    print(f"Excel file saved to {output_excel}")


# 示例使用
input_folder = "/media/gpux1/Pro1/app/intermediate_slide_fasta"  # 替换为实际输入文件夹路径
reference_folder = "/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy"  # 替换为实际的参考文件夹路径
output_excel = "/media/gpux1/Pro1/app/intermediate_slide_fasta"  # 替换为输出 Excel 文件路径

fasta_to_excel_sorted(input_folder, reference_folder, output_excel)
