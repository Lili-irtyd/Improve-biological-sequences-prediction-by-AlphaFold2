import os
import pandas as pd
import numpy as np

# 定义 20 种天然氨基酸和 "-" 字符
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

# 计算单个序列的 AAC 特征
def calculate_aac(sequence):
    sequence = sequence.upper()  # 转为大写
    total_length = len(sequence)
    aac = {aa: 0 for aa in amino_acids}  # 初始化 AAC 字典
    for aa in sequence:
        if aa in aac:
            aac[aa] += 1
    # 计算频率
    return [aac[aa] / total_length for aa in amino_acids]

# 处理 Excel 文件中的序列并保存为 .npy 文件
def process_excel_to_npy(input_excel, output_npy):
    print(f"Starting processing for Excel file: {input_excel}")

    # 读取 Excel 文件
    try:
        df = pd.read_excel(input_excel,engine="openpyxl")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 检查必要的列是否存在
    required_columns = ["File Name", "Sequence ID", "Window Sequence", "Numpy Data"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing one or more required columns: {required_columns}")
        return

    results = []

    # 遍历每一行数据
    for idx, row in df.iterrows():
        sequence = row["Window Sequence"]
        file_name = row["File Name"]
        sequence_id = row["Sequence ID"]

        # 计算 AAC 特征
        try:
            aac_features = calculate_aac(sequence)
        except Exception as e:
            print(f"Error processing sequence {sequence_id} in file {file_name}: {e}")
            continue

        results.append(aac_features)

    # 按 Excel 顺序保存为 .npy 文件
    results_array = np.array(results)
    np.save(output_npy, results_array)
    print(f"AAC features have been saved to {output_npy}")

# 使用示例
input_excel = "/media/gpux1/Pro1/app/intermediate_slide_fasta_label.xlsx"  # 替换为实际 Excel 文件路径
output_npy = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy"
process_excel_to_npy(input_excel, output_npy)
