import os
import numpy as np
import pandas as pd
from collections import Counter

def calculate_pssm_from_excel(input_excel, output_npy):
    """
    根据 Excel 文件计算 PSSM 并保存为 .npy 文件。

    参数:
        input_excel (str): 包含序列和标签的 Excel 文件路径。
        output_npy (str): 保存生成的 PSSM 文件路径。
    """
    # 读取 Excel 文件
    try:
        df = pd.read_excel(input_excel, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    # 检查必要列是否存在
    if not all(col in df.columns for col in ["Window Sequence", "Numpy Data"]):
        raise ValueError("Excel 文件中缺少必要的列: 'sequence' 或 'Numpy Data'")

    # 筛选正组和负组序列
    positive_sequences = df[df["Numpy Data"] == 1]["Window Sequence"].dropna().str.upper().tolist()#将 Pandas Series 转换为 Python 列表，顺序与 Excel 文件中 sequence 列的顺序一致。
    negative_sequences = df[df["Numpy Data"] == 0]["Window Sequence"].dropna().str.upper().tolist()

    if not positive_sequences:
        raise ValueError("正组序列为空，请检查 'Numpy Data' 列中是否存在值为 1 的记录。")
    if not negative_sequences:
        raise ValueError("负组序列为空，请检查 'Numpy Data' 列中是否存在值为 0 的记录。")

    # 定义氨基酸
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    # 构建频率矩阵
    positive_frequencies = np.zeros((30, len(amino_acids)))
    negative_frequencies = np.zeros((30, len(amino_acids)))

    for pos in range(30):
        # 正组统计
        pos_residues = [seq[pos] for seq in positive_sequences if len(seq) == 30]
        pos_counts = Counter(pos_residues)
        total_pos = len(pos_residues)
        for i, aa in enumerate(amino_acids):
            positive_frequencies[pos, i] = pos_counts.get(aa, 0) / total_pos if total_pos > 0 else 0

        # 负组统计
        neg_residues = [seq[pos] for seq in negative_sequences if len(seq) == 30]
        neg_counts = Counter(neg_residues)
        total_neg = len(neg_residues)
        for i, aa in enumerate(amino_acids):
            negative_frequencies[pos, i] = neg_counts.get(aa, 0) / total_neg if total_neg > 0 else 0

    # 生成 PSSM 特征
    sequences = df["Window Sequence"].dropna().str.upper().tolist()
    pssm_features = []

    for seq in sequences:
        if len(seq) != 30:
            print(f"Skipping sequence due to invalid length: {seq}")
            continue
        feature = []
        for pos in range(30):
            aa = seq[pos]
            if aa in amino_acids:
                aa_index = amino_acids.index(aa)
                feature.append(positive_frequencies[pos, aa_index])  # 正组频率
                feature.append(negative_frequencies[pos, aa_index])  # 负组频率
            else:
                feature.append(0)
                feature.append(0)
        pssm_features.append(feature)

    # 转换为 NumPy 数组并保存
    pssm_array = np.array(pssm_features)
    np.save(output_npy, pssm_array)
    print(f"PSSM saved to {output_npy}, shape: {pssm_array.shape}")

# 示例调用
input_excel = "/media/gpux1/Pro1/app/intermediate_slide_fasta_label.xlsx"  # 替换为实际的 Excel 文件路径
output_npy = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy"  # 替换为实际的输出文件路径
calculate_pssm_from_excel(input_excel, output_npy)
