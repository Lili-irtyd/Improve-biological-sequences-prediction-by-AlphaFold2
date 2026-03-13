import os
import numpy as np
import pandas as pd

def binary_encoding(sequence, encoding_map):
    """
    将氨基酸序列转换为 Binary Encoding 特征。

    参数:
        sequence (str): 氨基酸序列。
        encoding_map (dict): 氨基酸到二进制向量的映射。

    返回:
        numpy.ndarray: 21 * len(sequence) 的特征矩阵。
    """
    # 对每个氨基酸使用 encoding_map 转换为 21 维向量
    try:
        return np.array([encoding_map[aa] for aa in sequence])
    except KeyError as e:
        print(f"Error: Unrecognized amino acid '{e.args[0]}' in sequence.")
        raise

def extract_be_features_from_excel(input_excel, output_file):
    """
    从 Excel 文件中提取 Binary Encoding 特征。

    参数:
        input_excel (str): 输入的 Excel 文件路径。
        output_file (str): 保存提取特征的 .npy 文件路径。
    """
    # 定义 21 种字符的二进制编码
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding_map = {aa: [1 if i == idx else 0 for i in range(20)] for idx, aa in enumerate(amino_acids)}

    print(f"Reading Excel file: {input_excel}")
    try:
        df = pd.read_excel(input_excel, engine="openpyxl")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 检查必要的列是否存在
    required_columns = ["File Name", "Sequence ID", "Window Sequence", "Numpy Data"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing one or more required columns: {required_columns}")
        return

    all_features = []

    # 遍历每一行数据
    for idx, row in df.iterrows():
        sequence = row["Window Sequence"]
        if not isinstance(sequence, str):
            print(f"  Skipping row {idx + 1}: invalid sequence format.")
            continue

        print(f"  Processing sequence: {row['Sequence ID']} (length: {len(sequence)})")

        # 检查是否有未知字符
        for aa in sequence:
            if aa not in encoding_map:
                print(f"  Warning: Skipping sequence {row['Sequence ID']} due to unrecognized amino acid '{aa}'.")
                break
        else:
            # 将序列转换为 Binary Encoding 特征
            feature = binary_encoding(sequence, encoding_map)

            # 展平为一维数组，并添加到结果列表中
            all_features.append(feature.flatten())

    # 合并所有序列的特征
    all_features = np.array(all_features)

    # 保存特征到 .npy 文件
    np.save(output_file, all_features)
    print(f"特征已保存到 {output_file}, 特征矩阵形状: {all_features.shape}")

# 示例：提取特征
input_excel = "/media/gpux1/Pro1/app/intermediate_slide_fasta_label.xlsx"  # 替换为实际 Excel 文件路径
output_file = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy"  # 保存路径
extract_be_features_from_excel(input_excel, output_file)
