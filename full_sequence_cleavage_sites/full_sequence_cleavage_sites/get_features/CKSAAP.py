import os
import numpy as np
import pandas as pd
from itertools import product

def generate_cksaap_features(sequence, k_max=3):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    aa_pairs = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    feature_vector = {f"{pair}_k{k}": 0 for pair in aa_pairs for k in range(k_max + 1)}
    sequence = sequence.upper()

    sequence_length = len(sequence)
    for k in range(k_max + 1):
        for i in range(sequence_length - k - 1):
            aa1, aa2 = sequence[i], sequence[i + k + 1]
            if aa1 in amino_acids and aa2 in amino_acids:
                feature_vector[f"{aa1 + aa2}_k{k}"] += 1

    total_pairs = sum(feature_vector.values())
    if total_pairs > 0:
        for key in feature_vector:
            feature_vector[key] /= total_pairs

    return feature_vector

def process_excel_to_npy(input_excel, output_npy, k_max=3):
    print(f"🔍 开始处理文件: {input_excel}")
    try:
        df = pd.read_excel(input_excel, engine="openpyxl")
    except Exception as e:
        print(f"❌ 无法读取 Excel 文件: {e}")
        return

    required_column = "Window Sequence"
    if required_column not in df.columns:
        print(f"❌ 缺少必要列: {required_column}")
        return

    all_features = []
    skipped_rows = 0

    for idx, row in df.iterrows():
        sequence = row[required_column]
        if not isinstance(sequence, str) or len(sequence) < 10:
            skipped_rows += 1
            continue

        features = generate_cksaap_features(sequence, k_max)
        if sum(features.values()) == 0:
            print(f"⚠️  第 {idx + 1} 行序列全为 0，可能含非标准氨基酸或重复字符：{sequence}")
            skipped_rows += 1
            continue

        all_features.append(list(features.values()))

    if len(all_features) == 0:
        print("❗ 无有效特征提取，检查输入序列格式和内容")
        return

    features_array = np.array(all_features)
    np.save(output_npy, features_array)
    print(f"✅ 特征保存成功: {output_npy}（共 {features_array.shape[0]} 条序列）")
    print(f"🚫 跳过序列数: {skipped_rows}")

# 示例使用（路径请替换）
input_excel = "/media/gpux1/Pro1/app/intermediate_slide_fasta_label.xlsx"
output_npy = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap_fixed.npy"
process_excel_to_npy(input_excel, output_npy, k_max=3)

data = np.load(output_npy)
print("shape:", data.shape)
print("每条特征值是否全为 0:", np.all(data == 0, axis=1).sum(), "/", data.shape[0])

