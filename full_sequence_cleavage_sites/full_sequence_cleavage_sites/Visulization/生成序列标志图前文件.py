import pandas as pd


def calculate_amino_acid_frequencies_from_table(window_sequences):
    """
    根据表中提供的 Window Sequence 数据计算氨基酸频率。

    参数:
        window_sequences (list): 窗口序列列表，每个序列长度为 30。

    返回:
        dict: 各位置的氨基酸频率字典。
    """
    # 定义20种天然氨基酸
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    position_counts = {pos: {aa: 0 for aa in amino_acids} for pos in range(-15, 15)}  # 初始化计数
    total_counts = {pos: 0 for pos in range(-15, 15)}  # 初始化总计数

    # 遍历窗口序列
    for sequence in window_sequences:
        sequence = sequence.upper()  # 转为大写
        if len(sequence) != 30:  # 确保序列长度为 30
            print(f"Skipping sequence (length != 30): {sequence}")
            continue

        # 计算位置范围 (-15 到 15)
        for i, aa in enumerate(sequence):
            position = i - 15  # 位置映射到 -15 到 15
            if aa in amino_acids:
                position_counts[position][aa] += 1
                total_counts[position] += 1

    # 将计数转换为频率
    frequencies = {
        pos: {aa: count / total_counts[pos] if total_counts[pos] > 0 else 0 for aa, count in aa_counts.items()}
        for pos, aa_counts in position_counts.items()
    }

    return frequencies


# 从 Excel 文件中提取数据
def load_window_sequences_from_excel(excel_path, value):
    """
    从 Excel 表中提取 Numpy Data 列值等于指定值的 Window Sequence 数据。

    参数:
        excel_path (str): Excel 文件路径。
        value (int): Numpy Data 列的值（0 或 1）。

    返回:
        list: 窗口序列列表。
    """
    df = pd.read_excel(excel_path,engine="openpyxl")
    window_sequences = df[df["Numpy Data"] == value]["Window Sequence"].tolist()
    return window_sequences


# 主函数
def main():
    # Excel 文件路径
    excel_path = "/media/gpux1/Pro1/app/intermediate_slide_fasta_label.xlsx"  # 替换为实际 Excel 文件路径

    # 加载数据
    window_sequences_1 = load_window_sequences_from_excel(excel_path, value=1)  # 选择 Numpy Data = 1
    window_sequences_0 = load_window_sequences_from_excel(excel_path, value=0)  # 选择 Numpy Data = 0

    # 计算频率
    print("Calculating frequencies for Numpy Data = 1...")
    frequencies_1 = calculate_amino_acid_frequencies_from_table(window_sequences_1)
    print("Calculating frequencies for Numpy Data = 0...")
    frequencies_0 = calculate_amino_acid_frequencies_from_table(window_sequences_0)

    # 打印结果
    print("Frequencies for Numpy Data = 1:")
    for position, aa_freqs in frequencies_1.items():
        print(f"{position}: {aa_freqs},")

    print("\nFrequencies for Numpy Data = 0:")
    for position, aa_freqs in frequencies_0.items():
        print(f"{position}: {aa_freqs},")


if __name__ == "__main__":
    main()
