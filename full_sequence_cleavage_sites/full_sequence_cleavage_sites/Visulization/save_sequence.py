import pandas as pd


def split_excel_by_classification(input_excel, output_pos_file, output_neg_file):
    """
    根据 Numpy Data 列的值对 Excel 文件进行分类，并分别保存正类和负类中的 Window Sequence。

    参数:
        input_excel (str): 输入的 Excel 文件路径。
        output_pos_file (str): 输出的正类文本文件路径。
        output_neg_file (str): 输出的负类文本文件路径。
    """
    # 读取 Excel 文件
    df = pd.read_excel(input_excel,engine="openpyxl")

    # 检查必要列是否存在
    required_columns = ["Numpy Data", "Window Sequence"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"输入文件缺少必要的列：{required_columns}")

    # 根据 Numpy Data 列分类
    pos_sequences = df.loc[df["Numpy Data"] == 1, "Window Sequence"]
    neg_sequences = df.loc[df["Numpy Data"] == 0, "Window Sequence"]

    # 保存正类序列到文本文件
    with open(output_pos_file, "w") as pos_file:
        for sequence in pos_sequences:
            pos_file.write(sequence + "\n")

    # 保存负类序列到文本文件
    with open(output_neg_file, "w") as neg_file:
        for sequence in neg_sequences:
            neg_file.write(sequence + "\n")

    print(f"正类序列已保存到 {output_pos_file}")
    print(f"负类序列已保存到 {output_neg_file}")


# 示例使用
input_excel = "/media/gpux1/Pro1/app/intermediate_slide_fasta_label.xlsx"  # 替换为实际的 Excel 文件路径
output_pos_file = "/home/gpux1/Downloads/phos_Y_pos.txt"  # 保存正类序列的文本文件路径
output_neg_file = "/home/gpux1/Downloads/phos_Y_neg.txt"  # 保存负类序列的文本文件路径

split_excel_by_classification(input_excel, output_pos_file, output_neg_file)
