import numpy as np
import pandas as pd

def insert_npy_into_excel(excel_file, npy_file, output_excel):
    """
    将 .npy 文件的内容插入到 Excel 文件的最后一列。

    参数:
        excel_file (str): 输入的 Excel 文件路径。
        npy_file (str): 输入的 .npy 文件路径。
        output_excel (str): 输出的 Excel 文件路径，保存修改后的结果。
    """
    # 读取已有的 Excel 文件
    df = pd.read_excel(excel_file,engine="openpyxl")

    # 读取 .npy 文件
    npy_data = np.load(npy_file, allow_pickle=True)

    # 确保 .npy 数据的长度与 DataFrame 的行数相匹配
    if len(npy_data) != len(df):
        print(f"Error: The length of the .npy file does not match the number of rows in the Excel file.")
        return

    # 将 .npy 数据添加到 DataFrame 的最后一列
    df['Numpy Data'] = npy_data

    # 保存修改后的 DataFrame 到新的 Excel 文件
    df.to_excel(output_excel, index=False)
    print(f"Excel file with appended .npy data has been saved to {output_excel}")

# 示例使用
excel_file = "/media/gpux1/Pro1/app/intermediate_slide_fasta.xlsx"  # 替换为已有的 Excel 文件路径
npy_file = "/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy"  # 替换为 .npy 文件路径
output_excel = "/media/gpux1/Pro1/app/intermediate_slide_fasta_label.xlsx"  # 替换为输出的 Excel 文件路径

insert_npy_into_excel(excel_file, npy_file, output_excel)
