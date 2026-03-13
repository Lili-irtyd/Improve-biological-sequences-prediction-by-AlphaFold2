#the first step to turn xlsx to txt for download tab file
import pandas as pd

# 读取 Excel 文件
excel_file = '/home/gpux1/ccd/2019_new.xlsx'
df = pd.read_excel(excel_file)

# 提取第一列，并保存为 TXT 文件
txt_file = '2019_new.txt'
df.iloc[:, 0].to_csv(txt_file, index=False, header=False)  # 只导出第一列，无索引无表头

print(f"Excel 第一列已成功保存为 TXT 文件：{txt_file}")
