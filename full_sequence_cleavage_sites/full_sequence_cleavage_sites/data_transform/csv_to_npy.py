import pandas as pd
import numpy as np

# 读取 CSV 文件
def csv_to_npy(csv_file, npy_file):
    """
    将 CSV 文件转换为 NPY 文件。

    参数:
        csv_file (str): 输入的 CSV 文件路径。
        npy_file (str): 输出的 NPY 文件路径。
    """
    # 使用 Pandas 读取 CSV 文件
    data = pd.read_csv(csv_file)

    # 转换为 NumPy 数组
    np_data = data.values

    # 保存为 NPY 文件
    np.save(npy_file, np_data)
    print(f"转换完成: {csv_file} -> {npy_file}")

# 示例
csv_file = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.csv"
npy_file="/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy"
#出 NPY 文件路径
csv_to_npy(csv_file, npy_file)


## 加载 NPY 文件
# data = np.load("example.npy")
# print(data)

#保存为npy后就没有表头了