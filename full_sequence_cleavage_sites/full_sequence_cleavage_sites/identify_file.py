import numpy as np
data = np.load("combined_labels.npy")
print(data)
# 计算值为 1 的数量
count_ones = np.sum(data == 1)
print(f"Number of 1s in the file: {count_ones}")
# 计算值为 1 的数量
count_zeros = np.sum(data == 0)
print(f"Number of 1s in the file: {count_zeros}")