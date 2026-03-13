import numpy as np
#data = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy", allow_pickle=True)
import numpy as np

# 加载数据
data = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy", allow_pickle=True)

# # 计算第一个 1 出现的位置
# first_one_index = np.argmax(data == 1)

# # 打印结果
# print(f"First 1 appears at index: {first_one_index}")

print(data)