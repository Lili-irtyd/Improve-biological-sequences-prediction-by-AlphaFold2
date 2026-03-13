import numpy as np
#
#
#
# 加载 .npy 文件
npy_file_path = "/path/to/file.npy"
data = np.load("/media/gpux1/Pro1/test_AF2_single/0002_5gan_D_single.npy",allow_pickle=True)

# 查看 single 内容
print(f"single shape: {data.shape}")
print(f"single dtype: {data.dtype}")




