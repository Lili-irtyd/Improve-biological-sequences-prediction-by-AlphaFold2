import numpy as np
from pathlib import Path
import os

# 设置原始文件夹 和 输出文件夹
input_dir = Path("/media/gpux1/Pro1/new_output_sequence_train")
output_dir = Path("/media/gpux1/Pro1/train_AF2_single")

# 创建输出文件夹（如果不存在）
output_dir.mkdir(parents=True, exist_ok=True)

# 遍历 input_dir 中所有 .npy 文件
for npy_file in input_dir.glob("*.npy"):
    print(f"Processing: {npy_file.name}")

    # 加载 npy 文件
    data = np.load(npy_file, allow_pickle=True)

    # 取出 dict
    if isinstance(data, dict):
        data_dict = data
    else:
        data_dict = data.item()

    # 提取 single 内容
    single = data_dict['single']

    # 生成新的文件名
    new_filename = npy_file.stem + "_single.npy"
    new_filepath = output_dir / new_filename

    # 保存 single 为新的 npy 文件
    np.save(new_filepath, single)

    print(f"Saved single to: {new_filepath}")

print("✅ All files processed.")
