import os
import numpy as np
from pathlib import Path

# 设置包含 .pssm 文件的目录路径
pssm_dir = Path("/home/gpux1/RNA-117_Test_pssm")  # ← 修改为你的目录路径
output_dir = Path("/media/gpux1/Pro1/test_PSSM")   # 或者另一个路径，例如 Path("/path/to/save/npy")

# 遍历该目录中的所有 .pssm 文件
for pssm_file in pssm_dir.glob("*.pssm"):
    try:
        # 构造新的文件名
        base_name = pssm_file.stem  # 去除.pssm扩展名
        npy_filename = f"{base_name}_pssm.npy"
        npy_path = output_dir / npy_filename

        # 读取 PSSM 文件
        with open(pssm_file, "r") as f:
            lines = f.readlines()

        # 解析数据部分：跳过前几行非数据（如前3-5行标题）
        data_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 22 and parts[0].isdigit():
                values = [int(x) for x in parts[2:22]]  # 取中间20列整数
                data_lines.append(values)

        # 转换为 numpy 数组
        pssm_array = np.array(data_lines, dtype=np.int32)

        # 保存为 .npy 文件
        np.save(npy_path, pssm_array)
        print(f"✅ Saved: {npy_path}")

    except Exception as e:
        print(f"❌ Failed to process {pssm_file.name}: {e}")
