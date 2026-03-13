import os
import numpy as np
import matplotlib.pyplot as plt

def display_and_check_npy_file(file_path):
    """
    加载并显示 .npy 文件的内容，同时检查数据中是否存在 NaN 或 Inf 值。

    参数:
        file_path (str): .npy 文件的路径。
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件不存在: {file_path}")
        return

    try:
        # 加载 .npy 文件
        data = np.load(file_path)
    except Exception as e:
        print(f"❌ 错误：加载文件失败: {file_path}. 错误信息: {e}")
        return

    # 检查数据维度和类型
    print(f"\n--- 文件信息: {os.path.basename(file_path)} ---")
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")

    # =========================================================
    # 核心改进点：检查 NaN (Null) 和 Inf (无穷大)
    # =========================================================

    # 1. 检查 NaN (Not a Number)
    if np.issubdtype(data.dtype, np.floating) or np.issubdtype(data.dtype, np.complexfloating):
        # 只有浮点数或复数类型才有 NaN 概念
        nan_count = np.count_nonzero(np.isnan(data))
        if nan_count > 0:
            print(f"⚠️ 警告：检测到 {nan_count} 个 NaN (缺失值)。")
            print(f"   NaN 在数据中的占比: {nan_count / data.size * 100:.4f}%")
        else:
            print("✅ 检查通过：未检测到 NaN (缺失值)。")
            
        # 2. 检查 Inf (无穷大)
        inf_count = np.count_nonzero(np.isinf(data))
        if inf_count > 0:
            print(f"⚠️ 警告：检测到 {inf_count} 个 Inf (无穷大值)。")
            print(f"   Inf 在数据中的占比: {inf_count / data.size * 100:.4f}%")
        else:
            print("✅ 检查通过：未检测到 Inf (无穷大值)。")
    else:
        # 非浮点类型（如整数、布尔值）没有 NaN/Inf 概念
        print("ℹ️ 数据为非浮点类型，通常不包含 NaN/Inf。")

    print("-" * 40)
    # =========================================================
    # 可视化部分（保持原样）
    # =========================================================

    # 如果是 1D 或 2D 数据，进行可视化
    if data.ndim == 1:
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title(f"1D 数据展示: {os.path.basename(file_path)}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.show()
    elif data.ndim == 2:
        plt.figure(figsize=(8, 6))
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar(label="Value")
        plt.title(f"2D 数据展示: {os.path.basename(file_path)}")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.show()
    else:
        print("数据维度超过 2D，无法直接可视化。")
        print(f"数据内容示例（前几行）:\n{data.flat[:10]}")


# 示例：展示文件 (替换为您的路径)
file_path = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy"
display_and_check_npy_file(file_path)

# 您也可以手动创建一个包含 NaN 的测试文件来验证功能
# test_data = np.array([1.0, 2.0, np.nan, 4.0, np.inf, 6.0])
# np.save('test_nan_inf.npy', test_data)
# display_and_check_npy_file('test_nan_inf.npy')