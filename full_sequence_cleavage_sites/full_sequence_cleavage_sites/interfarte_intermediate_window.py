import numpy as np
import os

def combine_windows_from_folder(windows_folder):
    """
    将文件夹中的所有窗口数据合并成一个大的数组。

    参数:
        windows_folder (str): 存放窗口数据的文件夹。

    返回:
        numpy.ndarray: 合并后的窗口数据数组，形状为 (total_s, 30, 384)。
    """
    all_windows = []  # 用于存放所有窗口数据

    # 遍历窗口数据文件夹中的所有文件
    for window_file in os.listdir(windows_folder):
        if window_file.endswith('.npy'):
            # 获取窗口数据文件的完整路径
            windows_path = os.path.join(windows_folder, window_file)

            # 读取窗口数据
            windows = np.load(windows_path)

            # 将数据添加到列表中
            all_windows.append(windows)

    # 将所有窗口数据合并成一个大数组
    combined_windows = np.vstack(all_windows)

    return combined_windows

# 示例：指定窗口文件夹路径
windows_folder = "/media/gpux1/Pro1/app/intermediate-fasta"  # 包含窗口数据的文件夹

# 合并所有窗口数据
combined_windows = combine_windows_from_folder(windows_folder)
np.save("combined_windows.npy", combined_windows)


# 查看合并后的数据形状
print("合并后的窗口数据形状:", combined_windows.shape)
