import numpy as np
import os

def combine_labels_from_folder(labels_folder):
    """
    将文件夹中的所有标签数据按文件顺序合并成一个大的数组。

    参数:
        labels_folder (str): 存放标签数据的文件夹。

    返回:
        numpy.ndarray: 合并后的标签数据数组，形状为 (total_s,)。
    """
    all_labels = []  # 用于存放所有标签数据

    # 获取文件夹中的所有文件，按文件名排序
    label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith('.npy')])

    # 遍历排序后的标签数据文件
    for label_file in label_files:
        # 获取标签数据文件的完整路径
        labels_path = os.path.join(labels_folder, label_file)

        # 读取标签数据
        labels = np.load(labels_path)

        # 将标签数据添加到列表中
        all_labels.append(labels)

    # 将所有标签数据合并成一个大数组
    combined_labels = np.concatenate(all_labels, axis=0)  # 使用 axis=0 来按第一个维度合并

    return combined_labels

# 示例：指定标签文件夹路径
labels_folder = "/media/gpux1/Pro1/app/labels"  # 包含标签数据的文件夹

# 合并所有标签数据
combined_labels = combine_labels_from_folder(labels_folder)

# 保存合并后的标签数据
np.save("combined_labels.npy", combined_labels)

# 查看合并后的标签数据形状
print("合并后的标签数据形状:", combined_labels.shape)
