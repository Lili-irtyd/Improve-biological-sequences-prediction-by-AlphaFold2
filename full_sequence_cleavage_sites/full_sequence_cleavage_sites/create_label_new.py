import numpy as np
import os
import pandas as pd


def generate_custom_vector(n, idx_list):
    """
    生成一个向量，指定某些索引位置的元素为1，其他位置为0。

    参数:
        n (int): 向量的长度。
        idx_list (list): 需要设置为1的索引列表。

    返回:
        numpy.ndarray: 自定义生成的向量。
    """
    # 创建一个全零的向量
    vector = np.zeros(n, dtype=int)

    # 将指定的索引位置设置为1（确保索引有效）
    idx_list = [idx for idx in idx_list if 0 <= idx < n]  # 只保留有效的索引
    idx_list = [int(idx) for idx in idx_list]  # 强制转换为整数
    vector[idx_list] = 1

    return vector


def generate_labels_from_excel(excel_path, data_folder, output_folder, engine="openpyxl"):
    """
    从2019.xlsx文件生成标签并保存。

    参数:
        excel_path (str): 2019.xlsx文件路径。
        data_folder (str): 包含蛋白质数据的文件夹。
        output_folder (str): 保存标签向量的输出文件夹。
        engine (str): 使用的引擎，默认为 "openpyxl"。
    """
    # 加载Excel文件
    df = pd.read_excel(excel_path, engine=engine)

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历数据文件夹中的所有蛋白质文件
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.npy'):
            # 获取蛋白质编号（假设文件名即为蛋白质编号，如 O08736.npy）
            protein_name = os.path.splitext(file_name)[0]
            protein_name = protein_name.split('_')[0]

            # 查找对应的蛋白质在Excel中的记录
            protein_data = df[df['Acc'] == protein_name]

            if not protein_data.empty:
                # 获取Position列的值并减去15，得到idx_list
                positions = protein_data['Position'].values - 15

                # 获取蛋白质序列的长度，假设文件名和蛋白质名相同的 .npy 文件保存了蛋白质的序列
                protein_sequence = np.load(os.path.join(data_folder, file_name))  # 假设文件包含序列
                window_size = len(protein_sequence)  # 使用序列的长度作为window_size

                # 过滤掉无效的Position值，确保索引范围在 [0, window_size - 1] 内
                valid_positions = positions[(positions >= 0) & (positions < window_size)]  # 只保留有效的索引

                # 转换为列表
                idx_list = valid_positions.tolist()

                # 生成标签向量
                vector = generate_custom_vector(window_size, idx_list)

                # 保存标签向量到文件
                output_file_path = os.path.join(output_folder, f"{protein_name}_label.npy")
                np.save(output_file_path, vector)
                print(f"生成并保存了 {protein_name} 的标签向量")
            else:
                print(f"在Excel文件中未找到蛋白质 {protein_name}")


# 示例：生成标签
excel_path = "/home/gpux1/ccd/dataset/2019.xlsx"  # 替换为实际的2019.xlsx文件路径
data_folder = "/media/gpux1/Pro1/app/intermediate-slide"  # 替换为数据文件夹路径
output_folder = "/media/gpux1/Pro1/app/labels"  # 替换为输出文件夹路径

generate_labels_from_excel(excel_path, data_folder, output_folder, engine="openpyxl")
