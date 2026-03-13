#delete the unnecessary file of .npy
import os

# 指定要操作的文件夹路径
folder_path = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/output_CS_npy_files"

# 遍历文件夹中的所有文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 检查文件是否以指定的后缀结尾
        if file.endswith("_msa.npy") or file.endswith("_msa_first_row.npy") or file.endswith("_pair.npy"):
            file_path = os.path.join(root, file)  # 获取文件的完整路径
            try:
                os.remove(file_path)  # 删除文件
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
