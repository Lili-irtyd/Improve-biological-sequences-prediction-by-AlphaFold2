import os

# 定义文件夹路径
folder1 = "/home/gpux1/ccd/FASTA_files"  # 替换为第一个文件夹的实际路径
folder2 = "/media/gpux1/Pro1/app/intermediate"  # 替换为第二个文件夹的实际路径

# 获取文件名（不含后缀和分隔符后的部分）
def get_file_names_before_hyphen(folder):
    files = os.listdir(folder)
    file_set = set()
    for file in files:
        if os.path.isfile(os.path.join(folder, file)):
            file_name = os.path.splitext(file)[0]  # 去掉后缀
            file_name = file_name.split("_")[0]  # 获取 `-` 前的部分
            file_set.add(file_name)
    return file_set

def main():
    # 获取第二个文件夹中以 `-` 前为基准的文件名集合
    folder2_files = get_file_names_before_hyphen(folder2)

    # 遍历第一个文件夹，删除重名的文件
    for file in os.listdir(folder1):
        file_name, _ = os.path.splitext(file)  # 获取文件名（不含后缀）
        if file_name in folder2_files:  # 判断文件名是否存在于第二个文件夹的集合中
            file_path = os.path.join(folder1, file)
            os.remove(file_path)  # 删除文件
            print(f"Deleted: {file_path}")

if __name__ == "__main__":
    main()

