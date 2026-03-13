import os
import re
import requests

def extract_protein_names(folder_path):
    """
    提取文件夹中包含 '_single' 的文件名中，以 '_single' 之前的蛋白质名称。

    参数:
        folder_path (str): 包含 FASTA 文件的文件夹路径。

    返回:
        set: 提取的蛋白质名称集合。
    """
    protein_names = set()
    for file_name in os.listdir(folder_path):
        if "_single" in file_name:
            # 提取 "_single" 之前的部分作为蛋白质名称
            protein_name = file_name.split("_single")[0]
            protein_names.add(protein_name)
    return protein_names

def download_fasta(protein_name, output_folder):
    """
    根据蛋白质名称从 UniProt 下载 FASTA 文件。

    参数:
        protein_name (str): 蛋白质名称。
        output_folder (str): 下载的 FASTA 文件保存路径。
    """
    base_url = "https://www.uniprot.org/uniprot/"
    query_url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_name}&format=fasta"

    try:
        response = requests.get(query_url, timeout=10)
        if response.status_code == 200 and response.text.strip():
            # 将结果保存为 .fasta 文件
            file_path = os.path.join(output_folder, f"{protein_name}.fasta")
            with open(file_path, "w") as fasta_file:
                fasta_file.write(response.text)
            print(f"Downloaded: {protein_name}")
        else:
            print(f"No FASTA file found for protein: {protein_name}")
    except requests.RequestException as e:
        print(f"Error downloading {protein_name}: {e}")

def download_all_fastas(folder_path, output_folder):
    """
    下载文件夹中所有文件对应的蛋白质的 FASTA 文件。

    参数:
        folder_path (str): 包含文件的文件夹路径。
        output_folder (str): 下载的 FASTA 文件保存路径。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 提取蛋白质名称
    protein_names = extract_protein_names(folder_path)

    # 下载每个蛋白质的 FASTA 文件
    for protein_name in protein_names:
        download_fasta(protein_name, output_folder)

# 示例：设置输入和输出文件夹路径
input_folder = "/media/gpux1/Pro1/app/intermediate"  # 替换为包含文件的文件夹路径
output_folder = "/media/gpux1/Pro1/app/intermediate-fasta"  # 替换为保存 FASTA 文件的文件夹路径

# 执行下载
download_all_fastas(input_folder, output_folder)
