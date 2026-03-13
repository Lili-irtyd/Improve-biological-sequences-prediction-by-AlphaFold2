#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import glob

# 指定输入输出路径
input_dir = "/media/gpux1/Pro1/RNA-117_Test_fasta"
output_dir = "/home/gpux1/RNA-117_Test_pssm"
os.makedirs(output_dir, exist_ok=True)

# 获取所有 fasta 文件（后缀为 .fasta 的）
fasta_files = glob.glob(os.path.join(input_dir, "*.fasta"))

# 遍历每一个 fasta 文件
for fasta_path in fasta_files:
    # 获取不带路径和后缀的文件名，如 "0001_3pla_L"
    base_name = os.path.splitext(os.path.basename(fasta_path))[0]

    # 设置输出文件名，例如 "0001_3pla_L.pssm"
    out_path = os.path.join(output_dir, f"{base_name}.pssm")

    # 构造并执行 psiblast 命令
    blast_cmd = f"/home/gpux1/tools/ncbi-blast/ncbi-blast-2.12.0+/bin/psiblast -query {fasta_path} -db /home/gpux1/blast_db/swissprot/swissprot -num_iterations 3 -out_ascii_pssm {out_path}"
    os.system(blast_cmd)
