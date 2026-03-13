import os

# 输入 TXT 文件路径
input_file = "/media/gpux1/Pro1/GraphBind/Datasets/PHEM/HEM-175_Train.txt"

# 输出文件夹
output_dir = "/media/gpux1/Pro1/GraphBind/Datasets/PHEM/fasta_files_train"
os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith(">"):  # ID 行
        seq_id = line[1:]  # 去掉 >
        seq = lines[i+1].strip()  # 序列行
        # 可选：跳过标签行（lines[i+2]）
        fasta_file = os.path.join(output_dir, f"{seq_id}.fasta")
        with open(fasta_file, "w") as out_f:
            out_f.write(f">{seq_id}\n")
            # 每行写60个字母，符合FASTA规范
            for j in range(0, len(seq), 60):
                out_f.write(seq[j:j+60] + "\n")
        i += 3  # 跳到下一条记录
    else:
        i += 1