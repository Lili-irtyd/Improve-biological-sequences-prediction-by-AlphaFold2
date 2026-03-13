import os


def extract_fasta_from_txt(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    i = 0
    saved_count = 0
    file_index = 1  # 新加：记录顺序

    while i < len(lines) - 2:
        name_line = lines[i]
        sequence_line = lines[i + 1]
        label_line = lines[i + 2]

        if not name_line.startswith('>'):
            i += 1
            continue

        if not all(c in '01' for c in label_line):
            i += 1
            continue

        name = name_line[1:]  # 去掉开头的 >
        sequence = sequence_line

        # 保存时加上编号，比如 "0001_name.fasta"
        filename = f"{file_index:04d}_{name}.fasta"  # 4位数，不够补0
        fasta_path = os.path.join(output_dir, filename)
        with open(fasta_path, 'w') as fasta_file:
            fasta_file.write(f">{name}\n")
            fasta_file.write(f"{sequence}\n")

        saved_count += 1
        file_index += 1
        i += 3

    print(f"✅ 提取完成！共生成 {saved_count} 个FASTA文件，保存到 {output_dir}")


# === 实际调用 ===
input_file = "/media/gpux1/Pro1/RNA-117_Test.txt"
output_dir = "/media/gpux1/Pro1/RNA-117_Test_fasta"

extract_fasta_from_txt(input_file, output_dir)
