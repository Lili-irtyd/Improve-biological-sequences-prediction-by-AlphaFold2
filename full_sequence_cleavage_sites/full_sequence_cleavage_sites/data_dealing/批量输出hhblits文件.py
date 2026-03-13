import os
import subprocess
import numpy as np
from pathlib import Path


def parse_hhm_to_npy(hhm_path, npy_path):
    with open(hhm_path, 'r') as file:
        lines = file.readlines()

    # Skip header until 'HMM' line
    while lines and not lines[0].startswith('HMM'):
        lines.pop(0)

    if not lines:
        raise ValueError(f"{hhm_path} does not contain HMM section.")

    lines.pop(0)  # remove 'HMM'

    hhm_data = []
    for line in lines:
        if line.startswith('//'):
            break
        if line.strip() == '' or line.startswith('#'):
            continue
        try:
            values = [float(x.replace('*', '9999.0')) for x in line.strip().split()]
            if len(values) == 10:  # one line of the profile
                hhm_data.append(values)
        except ValueError:
            continue  # in case of format issues

    np.save(npy_path, np.array(hhm_data, dtype=np.float32))


# 配置路径
hhblits_path = "/usr/local/bin/hhblits"
db_path = "/media/gpux1/Pro1/DOWNLOAD_DIR/uniref30/UniRef30_2021_03"
input_dirs = ["/media/gpux1/Pro1/RNA-117_Test_fasta"]
output_dirs = ["/home/gpux1/RNA-117_Test_hhm"]

for input_dir, output_dir in zip(input_dirs, output_dirs):
    os.makedirs(output_dir, exist_ok=True)

    for fasta_file in Path(input_dir).glob("*.fasta"):
        base_name = fasta_file.stem
        hhm_file = Path(output_dir) / f"{base_name}.hhm"
        npy_file = Path(output_dir) / f"{base_name}.hhm.npy"

        print(f"🔄 Processing {base_name}...")

        command = [
            hhblits_path,
            "-i", str(fasta_file),
            "-d", db_path,
            "-ohhm", str(hhm_file),
            "-n", "3"
        ]

        try:
            subprocess.run(command, check=True)
            print(f"✅ HHM file saved: {hhm_file}")

            parse_hhm_to_npy(hhm_file, npy_file)
            print(f"📦 Converted to NumPy: {npy_file}")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error running hhblits for {base_name}: {e}")
        except Exception as e:
            print(f"❌ Error converting {hhm_file} to .npy: {e}")
