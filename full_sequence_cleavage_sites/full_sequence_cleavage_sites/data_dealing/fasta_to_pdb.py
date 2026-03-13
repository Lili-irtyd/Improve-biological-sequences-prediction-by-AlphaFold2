import os
import requests
from pathlib import Path
import time

# 输入和输出目录
input_fasta_dir = Path("/media/gpux1/Pro1/RNA-495_Train_fasta")
output_pdb_dir = Path("/media/gpux1/Pro1/train_AF2_pdb")

# 创建输出目录
output_pdb_dir.mkdir(parents=True, exist_ok=True)

# 辅助函数：PDB ID -> UniProt ID
def get_uniprot_id_from_pdb(pdb_id):
    pdb_id = pdb_id.lower()
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # 提取第一个 UniProt ID
        mappings = data.get(pdb_id, {}).get("UniProt", {})
        if mappings:
            first_uniprot_id = list(mappings.keys())[0]
            return first_uniprot_id
        else:
            return None

    except requests.exceptions.RequestException as e:
        print(f"Failed to get UniProt ID for PDB {pdb_id}: {e}")
        return None

# 遍历所有 .fasta 文件，按顺序排列
for fasta_file in sorted(input_fasta_dir.glob("*.fasta")):
    filename_parts = fasta_file.stem.split("_")

    if len(filename_parts) < 2:
        print(f"Skipping file {fasta_file.name}: cannot parse PDB ID.")
        continue

    pdb_id = filename_parts[1].lower()

    print(f"Getting UniProt ID for PDB {pdb_id}...")
    uniprot_id = get_uniprot_id_from_pdb(pdb_id)

    if uniprot_id is None:
        print(f"⚠️ No UniProt mapping found for {pdb_id}. Skipping.")
        continue

    print(f"Found UniProt ID: {uniprot_id}. Downloading AlphaFold structure...")

    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    pdb_output_path = output_pdb_dir / f"{fasta_file.stem}.pdb"

    try:
        response = requests.get(pdb_url)
        response.raise_for_status()

        with open(pdb_output_path, "wb") as f:
            f.write(response.content)

        print(f"✅ Saved AlphaFold PDB to: {pdb_output_path}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download AlphaFold PDB for {uniprot_id}: {e}")

    # Sleep to be polite to API servers (optional)
    time.sleep(0.5)

print("✅ All downloads finished.")
