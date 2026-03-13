#!/bin/bash
#PBS -N My_AF2_MOD_Batch      # 任务名称
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1  # 申请计算资源
#PBS -l walltime=48:00:00      # 任务最长运行时间

# --- 1. 定义工作目录和输入/输出文件夹 ---
#    请根据您的实际情况修改这些路径
PROJECT_DIR="$HOME/alphafold"
INPUT_FASTA_DIR="$PROJECT_DIR/input_fastas"
MAIN_OUTPUT_DIR="$PROJECT_DIR/output"

# --- 2. 加载 AlphaFold 环境模块 (关键步骤!) ---
echo "正在加载 AlphaFold 模块..."
# ！！！请替换成您用 module avail 找到的正确模块名！！！
module load AlphaFold/2.3.2

# --- 3. 定义所有数据库和工具的路径 (从超算文档中复制) ---
echo "正在定义数据库路径..."
# ！！！这个 PREFIX 路径也请根据超算文档确认！！！
PREFIX="/usr/appli/freeware/AlphaFold/2.3.2"

DATA_DIR="${PREFIX}/data"
UNIREF90_PATH="${PREFIX}/data/uniref90/uniref90.fasta"
MGNIFY_PATH="${PREFIX}/data/mgnify/mgy_clusters.fa"
PDB70_PATH="${PREFIX}/data/pdb70/pdb70"
BFD_PATH="${PREFIX}/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
UNIREF30_PATH="${PREFIX}/data/uniref30/UniRef30_2021_03"
MMCIF_PATH="${PREFIX}/data/pdb_mmcif/mmcif_files"
OBSOLETE_PATH="${PREFIX}/data/pdb_mmcif/obsolete.dat"
UNIPROT_PATH="${PREFIX}/data/uniprot/uniprot.fasta"
PDB_SEQRES_PATH="${PREFIX}/data/pdb_seqres/pdb_seqres.txt"

# --- 4. 循环处理您的每一个 FASTA 文件 (这替代了您的批量脚本) ---
echo "开始批量处理 FASTA 文件..."
mkdir -p "$MAIN_OUTPUT_DIR"

# 使用 for 循环遍历输入文件夹里所有 .fasta 文件
for fasta_file in ${INPUT_FASTA_DIR}/*.fasta; do

    # 提取蛋白质名字，为它创建一个独立的输出子目录
    protein_name=$(basename "${fasta_file}" .fasta)
    output_subdir="${MAIN_OUTPUT_DIR}/${protein_name}"

    echo "----------------------------------------------------"
    echo "正在处理: ${protein_name} (保存至: ${output_subdir})"
    echo "----------------------------------------------------"

    # 调用您修改过的 Python 脚本 (my_core_af2.py)，并将所有路径作为参数传入
    python ${PROJECT_DIR}/run_alphafold.py \
        --fasta_paths="${fasta_file}" \
        --output_dir="${output_subdir}" \
        --data_dir="${DATA_DIR}" \
        --uniref90_database_path="${UNIREF90_PATH}" \
        --mgnify_database_path="${MGNIFY_PATH}" \
        --pdb70_database_path="${PDB70_PATH}" \
        --bfd_database_path="${BFD_PATH}" \
        --uniref30_database_path="${UNIREF30_PATH}" \
        --template_mmcif_dir="${MMCIF_PATH}" \
        --obsolete_pdbs_path="${OBSOLETE_PATH}" \
        --uniprot_database_path="${UNIPROT_PATH}" \
        --pdb_seqres_database_path="${PDB_SEQRES_PATH}" \
        --max_template_date="2022-12-09" \
        --model_preset="monomer" \
        --db_preset="full_dbs"
done

echo "所有任务已提交或完成。"