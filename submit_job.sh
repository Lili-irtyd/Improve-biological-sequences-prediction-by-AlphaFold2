#!/bin/bash
#PBS -N My_AF2_MOD_Singularity
#PBS -q APG
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -j oe

# --- 运行命令 ---

# 1. 加载您个人的 bash 配置文件 (可能需要它来找到 'module')
source ~/.bashrc

# 2. ‼️‼️ 不再需要加载 Python 模块 ‼️‼️
#    我们将使用容器内部的 Python

# 3. 加载 AlphaFold 模块 (它可能提供了 singularity 命令或设置了路径)
echo "正在加载 AlphaFold 模块 (为 Singularity 做准备)..."
module load AlphaFold/2.3.2 

# 4. 进入工作目录
cd $PBS_O_WORKDIR

# 5. 定义文件和目录路径
echo "正在定义文件和目录路径..."
PROJECT_DIR=$(pwd) # 当前目录就是项目目录
INPUT_FASTA_DIR="${PROJECT_DIR}/input_fastas"
MAIN_OUTPUT_DIR="${PROJECT_DIR}/results"
# ！！！您的核心 Python 脚本，相对于 PROJECT_DIR 的路径！！！
MY_SCRIPT_REL_PATH="run_alphafold.py" 
MY_SCRIPT_ABS_PATH="${PROJECT_DIR}/${MY_SCRIPT_REL_PATH}"

# ！！！Singularity 镜像文件的完整路径 (请确认!)！！！
SIF_IMAGE="/usr/appli/freeware/AlphaFold/2.3.2/bin/alphafold232.cuda1180.H100.sif"

# --- 6. 定义所有数据库的路径 (这部分不变，因为数据库在主机上) ---
echo "正在定义数据库路径..."
PREFIX="/usr/appli/freeware/AlphaFold/2.3.2"
DATA_DIR="${PREFIX}/data"
# ... (省略了所有 UNIREF90_PATH, MGNIFY_PATH 等路径定义，请保留)
UNIREF90_PATH="${PREFIX}/data/uniref90/uniref90.fasta"
MGNIFY_PATH="${PREFIX}/data/mgnify/mgy_clusters.fa"
PDB70_PATH="${PREFIX}/data/pdb70/pdb70"
BFD_PATH="${PREFIX}/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
UNIREF30_PATH="${PREFIX}/data/uniref30/UniRef30_2021_03"
MMCIF_PATH="${PREFIX}/data/pdb_mmcif/mmcif_files"
OBSOLETE_PATH="${PREFIX}/data/pdb_mmcif/obsolete.dat"
UNIPROT_PATH="${PREFIX}/data/uniprot/uniprot.fasta"
PDB_SEQRES_PATH="${PREFIX}/data/pdb_seqres/pdb_seqres.txt"

# --- 7. 循环处理您的每一个 FASTA 文件，使用 singularity exec ---
echo "开始批量处理 FASTA 文件 (使用 Singularity)..."
mkdir -p "$MAIN_OUTPUT_DIR"

for fasta_file in ${INPUT_FASTA_DIR}/*.fasta; do
    
    protein_name=$(basename "${fasta_file}" .fasta)
    output_subdir="${MAIN_OUTPUT_DIR}/${protein_name}"
    mkdir -p "$output_subdir" # 提前创建输出子目录
    
    echo "----------------------------------------------------"
    echo "正在处理: ${protein_name} (使用 Singularity)"
    echo "FASTA 输入 (主机): ${fasta_file}"
    echo "输出目录 (主机): ${output_subdir}"
    echo "----------------------------------------------------"
    
    # ‼️‼️‼️ 这是核心命令：singularity exec ‼️‼️‼️
    singularity exec --nv \
        -B ${INPUT_FASTA_DIR}:/input \
        -B ${output_subdir}:/output \
        -B ${DATA_DIR}:/database \
        -B ${PROJECT_DIR}:/project \
        ${SIF_IMAGE} \
        python3 /project/${MY_SCRIPT_REL_PATH} \
            --fasta_paths=/input/$(basename ${fasta_file}) \
            --output_dir=/output \
            --data_dir=/database \
            --uniref90_database_path=/database/uniref90/uniref90.fasta \
            --mgnify_database_path=/database/mgnify/mgy_clusters.fa \
            --pdb70_database_path=/database/pdb70/pdb70 \
            --bfd_database_path=/database/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
            --uniref30_database_path=/database/uniref30/UniRef30_2021_03 \
            --template_mmcif_dir=/database/pdb_mmcif/mmcif_files \
            --obsolete_pdbs_path=/database/pdb_mmcif/obsolete.dat \
            --uniprot_database_path=/database/uniprot/uniprot.fasta \
            --pdb_seqres_database_path=/database/pdb_seqres/pdb_seqres.txt \
            --max_template_date="2022-12-09" \
            --model_preset="monomer" \
            --db_preset="full_dbs"

done 

echo "所有任务已提交或完成。"