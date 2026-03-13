#!/bin/sh 
# ------------------------------------------------
# PBS/Torque 作业参数设置 (PBS Directives)
# ------------------------------------------------
#PBS -N GraphBind_DataIO         
#PBS -q APG                       
#PBS -l select=1:ncpus=4:mem=256gb:ngpus=1
#PBS -o data_io2_output.log        
#PBS -e data_io2_error.log

# ------------------------------------------------
# 环境初始化 (Environment Initialization)
# ------------------------------------------------

# 6. 加载 module 命令所需的Shell环境 (通常保留此行)
source /etc/profile.d/modules.sh

# 7. 加载 GPU 依赖模块 (CUDA 库是 PyTorch 运行所必需的)
# ⚠️ 注意: H100 GPU 需要 CUDA 11.8 或更高版本。
#       您的 PyTorch 1.9.0 是 cu111，可能与 H100 的最佳性能不匹配，但先尝试。
#       如果运行失败，需要升级 PyTorch 到 2.x + cu12.x 版本。
module load cuda/12.1.1

# 8. Conda 初始化：使用您确认的路径
#    这行是关键，用于激活 Conda 环境
source /usr/appli/freeware/Anaconda3/2019.10/etc/profile.d/conda.sh

# 9. 激活您的 Conda 环境
conda activate graphbind_h100

# ------------------------------------------------
# 程序执行 (Execution)
# ------------------------------------------------

# 10. 移动到提交作业的目录（标准操作）
cd $PBS_O_WORKDIR

# 11. 执行您的 Python 脚本
echo "Job started on node: $(hostname)"
python extract_embeddings.py
python plot_tsne.py

echo "Job finished."
