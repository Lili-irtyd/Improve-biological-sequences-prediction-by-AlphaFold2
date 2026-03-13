#!/bin/sh 
# ------------------------------------------------
# PBS/Torque 作业参数设置 (PBS Directives)
# ------------------------------------------------
#PBS -N GraphBind_DataIO         
#PBS -q APG                       
#PBS -l select=1:ncpus=4:mem=256gb:ngpus=1
#PBS -o data_io_output.log        
#PBS -e data_io_error.logs

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
#python training.py --ligand DNA --psepos SC --features AF2 --context_radius 20 --trans_anno True --edge_radius 10 --use_GRU True  --apply_edgeattr True --apply_posemb True --aggr sum --hidden_size 256 --lr 0.0001 --batch_size 32
#python data_io_af2.py --ligand DNA --psepos SC --features PSSM HMM SS AF AF2 --context_radius 20 --trans_anno True
#python training.py --ligand DNA --psepos SC --features "PSSM,HMM,SS,AF,AF2" --context_radius 20 --trans_anno True --edge_radius 10 --use_GRU True  --apply_edgeattr True --apply_posemb True --aggr sum --hidden_size 128 --lr 0.00003 --batch_size 32 
# python data_io_af2.py --ligand DNA --psepos SC --features PSSM HMM SS AF --context_radius 20 --trans_anno True
#python compare_umap.py
#python training.py --ligand RNA --psepos SC --features "AF2" --context_radius 20 --trans_anno True --edge_radius 10 --use_GRU True  --apply_edgeattr True --apply_posemb True --aggr sum --hidden_size 128 --lr 0.00003 --batch_size 32 --epoch 5

# python training.py --ligand RNA --psepos SC --features "AF2" \
#     --context_radius 20 --trans_anno True --edge_radius 10 \
#     --use_GRU True --apply_edgeattr True --apply_posemb True \
#     --aggr sum --hidden_size 128 --lr 0.00003 --batch_size 32 \
#     --epoch 30 --gru_steps 2

python analysis_easy_hard.py
#CUDA_LAUNCH_BLOCKING=1 python train_gtm.py --ligand RNA --features AF2 --context_radius 20 --hidden_size 128 --lr 0.00003 --batch_size 32
#python3 train_hgnn_strategy.py
#python data_io_af2.py --ligand RNA --psepos SC --features AF2 --context_radius 20 --trans_anno True
# python -m torch.distributed.run --nproc_per_node=1 training_cv_ddp.py --ligand DNA --features  AF2 --context_radius 20 --psepos SC --edge_radius 10 --hidden_size 256 --lr 0.0001 --batch_size 32 --epoch 30 --folds 5 --gru_steps 4
# python -m torch.distributed.run --nproc_per_node=1 training_cv_ddp.py --ligand DNA --features  PSSM HMM SS AF --context_radius 20 --psepos SC --edge_radius 10 --hidden_size 256 --lr 0.0001 --batch_size 32 --epoch 30 --folds 5 --gru_steps 4

echo "Job finished."
