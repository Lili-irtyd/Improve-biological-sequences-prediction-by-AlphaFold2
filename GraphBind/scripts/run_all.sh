#!/bin/bash
# run_all.sh
echo "Step 2: 模型训练"
 #python training.py --ligand DNA --psepos SC --features AF2 --context_radius 20 --trans_anno True --edge_radius 10 --use_GRU True  --apply_edgeattr True --apply_posemb True --aggr sum --hidden_size 128 --lr 0.00003 --batch_size 32 
echo "Step 1: 数据预处理"
 python data_io_af2.py --ligand DNA --psepos SC --features PSSM HMM SS AF --context_radius 20 --trans_anno True
echo "Step 2: 模型训练"
 python training.py --ligand DNA --psepos SC --features "PSSM,HMM,SS,AF" --context_radius 20 --trans_anno True --edge_radius 10 --use_GRU True  --apply_edgeattr True --apply_posemb True --aggr sum --hidden_size 128 --lr 0.00003 --batch_size 32 
