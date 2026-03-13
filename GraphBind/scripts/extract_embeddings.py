import sys
import os
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))

import torch
import numpy as np
from torch_geometric.data import DataLoader
from tqdm import tqdm

# [TODO 1] 导入你所有的GNN模型和数据加载器
from data_io_af2 import NeighResidue3DPoint 
from ModelCode.GN_model_gru import MetaBind_MultiEdges
from training import Config, parse_args, checkargs # 假设你的训练脚本叫 train_hgnn.py

# --- 1. [TODO 2] 在此配置你的 PHSA 实验 ---
# 
# 确保这里的参数与你训练 PHSA 模型时使用的完全一致
#
sys.argv = [
    "extract_embeddings.py",       # 占位符 (sys.argv[0])
    "--ligand", "RNA",             # 你的配体类型
    "--context_radius", "20",      # 你的结构半径
    "--features", "PSSM,HMM,SS,AF", # [关键] 确保这里不包含 AF2，如果你的 PHSA 模型没用 AF2
    "--edge_radius", "10",         # 其他必需参数 (根据你的 train.py 默认值)
    "--hidden_size", "128"          # 确保模型维度一致
]
args = parse_args()
checkargs(args)
opt = Config(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. 加载数据 ---
print(f"加载数据集: {opt.data_root_dir}")
# 我们只关心测试集
test_data = opt.str_dataio(root=opt.data_root_dir, dataset='test')
test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

# --- 3. 加载你训练好的模型 ---
print("加载已训练的 HGNN + PHSA 模型...")

# [TODO 3] 修改为你训练好的 PHSA 模型的 .pth 文件路径
# 这应该是你那个 F1=0.526 的模型的保存路径
MODEL_PATH = "/user1/scl1/zhiqian/GraphBind/Datasets/PRNA/checkpoints/model_PHSA/model/model0.pth" 

# 加载模型 (从 GraphBind 的 train.py 中借鉴)
model, criterion, optimizer, th, _ = torch.load(MODEL_PATH, map_location=device)
model.to(device)
model.eval()

# --- 4. 激活我们添加的“钩子” ---
features_cache = []

def hook_fn(module, input):
    # input 是一个 tuple，第一个元素就是 tensor
    feature = input[0] 
    features_cache.append(feature.detach().cpu())

# 注册钩子到 model.clf 层
handle = model.clf.register_forward_pre_hook(hook_fn)
print("已注册特征提取钩子 (Hook registered).")

# --- 5. 遍历测试集并提取特征 ---
all_embeddings = []
all_labels = []

print("开始提取测试集的 GNN 嵌入向量...")
with torch.no_grad():
    for data in tqdm(test_loader):
        data = data.to(device)
        
        # [修正]：只需要运行模型，不需要接收 embeddings
        # 因为 Hook 会自动把 embeddings 存到 features_cache 列表里！
        _ = model(data) 
        
        # 保存标签
        all_labels.append(data.y.cpu())

# --- 6. 合并并保存 ---
print("合并所有向量...")
final_embeddings = torch.cat(features_cache, dim=0).numpy()
final_labels = torch.cat(all_labels, dim=0).numpy().squeeze() # 压平标签

print(f"提取完成! 嵌入向量 shape: {final_embeddings.shape}")
print(f"标签 shape: {final_labels.shape}")

# [TODO 4] 保存到文件
OUTPUT_DIR = "tsne_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "hgnn_phsa_embeddings.npy"), final_embeddings)
np.save(os.path.join(OUTPUT_DIR, "hgnn_phsa_labels.npy"), final_labels)

print(f"t-SNE 数据已保存到 {OUTPUT_DIR} 目录。")