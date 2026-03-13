#基于梯度（Gradient-based）的显著性分析脚本（这正是 Grad-CAM 的核心原理）。

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from torch_geometric.nn import radius_graph
from tqdm import tqdm
import seaborn as sns

# --- 路径设置 ---
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))

from data_io_af2 import NeighResidue3DPoint 
from ModelCode.GN_model_gru import MetaBind_MultiEdges
from training import Config, parse_args, checkargs 
TARGET_SEQ_IDX = 916 

# 你想要可视化的残基在该蛋白质中的索引 (例如: 0)
TARGET_RES_IDX = 0
# --- 1. 配置与加载 (模拟命令行参数) ---
sys.argv = [
    "visualize_explanation.py",
    "--ligand", "RNA",
    "--context_radius", "20",
    "--features", "PSSM,HMM,SS,AF", # 确保这里包含 AF2 (A2)
    "--edge_radius", "10",
    "--hidden_size", "128"
]

args = parse_args()
checkargs(args)
opt = Config(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. 辅助函数：绘图 ---
def plot_3d_subgraph(center_idx, neighbor_indices, pos, importance_scores, labels, pred_score, seq_id):
    """
    绘制 3D 子图，节点颜色深浅代表重要性
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 获取坐标
    center_pos = pos[center_idx].detach().cpu().numpy()
    neigh_pos = pos[neighbor_indices].detach().cpu().numpy()
    
    # 1. 绘制邻居节点 (根据重要性着色)
    # 归一化重要性分数到 0-1
    scores = importance_scores.detach().cpu().numpy()
    if scores.max() > 0:
        scores = scores / scores.max()
    
    # 使用 Coolwarm colormap (蓝色=不重要, 红色=重要)
    p = ax.scatter(neigh_pos[:, 0], neigh_pos[:, 1], neigh_pos[:, 2], 
                   c=scores, cmap='coolwarm', s=100, alpha=0.8, label='Neighbors')
    
    # 2. 绘制中心节点 (绿色)
    ax.scatter(center_pos[0], center_pos[1], center_pos[2], 
               c='green', s=200, edgecolors='black', label='Target Center', marker='*')

    # 3. 绘制连接边
    for i in range(len(neigh_pos)):
        ax.plot([center_pos[0], neigh_pos[i, 0]], 
                [center_pos[1], neigh_pos[i, 1]], 
                [center_pos[2], neigh_pos[i, 2]], 
                color='gray', alpha=0.3)

    # 标签和美化
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Explanation for Residue {center_idx} (Seq: {seq_id})\nPred Score: {pred_score:.3f} (True Label: {labels[center_idx].item()})\nColor Intensity = Gradient Importance", fontsize=12)
    
    # 添加 Colorbar
    cbar = plt.colorbar(p, ax=ax, shrink=0.6)
    cbar.set_label('Importance (Gradient Magnitude)')
    
    plt.legend()
    plt.tight_layout()
    
    filename = f"explanation_{seq_id}_res{center_idx}.png"
    plt.savefig(filename, dpi=300)
    print(f"可视化图片已保存: {filename}")
    plt.close()

def analyze_feature_importance(node_grads, feature_dim=384):
    """
    分析哪些特征维度最重要
    """
    # node_grads shape: [Num_Neighbors + 1, Feature_Dim]
    # 对所有相关节点求平均梯度绝对值
    avg_grads = torch.mean(node_grads, dim=0).detach().cpu().numpy()
    
    # 找出 Top 10 最重要的特征维度
    top_indices = np.argsort(avg_grads)[::-1][:10]
    top_values = avg_grads[top_indices]
    
    print("\n--- 结论 4: 关键特征维度分析 ---")
    print(f"模型最依赖的 Top 10 特征维度 (共 {len(avg_grads)} 维):")
    for idx, val in zip(top_indices, top_values):
        print(f"  Dimension {idx}: Importance {val:.6f}")
    
    return top_indices

# --- 3. 主逻辑 ---
def main():
    # 加载模型
    print("加载模型...")
    # [TODO] 请修改为你真实的模型路径
    MODEL_PATH = "/user1/scl1/zhiqian/GraphBind/Datasets/PRNA/checkpoints/model_PHSA/model/model0.pth" 
    
    loaded_content = torch.load(MODEL_PATH, map_location=device)
    model = loaded_content[0]
    model.to(device)
    model.eval()

    # 加载测试集
    print("加载测试数据...")
    test_data = opt.str_dataio(root=opt.data_root_dir, dataset='test')
    loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # --- [新增] 指定目标索引 ---
    TARGET_SEQ_IDX = 916   # 你的目标 Seq Index
    TARGET_RES_IDX = 0   # 你的目标 Residue Index

    print(f"正在寻找目标: Seq Index {TARGET_SEQ_IDX}, Residue Index {TARGET_RES_IDX} ...")

    target_found = False
    
    for i, data in enumerate(loader):
        # 1. [关键修改] 如果当前序列不是目标序列，直接跳过
        if i != TARGET_SEQ_IDX:
            continue
            
        print(f"已加载目标序列 (Seq Index: {i})。正在处理...")
        
        # [注意] 下面这一行就是你报错的地方，确保它和上面的 print 在同一列
        data = data.to(device)
        
        # 开启梯度追踪
        data.x.requires_grad = True
        
        # 前向传播
        scores = model(data) # Shape: [N_residues]
        
        # 2. [关键修改] 检查目标残基索引是否越界
        if TARGET_RES_IDX >= len(scores):
            print(f"错误：目标残基索引 {TARGET_RES_IDX} 超出了该蛋白质的长度 ({len(scores)})。")
            return

        # 3. [关键修改] 强制选中目标残基
        target_idx = TARGET_RES_IDX
        score = scores[target_idx]
        
        print(f"\n找到目标残基！")
        print(f"  Seq Index: {i}")
        print(f"  Residue Index: {target_idx}")
        print(f"  Prediction Score: {score.item():.4f}")
        # print(f"  True Label: {data.y[target_idx].item()}") 
        # 注意：如果你的 data.y 是一维的，直接用 data.y[target_idx]；如果是多维可能需要调整
        
        # --- 以下代码负责计算梯度和绘图 ---
        
        # 4. 反向传播
        model.zero_grad()
        score.backward()
        
        # 5. 获取梯度
        gradients = data.x.grad 
        node_importance = torch.norm(gradients, dim=1)
        
        # 6. 确定 3D 邻域
        pos = data.pos
        batch = data.batch
        edge_index = radius_graph(pos, r=opt.radius_list[0], batch=batch, max_num_neighbors=opt.max_nn)
        neighbors = edge_index[1][edge_index[0] == target_idx]
        
        print(f"  该残基在 {opt.radius_list[0]}A 半径内有 {len(neighbors)} 个邻居。")
        
        # 7. 分析结论 3
        if len(neighbors) > 0:
            neigh_imp = node_importance[neighbors].mean().item()
        else:
            neigh_imp = 0.0
            
        center_imp = node_importance[target_idx].item()
        print("\n--- 结论 3: 3D邻域重要性 ---")
        print(f"  中心残基重要性: {center_imp:.4f}")
        print(f"  邻居平均重要性: {neigh_imp:.4f}")
        
        # 8. 分析结论 4
        relevant_indices = torch.cat([neighbors, torch.tensor([target_idx], device=device)])
        relevant_grads = torch.abs(gradients[relevant_indices])
        analyze_feature_importance(relevant_grads)
        
        # 9. 绘图
        plot_3d_subgraph(target_idx, neighbors, pos, node_importance[neighbors], 
                         data.y, score.item(), f"Seq{i}_Res{target_idx}")
        
        target_found = True
        break # 找到并画图后，直接退出程序

    if not target_found:
        print(f"未能在测试集中找到 Seq Index {TARGET_SEQ_IDX}。请检查索引是否超出范围。")

if __name__ == "__main__":
    main()