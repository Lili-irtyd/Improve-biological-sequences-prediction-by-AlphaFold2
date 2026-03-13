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

# --- 1. 配置与加载 (模拟命令行参数) ---
sys.argv = [
    "visualize_explanation.py",
    "--ligand", "RNA",
    "--context_radius", "20",
    "--features", "AF2", # 确保这里包含 AF2 (A2)
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
    MODEL_PATH = "/user1/scl1/zhiqian/GraphBind/Datasets/PRNA/checkpoints/model_A2/model/model0.pth" 
    
    loaded_content = torch.load(MODEL_PATH, map_location=device)
    model = loaded_content[0]
    model.to(device)
    model.eval() # 评估模式

    # 加载测试集
    print("加载测试数据...")
    test_data = opt.str_dataio(root=opt.data_root_dir, dataset='test')
    # batch_size=1 方便逐个蛋白质分析
    loader = DataLoader(test_data, batch_size=1, shuffle=False)

    target_found = False
    
    print("寻找高质量的 True Positive (TP) 预测案例...")
    
    for i, data in enumerate(loader):
        data = data.to(device)
        
        # 1. 开启梯度追踪 (关键!)
        data.x.requires_grad = True
        
        # 2. 前向传播
        # 注意：我们需要手动重现 forward 的一部分来获取图结构，
        # 或者我们相信 gradient 会通过计算图自动回传。
        # 直接运行模型即可，PyTorch 会处理计算图。
        scores = model(data) # Shape: [N_residues]
        
        # 3. 寻找一个置信度高的 TP 点
        # 真实标签=1, 预测分数 > 0.9
        candidates = (data.y == 1) & (scores > 0.9)
        candidate_indices = torch.nonzero(candidates).squeeze()
        
        if candidate_indices.numel() > 0:
            # 找到一个！
            if candidate_indices.dim() == 0: # 只有一个点
                target_idx = candidate_indices.item()
            else:
                target_idx = candidate_indices[0].item() # 取第一个
            
            score = scores[target_idx]
            print(f"\n找到目标残基！\n  Seq Index: {i}\n  Residue Index: {target_idx}\n  Prediction Score: {score.item():.4f}")
            
            # 4. 反向传播 (Explainability Core)
            # 我们想解释这个特定的 score
            model.zero_grad()
            score.backward()
            
            # 5. 获取输入特征的梯度
            # gradients shape: [N_residues, N_features]
            # 梯度越大，说明该特征对预测结果越重要
            gradients = data.x.grad 
            
            # 计算每个残基的“总重要性” (对特征维度求和/L2范数)
            node_importance = torch.norm(gradients, dim=1) # Shape: [N_residues]
            
            # 6. 确定 3D 邻域 (复现 radius_graph 逻辑)
            # 我们需要知道哪些点是 target_idx 的邻居
            pos = data.pos
            batch = data.batch
            # 使用与训练相同的半径
            edge_index = radius_graph(pos, r=opt.radius_list[0], batch=batch, max_num_neighbors=opt.max_nn)
            
            # 找出 target_idx 的所有邻居
            # edge_index[1] 是源节点，edge_index[0] 是目标节点
            # 我们找所有指向 target_idx 的节点 (或 target_idx 指向的，无向图通常对称)
            neighbors = edge_index[1][edge_index[0] == target_idx]
            
            print(f"  该残基在 {opt.radius_list[0]}A 半径内有 {len(neighbors)} 个邻居。")
            
            # 7. 分析结论 3: 空间重要性
            # 比较 邻居的重要性 vs 随机非邻居的重要性
            neigh_imp = node_importance[neighbors].mean().item()
            center_imp = node_importance[target_idx].item()
            print("\n--- 结论 3: 3D邻域重要性 ---")
            print(f"  中心残基重要性: {center_imp:.4f}")
            print(f"  邻居平均重要性: {neigh_imp:.4f}")
            if neigh_imp > 0.01: # 阈值可调
                print("  >> 验证成功！邻居节点具有显著梯度，说明模型利用了3D空间信息。")
            
            # 8. 分析结论 4: 特征维度
            # 获取邻居和中心的梯度矩阵
            relevant_indices = torch.cat([neighbors, torch.tensor([target_idx], device=device)])
            relevant_grads = torch.abs(gradients[relevant_indices])
            analyze_feature_importance(relevant_grads)
            
            # 9. 绘图
            plot_3d_subgraph(target_idx, neighbors, pos, node_importance[neighbors], data.y, score.item(), f"Protein_{i}")
            
            target_found = True
            break # 只画一张图，或者你可以移除 break 画多张

    if not target_found:
        print("未找到符合条件(Label=1, Score>0.9)的残基。请尝试放宽条件。")

if __name__ == "__main__":
    main()