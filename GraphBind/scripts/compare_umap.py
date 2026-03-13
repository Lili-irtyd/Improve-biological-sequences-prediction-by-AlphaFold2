# import sys
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import umap
# from tqdm import tqdm
# from torch_geometric.data import DataLoader

# # --- 路径设置 ---
# sys.path.append(os.path.abspath(''))
# sys.path.append(os.path.abspath('..'))

# # 导入原始模型和配置工具
# from data_io_af2 import NeighResidue3DPoint 
# from ModelCode.GN_model_gru import MetaBind_MultiEdges
# from training import Config, parse_args, checkargs 

# # ==========================================
# # [配置区域] 请在这里填入你的模型路径和参数
# # ==========================================

# # 模型 A (基线: PHSA)
# MODEL_A_CONFIG = {
#     "name": "HGNN + PHSA",
#     "path": "/user1/scl1/zhiqian/GraphBind/Datasets/PRNA/checkpoints/model_PHSA/model/model0.pth", # [TODO] 修改路径
#     "features": "PSSM,HMM,SS,AF", # [TODO] 确保与训练时一致
#     "hidden_size": 128,           # [TODO] 确保与训练时一致
#     "context_radius": 20          # [TODO] 确保与训练时一致
# }

# # 模型 B (挑战者: AF2)
# MODEL_B_CONFIG = {
#     "name": "HGNN + AF2",
#     "path": "/user1/scl1/zhiqian/GraphBind/Datasets/PRNA/checkpoints/model_A2/model/model0.pth",  # [TODO] 修改路径
#     "features": "AF2",            # [TODO] 确保与训练时一致
#     "hidden_size": 128,           # [TODO] 确保与训练时一致 (比如你优化过的256)
#     "context_radius": 20
# }

# # 通用设置
# LIGAND = "RNA"
# EDGE_RADIUS = 10
# BATCH_SIZE = 32
# SAMPLE_SIZE = 20000 # 为了速度和清晰度，每类最多采样的点数 (总点数约 x2)

# # ==========================================

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def extract_features(config_dict):
#     """
#     加载特定配置的模型和数据，并提取嵌入向量
#     """
#     print(f"\n正在处理模型: {config_dict['name']} ...")
    
#     # 1. 动态构造 sys.argv 以欺骗 parse_args
#     # 我们需要重新初始化 Config 对象，因为不同模型的 features 可能不同
#     sys.argv = [
#         "compare_umap.py",
#         "--ligand", LIGAND,
#         "--context_radius", str(config_dict['context_radius']),
#         "--features", config_dict['features'],
#         "--edge_radius", str(EDGE_RADIUS),
#         "--hidden_size", str(config_dict['hidden_size'])
#     ]
    
#     args = parse_args()
#     checkargs(args)
#     opt = Config(args)
    
#     # 2. 加载数据 (Test Set)
#     print("加载测试集数据...")
#     test_data = opt.str_dataio(root=opt.data_root_dir, dataset='test')
#     loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
#     # 3. 加载模型
#     print(f"加载权重: {config_dict['path']}")
#     if not os.path.exists(config_dict['path']):
#         raise FileNotFoundError(f"找不到模型文件: {config_dict['path']}")
        
#     loaded_content = torch.load(config_dict['path'], map_location=device)
#     model = loaded_content[0] # model is the first element
#     model.to(device)
#     model.eval()
    
#     # 4. 注册 Hook 截获特征
#     features_cache = []
#     def hook_fn(module, input):
#         # input[0] 是进入分类器之前的 Latent Vector
#         features_cache.append(input[0].detach().cpu())

#     handle = model.clf.register_forward_pre_hook(hook_fn)
    
#     # 5. 推理
#     labels_cache = []
#     print("提取特征中...")
#     with torch.no_grad():
#         for data in tqdm(loader):
#             data = data.to(device)
#             _ = model(data) # Hook 会自动工作
#             labels_cache.append(data.y.cpu())
            
#     handle.remove() # 移除 Hook
    
#     # 6. 合并数据
#     X = torch.cat(features_cache, dim=0).numpy()
#     y = torch.cat(labels_cache, dim=0).numpy().squeeze()
    
#     print(f"提取完成。Shape: {X.shape}")
#     return X, y

# def plot_umap(ax, X, y, title):
#     """
#     在指定的子图上绘制 UMAP
#     """
#     print(f"正在为 [{title}] 计算 UMAP (这可能需要几分钟)...")
    
#     # 1. 降采样 (如果数据量太大，UMAP会很慢且图会很乱)
#     pos_idx = np.where(y == 1)[0]
#     neg_idx = np.where(y == 0)[0]
    
#     # 随机采样
#     if len(pos_idx) > SAMPLE_SIZE:
#         pos_idx = np.random.choice(pos_idx, SAMPLE_SIZE, replace=False)
#     if len(neg_idx) > SAMPLE_SIZE:
#         neg_idx = np.random.choice(neg_idx, SAMPLE_SIZE, replace=False)
        
#     sample_idx = np.concatenate([pos_idx, neg_idx])
#     np.random.shuffle(sample_idx)
    
#     X_sample = X[sample_idx]
#     y_sample = y[sample_idx]
    
#     # 2. 运行 UMAP
#     # reducer = umap.UMAP(n_neighbors=30, # 增加邻居数以保留更多全局结构
#     #                     min_dist=0.3,   # 控制点的紧凑程度
#     #                     metric='cosine', # 余弦距离通常在高维空间更好
#     #                     random_state=42)
#     reducer = umap.UMAP(
#     n_neighbors=50,      # 增大：从30增加到50或80，增强全局分离
#     min_dist=0.1,        # 减小：从0.3减小到0.1，让点更聚拢
#     metric='cosine',     # 保持：深度学习特征推荐使用余弦距离
#     n_components=2,      
#     random_state=42,     # 固定随机种子，保证结果可复现
#     init='pca'           # 建议添加：使用PCA初始化通常比随机初始化能更好地保留全局分离度
# )
#     embedding = reducer.fit_transform(X_sample)
    
#     # 3. 绘图
#     # 绘制背景 (非结合)
#     ax.scatter(embedding[y_sample==0, 0], embedding[y_sample==0, 1], 
#                c='royalblue', s=1, alpha=0.3, label='Non-Binding')
#     # 绘制前景 (结合)
#     ax.scatter(embedding[y_sample==1, 0], embedding[y_sample==1, 1], 
#                c='crimson', s=2, alpha=0.6, label='Binding')
    
#     ax.set_title(title, fontsize=14)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # ax.legend(loc='upper right', markerscale=5)

# def main():
#     # 1. 提取两组特征
#     X_a, y_a = extract_features(MODEL_A_CONFIG)
    
#     # 为了避免内存溢出，可能需要清理一下
#     # import gc; gc.collect()
    
#     X_b, y_b = extract_features(MODEL_B_CONFIG)
    
#     # 2. 绘图
#     print("\n开始绘图...")
#     fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
#     plot_umap(axes[0], X_a, y_a, MODEL_A_CONFIG['name'])
#     plot_umap(axes[1], X_b, y_b, MODEL_B_CONFIG['name'])
    
#     # 统一图例
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=14, markerscale=5)
    
#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.1) # 为图例留空间
    
#     save_path = "comparison_umap.png"
#     plt.savefig(save_path, dpi=300)
#     print(f"对比图已保存至: {save_path}")

# if __name__ == "__main__":
#     main()


import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add  # [新增] 用于实现 Sum Pooling

# --- 路径设置 ---
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))

# 导入原始模型和配置工具
from data_io_af2 import NeighResidue3DPoint 
from ModelCode.GN_model_gru import MetaBind_MultiEdges
from training import Config, parse_args, checkargs 

# ==========================================
# [配置区域] 
# ==========================================

# 配置 A: 用于提取 "Raw AF2 Features" (无需模型路径，只需特征类型)
# 对应论文 Figure 10A
RAW_AF2_CONFIG = {
    "name": "Raw AF2 Features (Sum Pooling)",
    "features": "AF2",      # [关键] 必须是 AF2
    "context_radius": 20,
    "hidden_size": 128      # 这里的 hidden_size 不影响 raw 提取，但也需占位
}

# 配置 B: 用于提取 "Latent Graph Features" (训练好的模型)
# 对应论文 Figure 10B
TRAINED_MODEL_CONFIG = {
    "name": "Learned Latent Features (GraphBind)",
    "path": "/user1/scl1/zhiqian/GraphBind/Datasets/PRNA/checkpoints/model_A2/model/model0.pth",  # [TODO] 你的 AF2 模型路径
    "features": "AF2",      # [关键] 确保与训练时一致
    "hidden_size": 128,     # [关键] 确保与训练时一致 (你提到可能是 256)
    "context_radius": 20
}

# 通用设置
LIGAND = "RNA"
EDGE_RADIUS = 10
BATCH_SIZE = 32
SAMPLE_SIZE = 20000 

# ==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataloader(config_dict):
    """
    辅助函数：根据配置初始化数据加载器
    """
    # 动态构造 sys.argv 以欺骗 parse_args
    sys.argv = [
        "compare_umap.py",
        "--ligand", LIGAND,
        "--context_radius", str(config_dict['context_radius']),
        "--features", config_dict['features'],
        "--edge_radius", str(EDGE_RADIUS),
        "--hidden_size", str(config_dict['hidden_size'])
    ]
    
    args = parse_args()
    checkargs(args)
    opt = Config(args)
    
    print(f"[{config_dict['name']}] 加载测试集数据...")
    test_data = opt.str_dataio(root=opt.data_root_dir, dataset='test')
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return loader

def extract_raw_features(config_dict):
    """
    [新增函数] 提取原始特征 (Raw Graph Features)
    逻辑：不经过模型，直接对 data.x 进行 Sum Pooling
    """
    print(f"\n正在提取原始特征: {config_dict['name']} ...")
    loader = get_dataloader(config_dict)
    
    features_list = []
    labels_list = []
    
    print("计算 Raw Features (Sum Pooling)...")
    for data in tqdm(loader):
        data = data.to(device)
        
        # data.x 的形状是 [总节点数, 特征维度]
        # data.batch 的形状是 [总节点数]，指示每个节点属于哪个样本(图)
        
        # --- 核心逻辑：论文中的 Sum Pooling ---
        # "sum of the raw feature vectors of all nodes in a graph"
        # 使用 scatter_add 根据 batch index 对 x 进行求和
        raw_graph_vec = scatter_add(data.x, data.batch, dim=0) 
        
        features_list.append(raw_graph_vec.cpu().numpy())
        labels_list.append(data.y.cpu().numpy())
        
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0).squeeze()
    
    print(f"原始特征提取完成。Shape: {X.shape}")
    return X, y

def extract_latent_features(config_dict):
    """
    [原有函数] 提取模型学习到的潜在特征 (Latent Graph Features)
    逻辑：经过模型，Hook 截取分类器前的向量
    """
    print(f"\n正在提取潜在特征: {config_dict['name']} ...")
    loader = get_dataloader(config_dict)
    
    # 加载模型
    print(f"加载权重: {config_dict['path']}")
    if not os.path.exists(config_dict['path']):
        raise FileNotFoundError(f"找不到模型文件: {config_dict['path']}")
        
    loaded_content = torch.load(config_dict['path'], map_location=device)
    model = loaded_content[0]
    model.to(device)
    model.eval()
    
    # 注册 Hook
    features_cache = []
    def hook_fn(module, input):
        # input[0] 是进入 clf 之前的 Tensor
        features_cache.append(input[0].detach().cpu())

    handle = model.clf.register_forward_pre_hook(hook_fn)
    
    # 推理
    labels_cache = []
    print("模型推理中...")
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            _ = model(data)
            labels_cache.append(data.y.cpu())
            
    handle.remove()
    
    X = torch.cat(features_cache, dim=0).numpy()
    y = torch.cat(labels_cache, dim=0).numpy().squeeze()
    
    print(f"潜在特征提取完成。Shape: {X.shape}")
    return X, y

def plot_umap(ax, X, y, title):
    """
    绘制 UMAP (保持不变，微调参数)
    """
    print(f"正在为 [{title}] 计算 UMAP...")
    
    # 降采样
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    if len(pos_idx) > SAMPLE_SIZE:
        pos_idx = np.random.choice(pos_idx, SAMPLE_SIZE, replace=False)
    if len(neg_idx) > SAMPLE_SIZE:
        neg_idx = np.random.choice(neg_idx, SAMPLE_SIZE, replace=False)
        
    sample_idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(sample_idx)
    
    X_sample = X[sample_idx]
    y_sample = y[sample_idx]
    
    # UMAP 配置
    reducer = umap.UMAP(
        n_neighbors=50,      
        min_dist=0.1,        
        metric='cosine',     
        n_components=2,      
        random_state=42,     
        init='pca'           
    )
    embedding = reducer.fit_transform(X_sample)
    
    # 绘图
    # 0: Non-binding (蓝色)
    ax.scatter(embedding[y_sample==0, 0], embedding[y_sample==0, 1], 
               c='#1f77b4', s=2, alpha=0.2, label='Non-Binding') # 调低 alpha 让重叠更明显
    # 1: Binding (红色)
    ax.scatter(embedding[y_sample==1, 0], embedding[y_sample==1, 1], 
               c='#d62728', s=4, alpha=0.6, label='Binding')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

def main():
    # 1. 提取图 A: 原始特征 (Raw AF2 Sum)
    X_raw, y_raw = extract_raw_features(RAW_AF2_CONFIG)
    
    # 2. 提取图 B: 学习后的特征 (Latent Features)
    X_latent, y_latent = extract_latent_features(TRAINED_MODEL_CONFIG)
    
    # 3. 绘图
    print("\n开始生成对比图...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    plot_umap(axes[0], X_raw, y_raw, "Raw Graph Features (AF2 Sum)")
    plot_umap(axes[1], X_latent, y_latent, "Latent Features (GraphBind Learned)")
    
    # 统一图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=16, markerscale=6)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) 
    
    save_path = "Figure10_Reproduction.png"
    plt.savefig(save_path, dpi=300)
    print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    main()