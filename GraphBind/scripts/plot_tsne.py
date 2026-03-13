import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

print("加载嵌入向量和标签...")
# [TODO 4] 确保这些路径与步骤二的输出一致
embeddings = np.load("tsne_data/hgnn_phsa_embeddings.npy")
labels = np.load("tsne_data/hgnn_phsa_labels.npy")

print(f"数据加载完成。总残基数: {len(labels)}")

# --- 1. 采样 (t-SNE 无法处理百万级数据！) ---
# 这是一个关键步骤。我们将使用一个“均衡采样”来清晰地显示边界。
N_SAMPLES = 5000 # 每类采样 5000 个点 (总共 10000 个点)

pos_indices = np.where(labels == 1)[0]
neg_indices = np.where(labels == 0)[0]

print(f"阳性 (Binding) 样本数: {len(pos_indices)}")
print(f"阴性 (Non-binding) 样本数: {len(neg_indices)}")

# 随机选择 N_SAMPLES 个索引
if len(pos_indices) > N_SAMPLES:
    pos_indices_sampled = np.random.choice(pos_indices, N_SAMPLES, replace=False)
else:
    pos_indices_sampled = pos_indices # 样本不足，全部使用

if len(neg_indices) > N_SAMPLES:
    neg_indices_sampled = np.random.choice(neg_indices, N_SAMPLES, replace=False)
else:
    neg_indices_sampled = neg_indices # 样本不足，全部使用

# 合并索引
sampled_indices = np.concatenate([pos_indices_sampled, neg_indices_sampled])
np.random.shuffle(sampled_indices) # 再次打乱

X_sampled = embeddings[sampled_indices]
y_sampled = labels[sampled_indices]

print(f"已采样 {len(y_sampled)} 个点进行 t-SNE...")

# --- 2. 运行 t-SNE ---
# (这可能需要几分钟时间)
tsne = TSNE(n_components=2, 
            perplexity=30,     # 经典值
            n_iter=1000,       # 迭代次数
            init='pca',        # 使用 PCA 初始化
            n_jobs=-1,         # 使用所有 CPU 核心
            verbose=1)

print("正在运行 t-SNE (这可能需要几分钟)...")
tsne_results = tsne.fit_transform(X_sampled)

print("t-SNE 运行完毕。")

# --- 3. 绘图 ---
plt.figure(figsize=(10, 8))

# 分别绘制 阴性 (0) 和 阳性 (1)
neg_points = tsne_results[y_sampled == 0]
pos_points = tsne_results[y_sampled == 1]

plt.scatter(neg_points[:, 0], neg_points[:, 1], 
            label="Non-Binding (Label=0)", 
            color="blue", 
            alpha=0.3, # 使用透明度
            s=5)       # 减小点的大小

plt.scatter(pos_points[:, 0], pos_points[:, 1], 
            label="Binding (Label=1)", 
            color="red", 
            alpha=0.5,
            s=5)

plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

output_filename = "tsne_data/tsne_HGNN_PHSA.png"
plt.savefig(output_filename, dpi=300)
print(f"t-SNE 图像已保存到 {output_filename}")