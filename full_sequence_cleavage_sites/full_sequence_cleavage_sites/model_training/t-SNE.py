import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# 加载数据（假设数据已经加载并重塑为合适的形状）
x_features = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")
x_features = x_features.reshape(x_features.shape[0], -1)  # (26844, 30x384)
labels = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")
# 数据标准化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_features)
# 使用 t-SNE 降维至 2 维
tsne = TSNE(n_components=2, random_state=42, n_iter=300, perplexity=30)
x_tsne = tsne.fit_transform(x_scaled)
# 可视化 t-SNE 结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, cmap='viridis', s=5)
plt.colorbar(scatter)  # 为每个点添加颜色条
plt.title('t-SNE Visualization of Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

