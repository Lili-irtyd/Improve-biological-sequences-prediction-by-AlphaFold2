from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
x_features = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")
x_features = x_features.reshape(x_features.shape[0], -1)  # (26844, 11520)

# 标准化
scaler = StandardScaler()

def analyze_pca_importance(features):
    """ 对特征进行标准化并进行 PCA 重要性分析 """
    features_scaled = scaler.fit_transform(features)
    pca = PCA()  # 不进行降维，获取所有主成分
    pca.fit(features_scaled)  # 拟合数据
    explained_variance_ratio = pca.explained_variance_ratio_  # 每个主成分解释的方差比例
    components = pca.components_  # 每个主成分的特征向量系数

    return pca, explained_variance_ratio, components

# 分析 AAC 特征的 PCA 重要性
pca_aac, variance_aac, components_aac = analyze_pca_importance(aac_features)
pca_be, variance_be, components_be = analyze_pca_importance(be_features)
pca_cksaap, variance_cksaap, components_cksaap = analyze_pca_importance(cksaap_features)
pca_pssm, variance_pssm, components_pssm = analyze_pca_importance(pssm_features)
pca_x, variance_x, components_x = analyze_pca_importance(x_features)

# 可视化每个主成分的方差解释比例
def plot_variance_ratio(variance_ratios, title):
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(variance_ratios), marker='o')
    plt.title(f"Cumulative Explained Variance by Principal Components - {title}")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.show()

# 绘制每个特征的方差解释比例
plot_variance_ratio(variance_aac, 'AAC')
plot_variance_ratio(variance_be, 'BE')
plot_variance_ratio(variance_cksaap, 'CKSAAP')
plot_variance_ratio(variance_pssm, 'PSSM')
plot_variance_ratio(variance_x, 'AF2 Features')

# 分析并可视化第一主成分中每个特征的重要性
def plot_feature_importance(components, feature_names, title):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, np.abs(components[0]), color='steelblue')
    plt.title(f"Feature Importance for First Principal Component - {title}")
    plt.xlabel("Importance (absolute value)")
    plt.ylabel("Feature")
    plt.show()

# 假设特征名
aac_feature_names = [f"AAC_{i}" for i in range(aac_features.shape[1])]
be_feature_names = [f"BE_{i}" for i in range(be_features.shape[1])]
cksaap_feature_names = [f"CKSAAP_{i}" for i in range(cksaap_features.shape[1])]
pssm_feature_names = [f"PSSM_{i}" for i in range(pssm_features.shape[1])]
x_feature_names = [f"AF2_{i}" for i in range(x_features.shape[1])]

# 绘制特征重要性
plot_feature_importance(components_aac, aac_feature_names, 'AAC')
plot_feature_importance(components_be, be_feature_names, 'BE')
plot_feature_importance(components_cksaap, cksaap_feature_names, 'CKSAAP')
plot_feature_importance(components_pssm, pssm_feature_names, 'PSSM')
plot_feature_importance(components_x, x_feature_names, 'AF2 Features')
