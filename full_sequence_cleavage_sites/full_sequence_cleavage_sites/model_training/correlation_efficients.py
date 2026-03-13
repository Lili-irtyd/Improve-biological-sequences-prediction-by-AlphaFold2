import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split

def load_features():
    """加载特征数据，并确保类别平衡"""
    aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
    be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
    cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
    pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
    labels = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")
    x_features = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")
    x_features = x_features.reshape(x_features.shape[0], -1)

    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    np.random.seed(42)
    selected_neg_indices = np.random.choice(neg_indices, size=500, replace=False)
    selected_indices = np.concatenate([pos_indices, selected_neg_indices])

    # 计算各个特征集的列范围
    aac_range = (0, aac_features.shape[1])
    be_range = (aac_range[1], aac_range[1] + be_features.shape[1])
    pssm_range = (be_range[1], be_range[1] + pssm_features.shape[1])
    cksaap_range = (pssm_range[1], pssm_range[1] + cksaap_features.shape[1])
    x_range = (cksaap_range[1], cksaap_range[1] + x_features.shape[1])

    # 先正确索引每个特征集的行
    aac_selected = aac_features[selected_indices]
    be_selected = be_features[selected_indices]
    pssm_selected = pssm_features[selected_indices]
    cksaap_selected = cksaap_features[selected_indices]  # 修正这里，去掉错误索引
    x_selected = x_features[selected_indices]

    # 合并特征
    base_features = np.hstack([aac_selected, be_selected, pssm_selected, cksaap_selected, x_selected])

    feature_ranges = {
        "AAC": aac_range,
        "BE": be_range,
        "PSSM": pssm_range,
        "CKSAAP": cksaap_range,
        "X_Features": x_range
    }

    return base_features, labels[selected_indices], feature_ranges

# 1️⃣ 解析 txt 文件，获取 top 10 特征索引
def read_top10_features(filename):
    """从 txt 文件中提取 top 10 特征索引"""
    feature_indices = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"Feature (\d+)", line)
            if match:
                feature_indices.append(int(match.group(1)))
    return feature_indices

# 2️⃣ 读取 BE 和 X_Features 的 top10 特征索引
top10_BE = read_top10_features("/home/gpux1/CCPR/full_sequence_cleavage_sites/model_training/top10_features.txt")       # 第一个特征集
top10_X = read_top10_features("/home/gpux1/CCPR/full_sequence_cleavage_sites/model_training/top10_features_5.txt")         # 第二个特征集

# 3️⃣ 提取特征数据
features, labels, feature_ranges = load_features()
# 合并两个特征集的特征索引
selected_features = top10_BE + top10_X
x_selected = features[:, selected_features]
y = labels # 提取数据

# 4️⃣ 计算相关性矩阵
df_top20 = pd.DataFrame(x_selected, columns=[f"Feature {idx}" for idx in selected_features])
corr_matrix = df_top20.corr(method='pearson')  # 计算 Pearson 相关系数

# 5️⃣ 绘制更大的相关性热力图
plt.figure(figsize=(16, 14))  # 增大图形尺寸

# 设置颜色区分不同特征集
feature_labels = [f"BE {idx}" if idx in top10_BE else f"X_Features {idx}" for idx in selected_features]

# 绘制热力图，增加边距
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5,
            xticklabels=feature_labels, yticklabels=feature_labels, cbar_kws={'label': 'Pearson Correlation'})

# 调整x、y轴标签的旋转角度和字体大小
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# 设置标题并调整布局
plt.title("Correlation Heatmap between BE and X_Features (Top 10 each)", fontsize=14)

# 增加紧凑布局，避免标签被切割
plt.tight_layout(pad=5.0)  # 增加pad参数，确保标签不被裁剪

plt.show()
