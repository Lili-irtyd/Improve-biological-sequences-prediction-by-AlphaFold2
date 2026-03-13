from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# 设定 AUC 评分标准
auc_scorer = make_scorer(roc_auc_score)

from xgboost import XGBClassifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")


# 加载特征数据
aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
labels = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")
X = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")

# 变换数据形状
X_reshaped = X.reshape(X.shape[0], -1)
high_dim_features = np.hstack([be_features, pssm_features, cksaap_features, X_reshaped])  # 所有高维特征

# PCA 降维
pca = PCA(n_components=230)
X_PCA = pca.fit_transform(high_dim_features)

# 定义不同的特征集
feature_sets = {
    "Original Features + X_reshaped": np.hstack([aac_features, high_dim_features]),
    "Original Features + PCA(X)": np.hstack([aac_features, X_PCA]),
    "Only Original Features": aac_features
}

# 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_means = []
auc_stds = []
feature_labels = []

for name, X in feature_sets.items():
    cv_scores = cross_val_score(xgb, X, labels, cv=cv, scoring=auc_scorer)
    auc_means.append(cv_scores.mean())
    auc_stds.append(cv_scores.std())
    feature_labels.append(name)
    print(f"{name} - Mean AUC: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")

# 可视化 AUC 结果
plt.bar(feature_labels, auc_means, yerr=auc_stds, capsize=5, color='skyblue')
plt.xlabel('Feature Set')
plt.ylabel('Mean AUC')
plt.title('AUC Comparison for Different Feature Sets')
plt.xticks(rotation=30)
plt.show()
