#####使用lasso+sigmoid完成二分类
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_features(use_x_features=False):
    """加载特征数据，并确保类别平衡"""
    try:
        aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
        be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
        cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
        pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
        labels = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")

        if use_x_features:
            x_features = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")
            x_features = x_features.reshape(x_features.shape[0], -1)
        else:
            x_features = None
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None

    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    np.random.seed(42)
    selected_neg_indices = np.random.choice(neg_indices, size=500, replace=False)
    selected_indices = np.concatenate([pos_indices, selected_neg_indices])

    aac_range = (0, aac_features.shape[1])
    be_range = (aac_range[1], aac_range[1] + be_features.shape[1])
    pssm_range = (be_range[1], be_range[1] + pssm_features.shape[1])
    cksaap_range = (pssm_range[1], pssm_range[1] + cksaap_features.shape[1])

    base_features = np.hstack([aac_features[selected_indices], be_features[selected_indices],
                               pssm_features[selected_indices], cksaap_features[selected_indices]])

    if use_x_features and x_features is not None:
        x_range = (cksaap_range[1], cksaap_range[1] + x_features.shape[1])
        base_features = np.hstack([base_features, x_features[selected_indices]])
    else:
        x_range = None

    feature_ranges = {
        "AAC": aac_range,
        "BE": be_range,
        "PSSM": pssm_range,
        "CKSAAP": cksaap_range
    }

    if x_range:
        feature_ranges["X_Features"] = x_range

    return base_features, labels[selected_indices], feature_ranges


# 1. 加载特征数据
X, y, _ = load_features(use_x_features=True)

# 2. 划分训练集和测试集（用于调参）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 在训练集上进行LassoCV，搜索最优 alpha
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train, y_train)

# 4. 输出最优alpha
best_alpha = lasso_cv.alpha_
print(f"最优的 alpha 值: {best_alpha:.4f}")

# 5. 定义5折交叉验证器
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6. 用最优alpha在完整数据集上进行五折交叉验证，并计算AUC和AUPR
auc_scores = []
aupr_scores = []

for train_idx, test_idx in stratified_kfold.split(X, y):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    # 标准化（可选，但推荐）
    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_test_fold = scaler.transform(X_test_fold)

    # 使用固定alpha重新训练Lasso
    lasso = LassoCV(alphas=[best_alpha], cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_fold, y_train_fold)

    # sigmoid转换得到概率
    y_pred = lasso.predict(X_test_fold)
    y_prob = 1 / (1 + np.exp(-y_pred))

    auc_scores.append(roc_auc_score(y_test_fold, y_prob))
    aupr_scores.append(average_precision_score(y_test_fold, y_prob))

# 7. 输出结果
print(f"平均 AUC: {np.mean(auc_scores):.4f}")
print(f"平均 AUPR: {np.mean(aupr_scores):.4f}")
# # 9. 可视化预测概率分布（可选）
# lasso_cv.fit(X, y)  # 使用整个数据集拟合模型
# y_pred_continuous = lasso_cv.predict(X)
# y_pred_prob = 1 / (1 + np.exp(-y_pred_continuous))  # Sigmoid函数转换为概率
#
# plt.hist(y_pred_prob[y == 0], bins=20, alpha=0.5, label="Class 0")
# plt.hist(y_pred_prob[y == 1], bins=20, alpha=0.5, label="Class 1")
# plt.axvline(0.5, color='red', linestyle='--', label="Decision Boundary")
# plt.legend()
# plt.title("Predicted Probabilities by Best LASSO Model")
# plt.xlabel("Predicted Probability")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()
