import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
import os

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

    # 计算各个特征集的列范围
    aac_range = (0, aac_features.shape[1])
    be_range = (aac_range[1], aac_range[1] + be_features.shape[1])
    pssm_range = (be_range[1], be_range[1] + pssm_features.shape[1])
    cksaap_range = (pssm_range[1], pssm_range[1] + cksaap_features.shape[1])

    base_features = np.hstack([aac_features[selected_indices], be_features[selected_indices],
                               pssm_features[selected_indices], cksaap_features[selected_indices]])
    # base_features = np.hstack([aac_features, be_features,
    #                            pssm_features, cksaap_features])

    if use_x_features and x_features is not None:
        x_range = (cksaap_range[1], cksaap_range[1] + x_features.shape[1])
        base_features = np.hstack([base_features, x_features[selected_indices]])
        #base_features = np.hstack([base_features, x_features])
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
    #return base_features, labels, feature_ranges

def tune_lasso(x_train, y_train):
    """调优 LASSO 的 alpha"""
    alphas = np.logspace(-4, 1, 50)  # 从 10^-4 到 10^1 取 50 个 alpha 值
    best_alpha = None
    best_score = -np.inf

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(x_train, y_train)
        score = lasso.score(x_train, y_train)

        if score > best_score:
            best_score = score
            best_alpha = alpha

    print(f"Best LASSO Alpha: {best_alpha}")
    return Lasso(alpha=best_alpha, max_iter=10000).fit(x_train, y_train)


def select_features(lasso_model, x_train, feature_ranges):
    """使用 LASSO 筛选特征，并统计每个特征集的贡献"""
    coef = lasso_model.coef_
    selected_features = np.where(np.abs(coef) > 1e-4)[0]  # 选出非零权重的特征
    print(f"选出 {len(selected_features)} 个非零特征")

    # 统计特征集贡献
    feature_counts = {key: 0 for key in feature_ranges.keys()}
    for feature_idx in selected_features:
        for feature_name, (start, end) in feature_ranges.items():
            if start <= feature_idx < end:
                feature_counts[feature_name] += 1
                break

    print("\n特征集贡献统计:")
    for feature_name, count in feature_counts.items():
        print(f"{feature_name}: {count} 个特征")

    # 返回筛选后的特征及对应的权重
    return x_train[:, selected_features], selected_features, coef[selected_features]

    return x_train[:, selected_features], selected_features  # 仅返回筛选后的特征
def train_logistic(x_train, y_train,lasso_weights):
    """训练 Logistic 回归，并用 LASSO 的权重初始化"""
    logistic = LogisticRegression(random_state=42, max_iter=10000,class_weight="balanced")

    # 用 LASSO 筛选出的特征权重初始化 Logistic 回归权重
    logistic.fit(x_train, y_train)
    logistic.coef_ = lasso_weights.reshape(1, -1)

    return logistic


def evaluate_model(model, X, y, cv=5):
    """交叉验证评估模型，并绘制 ROC 和 PR 曲线"""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    auc_scores, f1_scores, acc_scores, aupr_scores = [], [], [], []

    # 确保保存目录存在
    os.makedirs("../Picture", exist_ok=True)

    # 创建 ROC 图
    plt.figure(figsize=(6, 5))
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # 计算 AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_scores.append(auc(fpr, tpr))
        plt.plot(fpr, tpr, alpha=0.6, label=f"Fold AUC = {auc_scores[-1]:.3f}")

    plt.legend()
    plt.savefig("../Picture/Lasso-5-rf_AUC.svg", dpi=300)
    plt.close()

    # 创建 Precision-Recall 曲线图
    plt.figure(figsize=(6, 5))
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 训练模型
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # 计算 AUPR
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        aupr_scores.append(average_precision_score(y_test, y_pred_prob))
        plt.plot(recall, precision, alpha=0.6, label=f"Fold AUPR = {aupr_scores[-1]:.3f}")

        # 计算 F1 和 Accuracy
        f1_scores.append(f1_score(y_test, model.predict(X_test)))
        acc_scores.append(accuracy_score(y_test, model.predict(X_test)))

    plt.legend()
    plt.savefig("../Picture/Lasso-5-rf_AUPR.svg", dpi=300)
    plt.close()

    # 打印平均指标
    print(f"\n平均 AUC: {np.mean(auc_scores):.4f}, 平均 AUPR: {np.mean(aupr_scores):.4f}, 平均 F1: {np.mean(f1_scores):.4f}, 平均 Accuracy: {np.mean(acc_scores):.4f}")

if __name__ == "__main__":
    use_x_features = input("是否使用额外特征 x_features? (yes/no): ").strip().lower() == "yes"
    features, labels, feature_ranges = load_features(use_x_features)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # LASSO 选择特征
    best_lasso = tune_lasso(x_train, y_train)
    x_train_selected, selected_indices, lasso_weights = select_features(best_lasso, x_train, feature_ranges)
    x_test_selected = x_test[:, selected_indices]

    # 用 LASSO 权重初始化 Logistic 回归
    logistic_model = train_logistic(x_train_selected, y_train, lasso_weights)
    #logistic_model = train_logistic(x_train_selected, y_train)

    # 评估
    evaluate_model(logistic_model, x_test_selected, y_test)
