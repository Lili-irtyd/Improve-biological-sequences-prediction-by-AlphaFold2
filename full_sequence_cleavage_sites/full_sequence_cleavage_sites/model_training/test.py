import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_recall_curve, average_precision_score
import warnings
import os
from sklearn.svm import SVC
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from xgboost import XGBClassifier
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


# def select_features(lasso_model, x_train, feature_ranges):
#     """使用 LASSO 筛选特征，并统计每个特征集的贡献"""
#     coef = lasso_model.coef_
#     selected_features = np.where(np.abs(coef) > 1e-4)[0]  # 选出非零权重的特征
#     print(f"选出 {len(selected_features)} 个非零特征")
#
#     # 统计特征集贡献
#     feature_counts = {key: 0 for key in feature_ranges.keys()}
#     for feature_idx in selected_features:
#         for feature_name, (start, end) in feature_ranges.items():
#             if start <= feature_idx < end:
#                 feature_counts[feature_name] += 1
#                 break
#
#     print("\n特征集贡献统计:")
#     for feature_name, count in feature_counts.items():
#         print(f"{feature_name}: {count} 个特征")
#
#     # 返回筛选后的特征及对应的权重
#     return x_train[:, selected_features], selected_features
def select_features(lasso_model, x_train, feature_ranges):
    """使用 LASSO 筛选特征，并统计每个特征集的贡献，同时输出Top 10重要特征"""
    coef = np.abs(lasso_model.coef_)  # 取绝对值
    selected_features = np.where(coef > 1e-4)[0]  # 选出非零权重的特征
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

    # 选出 Top 10 重要特征
    top10_indices = np.argsort(coef[selected_features])[-10:][::-1]  # 选出前10重要特征（按权重排序）
    top10_features = selected_features[top10_indices]
    top10_importance = coef[top10_features]

    # 识别特征所属的特征集
    top10_info = []
    for idx, feature_idx in enumerate(top10_features):
        for feature_name, (start, end) in feature_ranges.items():
            if start <= feature_idx < end:
                feature_position = feature_idx - start + 1  # 在特征集中的位置
                top10_info.append(f"Rank {idx + 1}: Feature {feature_idx} (Importance {top10_importance[idx]:.4f}) - {feature_name} 的第 {feature_position} 个特征")
                break

    # 保存 Top 10 特征到文件
    with open("top10_features_5.txt", "w") as f:
        f.write("\n".join(top10_info))

    print("\nTop 10 重要特征已保存至 top10_features.txt")
    for line in top10_info:
        print(line)

    return x_train[:, selected_features], selected_features


def train_logistic(x_train, y_train):
    """训练 Logistic 回归并进行超参数调优"""
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # liblinear支持L1正则化
    }
    logistic = LogisticRegression(random_state=42, max_iter=10000, class_weight="balanced")

    grid_search = GridSearchCV(logistic, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    print(f"最佳超参数: {grid_search.best_params_}")
    return grid_search.best_estimator_


def train_random_forest(x_train, y_train):
    """训练 Random Forest 并进行超参数调优"""
    param_grid = {
        'n_estimators': [50, 100, 200],  # 树的数量
        'max_depth': [ 10, 20, 30],  # 树的最大深度
        'min_samples_split': [2, 5, 10],  # 拆分一个节点所需的最小样本数
        'min_samples_leaf': [1, 2, 4],  # 每个叶子节点的最小样本数
        'max_features': ['auto', 'sqrt', 'log2']  # 每棵树随机选择的特征数量
    }

    # 初始化 RandomForestClassifier
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

    # 使用 GridSearchCV 调整超参数
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # 打印最佳超参数
    print(f"最佳超参数: {grid_search.best_params_}")

    # 返回调整后的最佳模型
    return grid_search.best_estimator_

def train_xgboost(x_train, y_train):
    """训练 XGBoost 并进行超参数调优"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'reg_lambda': [1, 5, 10]
    }

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)

    print(f"最佳超参数: {grid_search.best_params_}")

    return grid_search.best_estimator_

def train_dt(x_train, y_train):
    """训练 Random Forest 并进行超参数调优"""
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # 初始化 RandomForestClassifier
    rf = DecisionTreeClassifier(random_state=42, class_weight='balanced')

    # 使用 GridSearchCV 调整超参数
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # 打印最佳超参数
    print(f"最佳超参数: {grid_search.best_params_}")

    # 返回调整后的最佳模型
    return grid_search.best_estimator_


def train_lightgbm(x_train, y_train):
    """交叉验证调优 LightGBM，然后在最佳超参数上加早停"""

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.75, 0.9],  # 限制每棵树使用的特征比例
        'reg_lambda': [1, 5, 10],
        'max_bin': [64]  # 降低过拟合风险
    }

    model = lgb.LGBMClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)

    print(f"最佳超参数: {grid_search.best_params_}")

    # 重新划分训练集和验证集
    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 使用最佳参数重新训练 LightGBM，并加早停
    best_params = grid_search.best_params_
    best_model = lgb.LGBMClassifier(**best_params, random_state=42)
    grid_search.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train)],  # 评估集
        callbacks=[early_stopping(10), log_evaluation(1)]  # 早停 & 日志
    )

    return best_model
def train_svm(x_train, y_train):
    """训练 Kernel SVM 并进行超参数调优"""
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # 初始化 SVM 分类器
    svm = SVC(probability=True, random_state=42, class_weight='balanced')

    # 使用 GridSearchCV 进行超参数调优
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # 打印最佳超参数
    print(f"最佳超参数: {grid_search.best_params_}")

    # 返回调整后的最佳模型
    return grid_search.best_estimator_


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
    # 根据特征数量设置文件名
    feature_count = X.shape[1]
    plt.savefig(f"../Picture/Logistic_AUC_{feature_count}_features.svg", dpi=300)
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
    # 根据特征数量设置文件名
    plt.savefig(f"../Picture/Logistic_AUPR_{feature_count}_features.svg", dpi=300)
    plt.close()

    # 打印平均指标
    print(
        f"\n平均 AUC: {np.mean(auc_scores):.4f}, 平均 AUPR: {np.mean(aupr_scores):.4f}, 平均 F1: {np.mean(f1_scores):.4f}, 平均 Accuracy: {np.mean(acc_scores):.4f}")




# 示例调用（假设你已经训练好 random_forest_model，并有 feature_ranges 数据）
# save_top_features(random_forest_model, feature_ranges)


if __name__ == "__main__":
    use_x_features = input("是否使用额外特征 x_features? (yes/no): ").strip().lower() == "yes"
    features, labels, feature_ranges = load_features(use_x_features)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42,
                                                        stratify=labels)

    # 使用LASSO回归选择特征
    lasso = Lasso(alpha=0.0001, max_iter=10000)  # 固定alpha，不进行超参数调优
    lasso.fit(x_train, y_train)

    # 使用Lasso选择的特征
    x_train_selected, selected_indices = select_features(lasso, x_train, feature_ranges)
    features_selected = features[:, selected_indices]


    # 训练Logistic回归模型并调优超参数
    #logistic_model = train_dt(x_train_selected, y_train)
    logistic_model = train_lightgbm(x_train_selected, y_train)

    # 评估模型
    evaluate_model(logistic_model, features_selected, labels)
