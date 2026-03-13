import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_recall_curve,
    average_precision_score, classification_report
)
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt



# 加载特征数据
from sklearn.decomposition import PCA
import numpy as np
from xgboost import XGBClassifier


def load_features_with_pca(use_x_features=False):
    """加载特征数据，使用 PCA 降维，并显示降维后特征的来源"""
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
        return None, None, None, None

    # 选择平衡数据集
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

    base_features = np.hstack([
        aac_features[selected_indices],
        be_features[selected_indices],
        pssm_features[selected_indices],
        cksaap_features[selected_indices]
    ])

    if use_x_features and x_features is not None:
        x_range = (cksaap_range[1], cksaap_range[1] + x_features.shape[1])
        base_features = np.hstack([base_features, x_features[selected_indices]])
    else:
        x_range = None

    # PCA 降维
    pca = PCA(n_components=0.90)  # 解释方差阈值 90%
    reduced_features = pca.fit_transform(base_features)

    # 计算每个特征对主成分的贡献
    loading_matrix = np.abs(pca.components_)  # 取绝对值表示贡献大小
    feature_importance = np.sum(loading_matrix, axis=0)

    # 计算每个特征集的贡献
    def calc_contribution(start, end):
        return np.sum(feature_importance[start:end])

    contributions = {
        "AAC": calc_contribution(*aac_range),
        "BE": calc_contribution(*be_range),
        "PSSM": calc_contribution(*pssm_range),
        "CKSAAP": calc_contribution(*cksaap_range)
    }

    if x_range:
        contributions["X_Features"] = calc_contribution(*x_range)

    # 归一化贡献度
    total_contribution = sum(contributions.values())
    contributions = {k: v / total_contribution for k, v in contributions.items()}

    print(f"降维后保留的特征数: {reduced_features.shape[1]}")
    print("各特征集贡献比例:")
    for key, value in contributions.items():
        print(f"  {key}: {value:.2%}")

    return reduced_features, labels[selected_indices], contributions


def evaluate_model(model, x, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    auc_scores, aupr_scores, f1_scores = [], [], []

    for train_idx, test_idx in skf.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(x_train, y_train)
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)

        auc_scores.append(roc_auc_score(y_test, y_pred_prob))
        aupr_scores.append(average_precision_score(y_test, y_pred_prob))
        f1_scores.append(f1_score(y_test, y_pred))

    print(f"Mean AUC: {np.mean(auc_scores):.4f}")
    print(f"Mean AUPR: {np.mean(aupr_scores):.4f}")
    print(f"Mean F1: {np.mean(f1_scores):.4f}")


def train_random_forest(x_train, y_train):
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

if __name__ == "__main__":
    use_x_features = True# 控制是否使用 x_feature
    X, y, contributions = load_features_with_pca(use_x_features=use_x_features)  # 仅解包三个变量
    best_model = train_random_forest(X, y)
    evaluate_model(best_model, X, y)


