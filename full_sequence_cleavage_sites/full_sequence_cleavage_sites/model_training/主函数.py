import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_recall_curve, average_precision_score, \
    roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

def load_features(feature_mode="base", use_full_data=False):
    """
    加载特征数据，并确保类别平衡。

    参数：
    - feature_mode: 'base'（只用传统特征），'x'（只用X结构特征），'all'（结合两者）
    - use_full_data: 是否使用全部数据（否则默认使用全部正样本 + 500个负样本）

    返回:
    - features: 特征矩阵
    - labels: 标签
    """
    try:
        aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
        be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
        cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap_fixed.npy")
        pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
        labels = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy")

        if feature_mode in ["x", "all"]:
            x_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_windows.npy")
            # X_mean = x_features.mean(axis=1)
            # X_max = x_features.max(axis=1)
            #x_features = np.concatenate([X_mean, X_max], axis=1)  # shape: (n_samples, 768)
            x_features = x_features.mean(axis=1)
            # 输入数据的形状
            input_dim = 384
            encoding_dim = 12  # 最终降维目标

            # 编码器
            input_layer = Input(shape=(input_dim,))
            x = Dense(256, activation='relu')(input_layer)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

            encoded = Dense(encoding_dim, activation='relu')(x)

            # 解码器
            x = Dense(128, activation='relu')(encoded)
            x = BatchNormalization()(x)

            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)

            decoded = Dense(input_dim, activation='sigmoid')(x)

            # 构建模型
            autoencoder = Model(input_layer, decoded)
            encoder = Model(input_layer, encoded)

            # 编译与训练
            autoencoder.compile(optimizer=Adam(1e-3), loss='mse')
            autoencoder.fit(x_features, x_features,
                            epochs=50, batch_size=256,
                            shuffle=True, validation_split=0.1)

            # 获取降维后的结果
            x_features_reduced = encoder.predict(x_features)

        else:
            x_features = None
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None

    # 正负样本索引
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    np.random.seed(42)

    if use_full_data:
        selected_indices = np.concatenate([pos_indices, neg_indices])
    else:
        selected_neg_indices = np.random.choice(neg_indices, size=500, replace=False)
        selected_indices = np.concatenate([pos_indices, selected_neg_indices])

    # 选择特征组合方式
    features = None
    if feature_mode == "base":
        features = np.hstack([
            aac_features[selected_indices],
            be_features[selected_indices],
            pssm_features[selected_indices],
            cksaap_features[selected_indices]
        ])
    elif feature_mode == "x":
        features = x_features_reduced[selected_indices]#拼接平均池化 + 最大池化（Mean + Max Pooling）原因： Mean 保留整体趋势，Max 抓住显著位点特征。
    elif feature_mode == "be":
        features = be_features[selected_indices]
    elif feature_mode == "aac":
        features = aac_features[selected_indices]
    elif feature_mode == "pssm":
        features = pssm_features[selected_indices]
    elif feature_mode == "cksaap":
        features = cksaap_features[selected_indices]
    elif feature_mode == "all":
        features = np.hstack([
            #aac_features[selected_indices],
            #be_features[selected_indices],
            #pssm_features[selected_indices],
            cksaap_features[selected_indices],
            x_features[selected_indices]
        ])
    else:
        raise ValueError("feature_mode 参数必须是 'base'、'x' 或 'all'")

    return features, labels[selected_indices]
def reduce_features(base_features, labels, method='PCA', aac_range=None, be_range=None,
                    pssm_range=None, cksaap_range=None, x_range=None):
    if method == 'PCA':
        pca = PCA(n_components=0.90)  # 保留90%累计方差
        reduced_features = pca.fit_transform(base_features)

        # 计算每个原始特征对主成分的绝对贡献
        loading_matrix = np.abs(pca.components_)
        feature_importance = np.sum(loading_matrix, axis=0)

    elif method == 'TSNE':
        # t-SNE 降维到固定维度（比如 50，用户也可以改成参数）
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(base_features)

        # 不能像PCA一样提取components，这里用均匀贡献作为估计
        feature_importance = np.ones(base_features.shape[1])

    else:
        raise ValueError("method must be 'PCA' or 'TSNE'")

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

    total_contribution = sum(contributions.values())
    contributions = {k: v / total_contribution for k, v in contributions.items()}

    print(f"降维方法: {method}")
    print(f"降维后保留的特征数: {reduced_features.shape[1]}")
    print("各特征集贡献比例:")
    for key, value in contributions.items():
        print(f"  {key}: {value:.2%}")

    return reduced_features, labels, contributions


def tune_hyperparameters(x_train, y_train, model_type="decision_tree"):
    """根据用户选择的模型优化超参数"""
    if model_type == "decision_tree":
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced')

    elif model_type == "logistic_regression":
        param_grid = {
            'logistic__C': [0.001, 0.0001, 0.01, 0.1, 1, 10],
            'logistic__penalty': ['elasticnet'],
            'logistic__solver': ['saga'],
            'logistic__l1_ratio': [0.1, 0.5, 0.9]  # elasticnet 专属
        }

        # 创建逻辑回归的 Pipeline，包含标准化步骤
        model = Pipeline([
            ('scaler', StandardScaler()),  # 归一化数据，防止特征值范围影响权重
            ('logistic', LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced"))
        ])

    elif model_type == "svm_rfe":
        # 定义支持向量机模型
        svm_model = SVC(kernel='linear', random_state=42, class_weight="balanced", probability=True)

        # 定义RFE特征选择器，使用SVM作为基础模型
        n_features_to_select = int(features.shape[1] * 0.01)
        rfe_selector = RFE(estimator=svm_model, n_features_to_select=n_features_to_select, step=1)

        # 创建 SVM + RFE 的 Pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),  # 归一化数据
            ('feature_selection', rfe_selector),  # 使用递归特征消除
            ('svm', svm_model)  # 最终使用支持向量机模型
        ])

        # 设置参数网格（仅调节C参数）
        param_grid = {
            'svm__C': [0.001, 0.01, 0.1, 1, 10],  # 调节正则化强度
        }
    elif model_type == "randomforest":

        # 设置参数网格（调节随机森林的相关超参数）
        param_grid = {
            'n_estimators': [100, 200, 300],  # 调节随机森林中的树的数量
            'max_depth': [1,2,3,5,10, 20, 30],  # 调节树的最大深度
            'min_samples_split': [2, 5, 10],  # 调节每个节点的最小样本分裂数
        }
        # 创建 RandomForest + RFE 的 Pipeline
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")

        # If the model is a RandomForest, extract feature importances
    if model_type == "randomforest":
        best_rf_model = grid_search.best_estimator_
        feature_importances = best_rf_model.feature_importances_

        # Sort the features by importance
        sorted_idx = np.argsort(feature_importances)[::-1]
        top_features = sorted_idx[:10]  # Top 10 features, adjust as needed
        print(f"Top 10 Features from RandomForest:")
        for i in top_features:
            print(f"Feature {i}: Importance {feature_importances[i]:.4f}")
    return grid_search.best_estimator_

def evaluate_model(model, X, y, cv=5):
    """交叉验证评估模型，并绘制 ROC 和 PR 曲线"""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    auc_scores, f1_scores, acc_scores, aupr_scores = [], [], [], []

    # 创建 ROC 图
    plt.figure(figsize=(6, 5))
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\n------ Fold {fold} ------")
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # 计算 AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fold_auc = auc(fpr, tpr)
        auc_scores.append(fold_auc)
        plt.plot(fpr, tpr, alpha=0.6, label=f"Fold AUC = {fold_auc:.3f}")

    plt.legend()
    plt.savefig("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/Picture/Lasso-5-rf_AUC.svg", dpi=300)
    plt.close()

    # Output feature importance for RandomForest
    if isinstance(model, RandomForestClassifier):
        feature_importances = model.feature_importances_
        sorted_idx = np.argsort(feature_importances)[::-1]
        print("\nTop 10 features for RandomForest based on feature importance:")
        for i in sorted_idx[:10]:
            print(f"Feature {i}: Importance {feature_importances[i]:.4f}")

    # Precision-Recall Curve
    plt.figure(figsize=(6, 5))
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        fold_aupr = average_precision_score(y_test, y_pred_prob)
        aupr_scores.append(fold_aupr)
        plt.plot(recall, precision, alpha=0.6, label=f"Fold AUPR = {fold_aupr:.3f}")

        fold_f1 = f1_score(y_test, model.predict(X_test))
        f1_scores.append(fold_f1)

        fold_acc = accuracy_score(y_test, model.predict(X_test))
        acc_scores.append(fold_acc)

    plt.legend()
    plt.savefig("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/Picture/Lasso-5-rf_AUPR.svg", dpi=300)
    plt.close()

    # Print average metrics
    print(
        f"\n平均 AUC: {np.mean(auc_scores):.4f}, 平均 AUPR: {np.mean(aupr_scores):.4f}, 平均标准差：{np.std(aupr_scores):.4f} "
        f"平均 F1: {np.mean(f1_scores):.4f}, 平均 Accuracy: {np.mean(acc_scores):.4f}")

def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, scoring='roc_auc'):
    """绘制模型学习曲线，判断是否过拟合"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/Picture/Learning_curve.svg", dpi=300)

    # 简单判断是否过拟合
    gap = train_scores_mean[-1] - test_scores_mean[-1]
    train_auc = train_scores_mean[-1]
    test_auc = test_scores_mean[-1]
    if gap > 0.1 and train_auc > 0.85:
        print("⚠️ 明显过拟合")
    elif test_auc < 0.6:
        print("⚠️ 泛化能力差，模型效果差")
    elif gap > 0.05:
        print("⚠️ 存在一定程度的过拟合，建议调参")
    else:
        print("✅ 没有明显过拟合，表现稳定")
if __name__ == "__main__":
    # 让用户选择是否使用 x_features 和机器学习方法
    feature_mode = input("是否使用额外特征 x_features? (base/x/all): ").strip().lower()
    use_full_data = input("是否使用全部数据？(yes/no): ").strip().lower() == "yes"
    model_type = input("请选择模型 (decision_tree / lasso/ elastic_net/randomforest/ logistic_regression): ").strip().lower()

    features, labels = load_features(feature_mode,use_full_data)
#     reduced_features, labels, contributions = reduce_features(
#     features,
#     labels,
#     method='PCA',  # 或 'PCA'
#     aac_range=(0, 20),
#     be_range=(20, 620),
#     pssm_range=(620, 680),
#     cksaap_range=(680, 2280),
#     x_range=(2280, 13800)  # 可选
# )

    if features is None:
        exit()

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 检查逻辑回归 L1 正则化后哪些特征权重接近 0
    # if model_type == "logistic_regression":
    #     best_logistic = best_model.named_steps["logistic"]有
    #     feature_importance = np.abs(best_logistic.coef_).flatten()
    #
    #     # 找到权重接近 0 的特征
    #     zero_weight_features = np.where(feature_importance < 1e-4)[0]
    #     zero_weight_features_after_2280 = zero_weight_features[zero_weight_features >= 2280]
    #
    #     print(f"⚠️ 共有 {len(zero_weight_features_after_2280)} 个特征权重接近 0，建议去除这些特征:")
    #     print(zero_weight_features)
    # 构造特征名并转为 DataFrame（关键步骤）
    feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]
    features_df = pd.DataFrame(x_train, columns=feature_names)

    # 拟合模型
    model = Pipeline([
        ('scaler', StandardScaler()),  # 归一化数据，防止特征值范围影响权重
        ('logistic', LogisticRegression(penalty='l1',solver='liblinear',  random_state=42,max_iter=2000,class_weight="balanced"))])

    # # 学习曲线 + 过拟合检测
    # #plot_learning_curve(best_model, X_selected, y_train, title=f"Learning Curve ({model_type})")
    # best_model = tune_hyperparameters(x_train, y_train, model_type)
    # plot_learning_curve(best_model, features, labels, title=f"Learning Curve ({model_type})")
    # # 交叉验证
    # evaluate_model(best_model,features,labels)
    # # 用训练好的 best_model 在测试集上做预测
    # y_test_pred_proba = best_model.predict_proba(x_test)[:, 1]
    # y_test_pred = best_model.predict(x_test)

    #初步验证
    plot_learning_curve(model, features, labels, title=f"Learning Curve ({model_type})")
    # 交叉验证
    evaluate_model(model,features,labels)
    # 用训练好的 best_model 在测试集上做预测
    y_test_pred_proba = model.predict_proba(x_test)[:, 1]
    y_test_pred = model.predict(x_test)

    # 🧪 打印评估指标
    print("🧪【测试集评估】")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"AUC (ROC): {roc_auc_score(y_test, y_test_pred_proba):.4f}")
    print(f"AUPR: {average_precision_score(y_test, y_test_pred_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    # 📈 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_test_pred_proba):.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/Picture/Test_ROC_curve.svg", dpi=300)
    plt.show()

    # 📈 绘制 PR 曲线
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPR = {average_precision_score(y_test, y_test_pred_proba):.3f}",
             color='darkorange')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/Picture/Test_PR_curve.svg", dpi=300)
    plt.show()


