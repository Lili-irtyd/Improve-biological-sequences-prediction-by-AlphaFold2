import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

def load_features(feature_mode='base'):
    """
    加载特征数据，并确保类别平衡。
    feature_mode: 
        - 'base': 仅使用基础序列特征 (AAC, BE, CKSAAP, PSSM)
        - 'all': 使用 基础特征 + AF2特征
        - 'af2': 仅使用 AF2特征
    """
    try:
        # 1. 首先加载标签以确定采样索引 (这部分是所有模式通用的)
        labels = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy")
        
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        # np.random.seed(42)
        # # 随机选择负样本进行欠采样
        # selected_neg_indices = np.random.choice(neg_indices, size=500, replace=False)
        # selected_indices = np.concatenate([pos_indices, selected_neg_indices])
        
        # final_labels = labels[selected_indices]

        print(f"正样本数: {len(pos_indices)}, 负样本数: {len(neg_indices)}")
        print("正在使用完整数据集 (不进行欠采样)...")
        selected_indices = np.concatenate([pos_indices, neg_indices])
        # --- 修改结束 ---
        
        final_labels = labels[selected_indices]
        
        feature_arrays = [] 
        feature_sources = []

        # 2. 根据模式加载基础特征 (Base Features)
        if feature_mode in ['base', 'all']:
            print("正在加载基础序列特征 (AAC, BE, CKSAAP, PSSM)...")
            aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
            be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
            cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
            pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")

            # 记录来源名称
            feature_sources += ['AAC'] * aac_features.shape[1]
            feature_sources += ['BE'] * be_features.shape[1]
            feature_sources += ['CKSAAP'] * cksaap_features.shape[1]
            feature_sources += ['PSSM'] * pssm_features.shape[1]

            # 筛选样本并合并基础特征
            base_features = np.hstack([
                aac_features[selected_indices], 
                be_features[selected_indices],
                cksaap_features[selected_indices], 
                pssm_features[selected_indices]
            ])
            feature_arrays.append(base_features)

        # 3. 根据模式加载 AF2 特征 (X Features)
        if feature_mode in ['af2', 'all']:
            print("正在加载 AlphaFold2 特征...")
            x_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_windows.npy")
            # 展平 AF2 特征
            x_features = x_features.reshape(x_features.shape[0], -1)
            # 记录来源名称
            feature_sources += ['AF2Feature'] * x_features.shape[1]
            
            # 筛选样本并添加到列表
            feature_arrays.append(x_features[selected_indices])

        # 4. 合并所有特征
        if not feature_arrays:
            raise ValueError("未选择任何特征模式！")
            
        final_features = np.hstack(feature_arrays)
        
        return final_features, final_labels, feature_sources

    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def tune_hyperparameters(x_train, y_train):
    """使用 GridSearchCV 进行超参数调优"""
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    # class_weight='balanced' 自动处理类别不平衡
    model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model(model, X, y, feature_sources, cv=5):
    """使用交叉验证计算 AUC、AUPR、F1-score、Accuracy，并输出特征选择情况"""
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    test_auc_scores, test_aupr_scores = [], []
    test_f1_scores, test_acc_scores = [], []

    plt.figure(figsize=(8, 6))

    for i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)

        # 计算 AUC
        test_fpr, test_tpr, _ = roc_curve(y_test, y_test_prob)
        test_auc_scores.append(auc(test_fpr, test_tpr))

        # 计算 AUPR
        precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
        test_aupr_scores.append(average_precision_score(y_test, y_test_prob))

        # F1-score & Accuracy
        test_f1_scores.append(f1_score(y_test, y_test_pred))
        test_acc_scores.append(accuracy_score(y_test, y_test_pred))

        plt.plot(test_fpr, test_tpr, lw=1, alpha=0.6, label=f"Fold {i + 1} AUC = {test_auc_scores[-1]:.3f}")

    # 计算均值
    mean_test_auc, mean_test_aupr = np.mean(test_auc_scores), np.mean(test_aupr_scores)
    mean_test_f1, mean_test_acc = np.mean(test_f1_scores), np.mean(test_acc_scores)

    print("\n==== 5折交叉验证结果 ====")
    print(f"{'Metric':<15}{'Test Mean':<15}")
    print("=" * 30)
    print(f"{'AUC':<15}{mean_test_auc:.4f}")
    print(f"{'AUPR':<15}{mean_test_aupr:.4f}")
    print(f"{'F1-score':<15}{mean_test_f1:.4f}")
    print(f"{'Accuracy':<15}{mean_test_acc:.4f}")
    print("=" * 30)

    # 输出被选择的特征
    if hasattr(model, 'feature_importances_'):
        selected_features = np.where(model.feature_importances_ > 0)[0]
        selected_feature_names = [feature_sources[i] for i in selected_features]
        unique_selected_sources = set(selected_feature_names)
        print(f"决策树选择了 {len(selected_features)} 个特征，来源如下:")
        for source in unique_selected_sources:
            count = selected_feature_names.count(source)
            print(f"- {source}: {count} 个")

    # 绘制平均 ROC 曲线
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - 5-Fold Cross Validation")
    plt.legend(loc="lower right")
    plt.show()


def plot_decision_tree(model, feature_names):
    """绘制决策树，并生成 PDF 文件"""
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=['Non-Cleavage', 'Cleavage'], max_depth=3) # 限制显示深度防止太乱
    plt.title("Decision Tree Visualization (Top 3 Levels)")
    plt.show()

    # 生成更清晰的 Graphviz 图像
    try:
        dot_data = export_graphviz(model, out_file=None, filled=True, feature_names=feature_names,
                                   class_names=['Non-Cleavage', 'Cleavage'])
        tree_graph = graphviz.Source(dot_data)
        tree_graph.render("decision_tree")  # 生成 PDF 文件
        print("决策树 PDF 已生成: decision_tree.pdf")
    except Exception as e:
        print(f"Graphviz 生成失败 (可能是未安装 graphviz 软件): {e}")

if __name__ == "__main__":
    print("请选择特征组合模式:")
    print("1: 仅基础特征 (AAC+BE+CKSAAP+PSSM)")
    print("2: 基础特征 + AF2特征")
    print("3: 仅 AF2特征")
    
    mode_map = {
        '1': 'base',
        '2': 'all',
        '3': 'af2'
    }
    
    feature_mode = mode_map.get('3')
    
    if not feature_mode:
        print("无效输入，默认使用基础特征模式 (base)")
        feature_mode = 'base'
    
    print(f"当前选定模式: {feature_mode}")

    features, labels, feature_sources = load_features(feature_mode)
    
    if features is None:
        print("程序终止：特征加载失败")
        exit()

    print(f"特征矩阵形状: {features.shape}")
    print(f"标签形状: {labels.shape}")

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42,
                                                        stratify=labels)
    
    print("正在进行超参数调优...")
    best_model = tune_hyperparameters(x_train, y_train)
    
    print("正在评估模型...")
    evaluate_model(best_model, features, labels, feature_sources)
    
    print("正在绘制决策树...")
    plot_decision_tree(best_model, feature_sources)
    
    print("训练集实际样本数:", len(x_train))
    print("决策树根节点样本数:", best_model.tree_.n_node_samples[0])