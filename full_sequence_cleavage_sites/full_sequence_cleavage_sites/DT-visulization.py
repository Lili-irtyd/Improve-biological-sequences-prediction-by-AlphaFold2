import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm

# 定义特征名称
base_feature_names = [f"AAC_{i}" for i in range(20)] + \
                     [f"BE_{i}" for i in range(600)] + \
                     [f"CKSAAP_{i}" for i in range(1600)] + \
                     [f"PSSM_{i}" for i in range(60)]

# 额外特征的名称 (如果使用 x_features)
x_feature_names = [f"XFeature_{i}" for i in range(30 * 384)]  # 30x384 维


# 加载数据
def load_features(use_x_features=False):
    """加载特征数据，并确保类别平衡，可选择是否增加额外特征"""
    try:
        aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
        be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
        cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
        pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
        labels = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")

        if use_x_features:
            x_features = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")
            x_features = x_features.reshape(x_features.shape[0], -1)  # (26844, 30x384)
        else:
            x_features = None
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None

    # 获取正负样本索引
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    # 使正负样本平衡
    np.random.seed(42)
    selected_neg_indices = np.random.choice(neg_indices, size=500, replace=False)
    selected_indices = np.concatenate([pos_indices, selected_neg_indices])

    base_features = np.hstack([aac_features[selected_indices], be_features[selected_indices],
                               pssm_features[selected_indices], cksaap_features[selected_indices]])

    # 选择合适的特征名称
    feature_names = base_feature_names
    if use_x_features and x_features is not None:
        base_features = np.hstack([base_features, x_features[selected_indices]])
        feature_names += x_feature_names

    return base_features, labels[selected_indices], feature_names


from tqdm.auto import tqdm


# 超参数搜索
def tune_decision_tree(x_train, y_train):
    """使用 GridSearchCV 进行超参数搜索"""
    print("🔍 正在搜索最佳超参数...")

    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    tree = DecisionTreeClassifier(random_state=42, class_weight='balanced')

    # 计算所有超参数组合的数量
    total_combinations = (len(param_grid['max_depth']) *
                          len(param_grid['min_samples_split']) *
                          len(param_grid['min_samples_leaf']) *
                          len(param_grid['criterion']))

    with tqdm(total=total_combinations, desc="超参数搜索进度") as pbar:
        def update_progress(*args, **kwargs):
            pbar.update(1)  # 每次更新进度

        grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)

        # 在循环中手动更新进度条
        for _ in grid_search.fit(x_train, y_train).cv_results_['params']:
            update_progress()

    print(f"✅ 超参数搜索完成！最佳参数: {grid_search.best_params_}")
    return grid_search.best_estimator_


# 决策树可视化
def visualize_decision_tree(model, feature_names):
    """绘制决策树"""
    print("📊 正在绘制决策树...")
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, filled=True, fontsize=6, impurity=False)
    plt.title("Optimized Decision Tree Visualization")
    plt.show()
    print("✅ 决策树绘制完成！")


# 主流程
if __name__ == "__main__":
    use_x_features = input("是否使用额外特征 x_features? (yes/no): ").strip().lower() == "yes"

    # 加载数据
    features, labels, feature_names = load_features(use_x_features)
    if features is None:
        exit()

    print("🔄 正在拆分训练集和测试集...")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"✅ 数据集拆分完成！训练集: {x_train.shape}, 测试集: {x_test.shape}")

    # 进行超参数调优
    best_tree = tune_decision_tree(x_train, y_train)

    # 可视化最优决策树
    visualize_decision_tree(best_tree, feature_names)
