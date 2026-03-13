import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
#shap.initjs()
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
x_feature_names = [f"af2_{i}" for i in range(30 * 384)]  # 30x384 维


# 加载数据
def load_features(use_x_features=False):
    """加载特征数据，并确保类别平衡，可选择是否增加额外特征"""
    try:
        aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
        be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
        cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
        pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
        labels = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy")

        if use_x_features:
            x_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_windows.npy")
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
def compute_shap_visualizations(model, X_train, feature_names):
    """计算 SHAP 特征重要性并保存图像"""
    print("📊 正在计算 SHAP 特征重要性...")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # 确保 SHAP 值是 NumPy 数组
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    shap_values_selected = shap_values[1]  # 选择类别 1

    # **确保 X_train 是 DataFrame**
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    # **方法 1：SHAP 交互 HTML (适用于多个样本)**
    shap_html = shap.force_plot(explainer.expected_value[1], shap_values_selected, X_train_df)
    shap.save_html("../Picture/shap_force_plot.html", shap_html)
    print("✅ SHAP Force Plot (所有样本) 已保存为 HTML!")

    # **方法 2：Matplotlib 仅绘制单个样本**
    plt.figure(figsize=(10, 5))  # 增大图像大小，防止白屏
    shap.force_plot(explainer.expected_value[1], shap_values_selected[0, :], X_train_df.iloc[0, :], matplotlib=True)
    plt.show(block=True)  # 解决白屏问题
    plt.savefig("../Picture/shap_force_plot_sample.svg", format="svg", dpi=300)
    plt.close()
    print("✅ SHAP Force Plot (单个样本) 已保存!")

    # **方法 3：SHAP 总结图 (全局特征重要性)**
    plt.figure(figsize=(12, 8))  # 调整大小，确保特征名完整
    shap.summary_plot(shap_values_selected, X_train_df, feature_names=feature_names, show=False)
    plt.tight_layout()  # 自动调整元素防止重叠
    plt.savefig("../Picture/extend_shap_summary_plot.svg", format="svg", dpi=300)  # 提高清晰度
    plt.close()
    print("✅ SHAP Summary Plot 已保存!")

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

# 训练模型并计算 SHAP
if __name__ == "__main__":
    # 加载数据
    use_x_features = input("是否使用额外特征 x_features? (yes/no): ").strip().lower() == "yes"
    features, labels, feature_names = load_features(use_x_features)
    if features is None:
        exit()

    print("🔄 正在拆分训练集和测试集...")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"✅ 数据集拆分完成！训练集: {x_train.shape}, 测试集: {x_test.shape}")

    # 训练决策树模型
    tree = tune_decision_tree(x_train, y_train)
    tree.fit(x_train, y_train)

    # 计算 SHAP 并保存可视化结果
    compute_shap_visualizations(tree, x_train, feature_names)
