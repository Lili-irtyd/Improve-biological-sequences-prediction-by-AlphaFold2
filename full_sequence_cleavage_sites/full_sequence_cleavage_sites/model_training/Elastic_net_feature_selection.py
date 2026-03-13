import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.svm import SVC
import warnings
from sklearn.exceptions import ConvergenceWarning

# 过滤收敛警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_features(mode="all"):
    """
    加载特征数据
    mode: "all" (全部特征), "base" (AAC, BE, PSSM, CKSAAP), "only_x" (仅 x_features)
    """
    try:
        # 预先定义变量
        aac_features = be_features = cksaap_features = pssm_features = x_features = None
        
        # 无论什么模式，标签都是必须的
        labels = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy")

        # 加载基础特征 (如果模式不是 'only_x')
        if mode in ["all", "base"]:
            aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
            be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
            cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
            pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")

        # 加载 X 特征 (如果模式不是 'base')
        if mode in ["all", "only_x"]:
            x_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_windows.npy")
            print(x_features.shape())
            
            # --- 修改部分开始 ---
            # 假设 x_features 原形状是 (Samples, Window_Size, Feature_Dim)
            # 例如 (1000, 31, 128) -> 平均后变为 (1000, 128)
            if x_features.ndim == 3:
                print(f"检测到三维特征，正在对序列长度维度 (axis=1) 求平均...")
                #x_features = x_features.mean(axis=1) 
                x_features = x_features.reshape(x_features.shape[0], -1)
            else:
                # 如果已经是二维或其他形状，执行平整化作为兜底
                x_features = x_features.reshape(x_features.shape[0], -1)
            # --- 修改部分结束 ---

    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None

    selected_indices = np.arange(len(labels))
    feature_ranges = {}
    current_idx = 0

    # 根据模式拼接特征
    feature_list = []
    
    if mode in ["all", "base"]:
        # 添加 AAC
        feature_list.append(aac_features[selected_indices])
        feature_ranges["AAC"] = (current_idx, current_idx + aac_features.shape[1])
        current_idx += aac_features.shape[1]
        
        # 添加 BE
        feature_list.append(be_features[selected_indices])
        feature_ranges["BE"] = (current_idx, current_idx + be_features.shape[1])
        current_idx += be_features.shape[1]
        
        # 添加 PSSM
        feature_list.append(pssm_features[selected_indices])
        feature_ranges["PSSM"] = (current_idx, current_idx + pssm_features.shape[1])
        current_idx += pssm_features.shape[1]
        
        # 添加 CKSAAP
        feature_list.append(cksaap_features[selected_indices])
        feature_ranges["CKSAAP"] = (current_idx, current_idx + cksaap_features.shape[1])
        current_idx += cksaap_features.shape[1]

    if mode in ["all", "only_x"]:
        feature_list.append(x_features[selected_indices])
        feature_ranges["X_Features"] = (current_idx, current_idx + x_features.shape[1])
        current_idx += x_features.shape[1]

    # 最终合并
    base_features = np.hstack(feature_list)

    print(f"模式: {mode}")
    print(f"样本总数: {len(labels)}, 总特征维度: {base_features.shape[1]}")
    return base_features, labels[selected_indices], feature_ranges
def tune_elasticnet(x_train, y_train):
    """调优 ElasticNet 的 alpha 和 l1_ratio"""
    print("开始 ElasticNet 参数调优...")
    alphas = np.logspace(-4, 1, 50)  # 从 10^-4 到 10^1 取 50 个 alpha 值
    l1_ratios = [0.3, 0.5, 0.7, 0.9, 1.0]  # 取 l1_ratio 
    best_alpha = None
    best_l1_ratio = None
    best_score = -np.inf

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            elasticnet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            elasticnet.fit(x_train, y_train)
            score = elasticnet.score(x_train, y_train)

            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_l1_ratio = l1_ratio

    print(f"Best ElasticNet Alpha: {best_alpha}, L1 Ratio: {best_l1_ratio}")
    return ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000).fit(x_train, y_train)

def select_features(elasticnet_model, x_train, feature_ranges):
    """使用 ElasticNet 筛选特征，统计贡献，并返回详细的特征来源列表"""
    coef = elasticnet_model.coef_
    selected_features_indices = np.where(np.abs(coef) > 1e-4)[0]  # 选出非零权重的特征索引
    print(f"选出 {len(selected_features_indices)} 个非零特征")

    # 1. 统计特征集贡献
    # 2. 收集详细信息 (Feature Name, Original Index, Coefficient)
    feature_counts = {key: 0 for key in feature_ranges.keys()}
    feature_details = []

    for idx in selected_features_indices:
        for feature_name, (start, end) in feature_ranges.items():
            if start <= idx < end:
                feature_counts[feature_name] += 1
                # 记录详细信息：特征组名，组内相对索引，全局索引，权重
                feature_details.append({
                    "Group": feature_name,
                    "Global_Index": idx,
                    "Weight": coef[idx]
                })
                break

    print("\n特征集贡献统计:")
    for feature_name, count in feature_counts.items():
        print(f"{feature_name}: {count} 个特征")

    # 返回：
    # 1. 筛选后的训练集矩阵
    # 2. 选中的全局索引
    # 3. 选中的特征对应的权重
    # 4. 详细的特征信息列表（新增返回项）
    return x_train[:, selected_features_indices], selected_features_indices, coef[selected_features_indices], feature_details

def train_logistic(x_train, y_train, elasticnet_weights):
    """训练 Logistic 回归，并用 ElasticNet 的权重初始化"""
    logistic = LogisticRegression(random_state=42, max_iter=10000, class_weight="balanced")

    # 用 ElasticNet 筛选出的特征权重初始化 Logistic 回归权重
    logistic.fit(x_train, y_train)
    # 注意：直接赋值 coef_ 需要维度匹配
    if elasticnet_weights is not None:
        logistic.coef_ = elasticnet_weights.reshape(1, -1)

    return logistic

def train_svm(x_train, y_train, elasticnet_weights):
    """训练 Kernel SVM 并进行超参数调优"""
    # 注意：SVM (SVC) 通常不直接使用 ElasticNet 的权重作为初始化，
    # 因为 SVM 寻找的是最大间隔超平面，优化目标不同。
    # 这里我们保留 elasticnet_weights 参数是为了接口兼容，但在 SVC 网格搜索中不使用它。
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # 初始化 SVM 分类器
    svm = SVC(probability=True, random_state=42, class_weight='balanced')

    # 使用 GridSearchCV 进行超参数调优
    print("开始 SVM 超参数调优...")
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # 打印最佳超参数
    print(f"SVM 最佳超参数: {grid_search.best_params_}")

    # 返回调整后的最佳模型
    return grid_search.best_estimator_

def evaluate_model(model, X, y, cv=5):
    """交叉验证评估模型，并绘制 Mean ROC 和 Mean PR 曲线"""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 确保保存目录存在
    os.makedirs("../Picture", exist_ok=True)

    # --- 准备数据存储 ---
    # ROC 相关
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # PR 相关
    precisions_list = []
    auprs = []
    mean_recall = np.linspace(0, 1, 100)
    
    # 其他指标
    f1_scores = []
    acc_scores = []

    # 设置绘图画布 (1行2列)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 循环每一折 (Fold) ---
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测概率 (用于曲线) 和 类别 (用于 F1/Acc)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # 1. 计算 ROC 数据
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # 插值 TPR 以便计算平均值
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        # 绘制当前折的 ROC
        ax1.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {roc_auc:.3f})')

        # 2. 计算 PR 数据
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        curr_aupr = average_precision_score(y_test, y_pred_prob)
        auprs.append(curr_aupr)
        # 插值 Precision (注意 recall 通常是降序，需要反转用于 interp)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions_list.append(interp_precision)
        # 绘制当前折的 PR
        ax2.plot(recall, precision, lw=1, alpha=0.3, label=f'Fold {i+1} (AUPR = {curr_aupr:.3f})')

        # 3. 统计基础指标
        f1_scores.append(f1_score(y_test, y_pred))
        acc_scores.append(accuracy_score(y_test, y_pred))

    # --- 绘制 Mean ROC ---
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax1.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})', lw=2)
    
    # 填充标准差范围 (可选)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")

    # --- 绘制 Mean PR ---
    mean_precision = np.mean(precisions_list, axis=0)
    mean_aupr_val = np.mean(auprs)
    std_aupr = np.std(auprs)
    ax2.plot(mean_recall, mean_precision, color='b', label=f'Mean PR (AUPR = {mean_aupr_val:.3f} $\pm$ {std_aupr:.3f})', lw=2)
    
    # 填充标准差范围 (可选)
    std_precision = np.std(precisions_list, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    ax2.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    ax2.set_xlim([0.0, 1.05])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("../Picture/Elastic_SVM_Mean_Performance.svg", dpi=300)
    plt.close()

    # 打印最终平均指标
    print(f"\n===== 交叉验证最终结果 (CV={cv}) =====")
    print(f"Mean AUC:       {mean_auc:.4f} (+/- {std_auc:.4f})")
    print(f"Mean AUPR:      {mean_aupr_val:.4f} (+/- {std_aupr:.4f})")
    print(f"Mean F1 Score:  {np.mean(f1_scores):.4f}")
    print(f"Mean Accuracy:  {np.mean(acc_scores):.4f}")

if __name__ == "__main__":
    # 选项: 'all', 'base', 'only_x'
    #if choice not in ['all', 'base', 'only_x']:
    choice = 'only_x' # 默认全部
    
    # 1. 加载数据
    features, labels, feature_ranges = load_features(mode=choice)
    
    if features is not None:
        # 划分训练集和独立测试集
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

        # 2. ElasticNet 特征筛选与调优
        best_elasticnet = tune_elasticnet(x_train, y_train)
        
        # 3. 获取筛选后的特征以及详细信息 (feature_details)
        x_train_selected, selected_indices, elasticnet_weights, feature_details = select_features(best_elasticnet, x_train, feature_ranges)
        
        # --- (可选) 打印前10个最重要的特征信息 ---
        print("\n--- Top 10 Selected Features ---")
        # 按权重绝对值排序
        sorted_details = sorted(feature_details, key=lambda x: abs(x['Weight']), reverse=True)
        for item in sorted_details[:10]:
            print(f"Group: {item['Group']}, Global Idx: {item['Global_Index']}, Weight: {item['Weight']:.5f}")
        # ----------------------------------------

        # 准备完整的筛选后数据集 (用于交叉验证评估)
        features_selected = features[:, selected_indices]

        # 4. 训练最终分类器 (SVM)
        # 注意：这里传入 elasticnet_weights 仅为流程完整，SVM 内部使用 GridSearch 寻找最优参数
        logistic_model = train_svm(x_train_selected, y_train, elasticnet_weights)

        # 5. 评估并绘制 Mean 曲线
        evaluate_model(logistic_model, features_selected, labels)