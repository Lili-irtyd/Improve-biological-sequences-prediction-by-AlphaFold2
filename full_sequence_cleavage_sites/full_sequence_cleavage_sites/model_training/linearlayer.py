import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import tensorflow as tf

# --- 1. 数据加载部分 (修改) ---
def load_evoformer_features():
    """
    加载 Evoformer 的 Single Representation 特征。
    假设你已经提取了特征并保存为 .npy 文件。
    """
    # 路径请修改为你实际保存 Evoformer 特征的路径
    # 假设特征维度是 (样本数, 384) 或者 (样本数, 384 * 窗口大小)
    # 如果你是取截切位点及其周围窗口，需要先展平
    feature_path = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_windows.npy" 
    label_path = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy"
    
    features = np.load(feature_path)
    labels = np.load(label_path)
    
    # 确保特征是 2D 的 (Num_samples, Feature_dim)
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
        
    print(f"Evoformer features loaded. Shape: {features.shape}")
    return features, labels

# --- 2. 构建线性模型 (核心修改) ---
def build_linear_baseline_model(input_dim):
    """
    构建线性探测模型 (Linear Probe)。
    结构：Input -> Dense(1, Sigmoid)
    这是验证特征线性可分性的最严格方式。
    """
    inputs = Input(shape=(input_dim,))
    
    # 论文中提到的 "Linear Head" 就是这层：
    # 直接映射到 1 个输出节点，用 Sigmoid 激活做二分类
    # 没有隐藏层 (Hidden Layers)，没有 ReLU，没有 Dropout (除非为了防过拟合微调)
    outputs = Dense(1, activation='sigmoid')(inputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- 3. 绘制 AUC ---
def plot_auc(y_true, y_pred, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.2f}")
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    # 保存路径请按需修改
    plt.savefig("./AUC_Evoformer_Linear_Baseline.svg") 
    plt.show()

# --- 4. 主函数 ---
def main():
    # A. 加载数据
    # 请确保你已经生成了这个 npy 文件
    # 如果没有，你需要写一个脚本从 AF2 的结果 pkl 中提取 'single' 表示并拼接
    X, y = load_evoformer_features()
    
    # 获取维度
    input_dim = X.shape[1]
    print(f"Input Feature Dimension: {input_dim}")

    # B. 数据划分 (这里改用了 sklearn 的 split，比硬编码切片更安全)
    # random_state 保证可复现
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # C. 构建线性模型
    model = build_linear_baseline_model(input_dim)
    
    # 打印模型结构
    model.summary()

    # D. 编译与训练
    # 线性模型比较简单，通常不需要太复杂的调参
    model.compile(optimizer=Adam(learning_rate=1e-3), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=50,       # 线性层收敛很快，50轮通常足够
        batch_size=32, 
        verbose=1
    )

    # E. 预测与评估
    y_pred = model.predict(X_val)
    plot_auc(y_val, y_pred, title="Evoformer + Linear Probe Baseline")

    # F. 保存
    model.save("Evoformer_Linear_Baseline.h5")
    print("Baseline model saved.")

if __name__ == "__main__":
    main()