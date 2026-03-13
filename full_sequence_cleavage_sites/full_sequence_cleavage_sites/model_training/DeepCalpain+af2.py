import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from pyswarm import pso
import tensorflow as tf


# 加载特征数据
def load_features():
    """
    加载预处理的四种特征和额外的 x 特征。
    """
    aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")  # Shape: (num_samples, aac_dim)
    be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")  # Shape: (num_samples, be_dim)
    cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")  # Shape: (num_samples, cksaap_dim)
    pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")  # Shape: (num_samples, pssm_dim)
    labels = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")  # Shape: (num_samples,)

    # 加载 x 特征，并展平为 2D 数据
    x_features = np.load(
        "/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")  # Shape: (num_samples, window_height, window_width)
    x_features = x_features.reshape(x_features.shape[0], -1)  # 展平为 (num_samples, x_dim)

    print(f"x_features shape after flattening: {x_features.shape}")
    return aac_features, be_features, pssm_features, cksaap_features, x_features, labels


# 构建子模块
def build_submodule(input_dim, dropout_rate=0.5):
    """
    构建子模块，用于处理每种特征输入。
    """
    input_dim = int(input_dim)  # 确保 input_dim 是整数
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    return inputs, x


# 构建 DeepCalpain 模型
def build_model(aac_dim, be_dim, pssm_dim, cksaap_dim, x_dim, dropout_rate=0.5):
    """
    构建 DeepCalpain 模型，加入 x 特征作为第五个输入。
    """
    # 各特征子模块
    aac_input, aac_output = build_submodule(aac_dim, dropout_rate)
    be_input, be_output = build_submodule(be_dim, dropout_rate)
    pssm_input, pssm_output = build_submodule(pssm_dim, dropout_rate)
    cksaap_input, cksaap_output = build_submodule(cksaap_dim, dropout_rate)
    x_input, x_output = build_submodule(x_dim, dropout_rate)

    # 合并子模块输出
    merged = concatenate([aac_output, be_output, pssm_output, cksaap_output, x_output])
    x = Dense(256, activation='relu')(merged)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # 二分类输出

    # 创建模型
    model = Model(inputs=[aac_input, be_input, pssm_input, cksaap_input, x_input], outputs=outputs)
    return model


# 粒子群优化 (PSO) 优化超参数
def optimize_hyperparameters(aac_dim, be_dim, pssm_dim, cksaap_dim, x_dim, x_train, y_train, x_val, y_val):
    """
    使用粒子群优化 (PSO) 优化超参数。
    """

    def objective_function(params):
        dropout_rate, learning_rate = params

        # 构建模型
        model = build_model(aac_dim, be_dim, pssm_dim, cksaap_dim, x_dim, dropout_rate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # 训练模型
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=10,
            batch_size=32,
            verbose=0
        )
        val_loss = history.history['val_loss'][-1]
        return val_loss

    # 定义超参数范围
    lb = [0.1, 1e-5]  # 最小值: dropout_rate, learning_rate
    ub = [0.5, 1e-2]  # 最大值: dropout_rate, learning_rate

    # 运行 PSO 优化
    best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=10)
    return best_params


# 绘制 AUC 曲线
def plot_auc(y_true, y_pred, title="ROC Curve"):
    """
    绘制 AUC 曲线。

    参数:
        y_true (array): 真实标签。
        y_pred (array): 模型预测概率。
        title (str): 图像标题。
    """
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
    plt.savefig("/home/gpux1/CCPR/full_sequence_cleavage_sites/Picture/AUC_5features.svg")


# 主函数
def main():
    # 加载数据
    aac_features, be_features, pssm_features, cksaap_features, x_features, labels = load_features()

    # 数据划分
    x_train = [aac_features[:800], be_features[:800], pssm_features[:800], cksaap_features[:800], x_features[:800]]
    y_train = labels[:800]
    x_val = [aac_features[800:], be_features[800:], pssm_features[800:], cksaap_features[800:], x_features[800:]]
    y_val = labels[800:]

    # 获取特征维度
    aac_dim, be_dim, pssm_dim, cksaap_dim, x_dim = (
        aac_features.shape[1], be_features.shape[1], pssm_features.shape[1], cksaap_features.shape[1],
        x_features.shape[1]
    )
    print(f"aac_dim: {aac_dim}, be_dim: {be_dim}, pssm_dim: {pssm_dim}, cksaap_dim: {cksaap_dim}, x_dim: {x_dim}")

    # 优化超参数
    best_dropout_rate, best_learning_rate = optimize_hyperparameters(
        aac_dim, be_dim, pssm_dim, cksaap_dim, x_dim, x_train, y_train, x_val, y_val
    )
    print(f"Best Dropout Rate: {best_dropout_rate}, Best Learning Rate: {best_learning_rate}")

    # 构建模型
    model = build_model(aac_dim, be_dim, pssm_dim, cksaap_dim, x_dim, dropout_rate=0.5)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32, verbose=1)

    # 预测并绘制 AUC 曲线
    y_pred = model.predict(x_val)
    plot_auc(y_val, y_pred, title="ROC Curve for DeepCalpain with X Feature")

    # 保存模型
    model.save("DeepCalpain_model_with_x.h5")
    print("Model saved as DeepCalpain_model_with_x.h5.")


if __name__ == "__main__":
    main()
