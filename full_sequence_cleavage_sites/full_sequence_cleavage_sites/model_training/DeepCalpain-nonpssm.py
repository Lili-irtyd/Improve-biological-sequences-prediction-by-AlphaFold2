import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, concatenate
from pyswarm import pso
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#pip install tensorflow pyswarm numpy pandas scikit-learn matplotlib

# 加载特征数据
def load_features():
    """
    加载预处理的四种特征。
    假设特征已经保存为 .npy 文件，每个特征一个文件。
    """
    aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")  # Shape: (26844, 20)
    be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")    # Shape: (26844,600)
    cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")  # Shape: (26844, 1600)
    labels = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")  # Binary labels, Shape: (num_samples,)
    return aac_features, be_features, cksaap_features, labels

# 构建子模块
def build_submodule(input_dim, dropout_rate=0.5):
    """
    构建子模块，用于处理每种特征输入。
    """
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    return inputs, x

# 构建 DeepCalpain 模型
def build_model(aac_dim, be_dim, cksaap_dim, dropout_rate=0.5):
    """
    构建 DeepCalpain 模型。
    """
    # 各特征子模块
    aac_input, aac_output = build_submodule(aac_dim, dropout_rate)
    be_input, be_output = build_submodule(be_dim, dropout_rate)
    cksaap_input, cksaap_output = build_submodule(cksaap_dim, dropout_rate)

    # 合并子模块输出
    merged = concatenate([aac_output, be_output,  cksaap_output])
    x = Dense(256, activation='relu')(merged)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # 二分类输出

    # 创建模型
    model = Model(inputs=[aac_input, be_input,  cksaap_input], outputs=outputs)
    return model

# 粒子群优化 (PSO) 优化超参数
def optimize_hyperparameters(aac_dim, be_dim, cksaap_dim, x_train, y_train, x_val, y_val):
    """
    使用粒子群优化 (PSO) 优化超参数。
    """
    def objective_function(params):
        dropout_rate, learning_rate = params

        # 构建模型
        model = build_model(aac_dim, be_dim,  cksaap_dim, dropout_rate)
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

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# 主函数
def main():
    # 加载数据
    aac_features, be_features,  cksaap_features, labels = load_features()

    # 数据划分
    from sklearn.model_selection import train_test_split
    x_train = [aac_features[:800], be_features[:800],  cksaap_features[:800]]
    y_train = labels[:800]
    x_val = [aac_features[800:], be_features[800:], cksaap_features[800:]]
    y_val = labels[800:]

    # 获取特征维度
    aac_dim, be_dim,  cksaap_dim = aac_features.shape[1], be_features.shape[1],  cksaap_features.shape[1]

    # 优化超参数
    best_dropout_rate, best_learning_rate = optimize_hyperparameters(
        aac_dim, be_dim,  cksaap_dim, x_train, y_train, x_val, y_val
    )
    print(f"Best Dropout Rate: {best_dropout_rate}, Best Learning Rate: {best_learning_rate}")

    # 构建最终模型
    model = build_model(aac_dim, be_dim,  cksaap_dim, best_dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 训练最终模型
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # 预测并绘制 AUC 曲线
    y_pred = model.predict(x_val)
    plot_auc(y_val, y_pred, title="ROC Curve for DeepCalpain")

    # 保存模型
    model.save("DeepCalpain_model.h5")
    print("Model saved as DeepCalpain_model.h5.")

if __name__ == "__main__":
    main()
