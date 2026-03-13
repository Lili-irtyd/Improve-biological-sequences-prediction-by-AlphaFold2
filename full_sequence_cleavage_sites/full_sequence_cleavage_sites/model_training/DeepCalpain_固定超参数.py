import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 加载特征数据
def load_features():
    aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
    be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
    cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
    pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
    labels = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")
    return aac_features, be_features, pssm_features, cksaap_features, labels


# 构建子模块
def build_submodule(input_dim, params):
    inputs = Input(shape=(input_dim,))
    x = Dense(params["hidden_neuro_number"],
              activation=params["hidden_activation_function"],
              kernel_initializer=params["hidden_kernel_initializer"],
              bias_initializer=params["hidden_bias_initializer"])(inputs)
    x = Dropout(params["hidden_dropout_rate"])(x)
    return inputs, x


# 构建 DeepCalpain 模型
def build_model(aac_dim, be_dim, pssm_dim, cksaap_dim):
    # 子模型 1: AAC
    aac_params = {
        "hidden_neuro_number": 500,
        "hidden_activation_function": "linear",
        "hidden_kernel_initializer": "he_uniform",
        "hidden_bias_initializer": "zero",
        "hidden_dropout_rate": 0.788417751,
    }
    aac_input, aac_output = build_submodule(aac_dim, aac_params)

    # 子模型 2: PSSM
    pssm_params = {
        "hidden_neuro_number": 126,
        "hidden_activation_function": "linear",
        "hidden_kernel_initializer": "glorot_normal",
        "hidden_bias_initializer": "uniform",
        "hidden_dropout_rate": 0,
    }
    pssm_input, pssm_output = build_submodule(pssm_dim, pssm_params)

    # 子模型 3: CKSAAP
    cksaap_params = {
        "hidden_neuro_number": 488,
        "hidden_activation_function": "selu",
        "hidden_kernel_initializer": "zero",
        "hidden_bias_initializer": "uniform",
        "hidden_dropout_rate": 0.5,
    }
    cksaap_input, cksaap_output = build_submodule(cksaap_dim, cksaap_params)

    # 子模型 4: BE
    be_params = {
        "hidden_neuro_number": 500,
        "hidden_activation_function": "softsign",
        "hidden_kernel_initializer": "zero",
        "hidden_bias_initializer": "uniform",
        "hidden_dropout_rate": 0.124093898,
    }
    be_input, be_output = build_submodule(be_dim, be_params)

    # 合并子模型
    merged = concatenate([aac_output, pssm_output, cksaap_output, be_output])

    # 主模型
    x = Dense(20, activation="softsign",
              kernel_initializer="uniform",
              bias_initializer="he_normal")(merged)
    x = Dropout(0.433042978)(x)
    outputs = Dense(1, activation="softmax",
                    kernel_initializer="he_normal",
                    bias_initializer="normal")(x)

    # 创建模型
    model = Model(inputs=[aac_input, pssm_input, cksaap_input, be_input], outputs=outputs)
    return model


# 绘制 AUC 曲线
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
    plt.savefig("/home/gpux1/CCPR/full_sequence_cleavage_sites/Picture/AUC_DeepCalpain.svg")


# 主函数
def main():
    # 加载数据
    aac_features, be_features, pssm_features, cksaap_features, labels = load_features()

    # 数据划分
    x_train = [aac_features[:800], pssm_features[:800], cksaap_features[:800], be_features[:800]]
    y_train = labels[:800]
    x_val = [aac_features[800:], pssm_features[800:], cksaap_features[800:], be_features[800:]]
    y_val = labels[800:]

    # 获取特征维度
    aac_dim, pssm_dim, cksaap_dim, be_dim = (
        aac_features.shape[1],
        pssm_features.shape[1],
        cksaap_features.shape[1],
        be_features.shape[1]
    )

    # 构建模型
    model = build_model(aac_dim, be_dim, pssm_dim, cksaap_dim)
    model.compile(optimizer="Adadelta",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=155,
        epochs=50,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
    )

    # 预测并绘制 AUC 曲线
    y_pred = model.predict(x_val).ravel()
    plot_auc(y_val, y_pred, title="ROC Curve for DeepCalpain")

    # 保存模型
    model.save("DeepCalpain_model.h5")
    print("Model saved as DeepCalpain_model.h5.")


if __name__ == "__main__":
    main()
