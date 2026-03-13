import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def train_model_with_nn(windows, labels):
    """
    使用神经网络训练模型，并计算 AUC。

    参数:
        windows (numpy.ndarray): 滑动窗口样本数据，形状为 (num_windows, window_size, 384)。
        labels (numpy.ndarray): 标签，0 或 1，形状为 (num_windows,)。

    返回:
        model (Sequential): 训练好的神经网络模型。
        accuracy (float): 测试集上的准确率。
        auc (float): 测试集上的AUC。
    """
    # 将窗口数据展平为 2D 数组（每个窗口一行）
    X = windows.reshape(windows.shape[0], -1)  # 将每个窗口展平为一个一维向量

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # 构建神经网络模型
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))  # 第一层
    model.add(Dense(64, activation='relu'))  # 第二层
    model.add(Dense(1, activation='sigmoid'))  # 输出层（0或1）

    # 编译模型
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)  # 阈值为0.5，将预测转为类别

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred_classes)

    # 计算 AUC
    auc = roc_auc_score(y_test, y_pred)

    return model, accuracy, auc


# 假设 windows 和 labels 已经准备好了
# windows: 滑动窗口数据，形状为 (num_windows, window_height, window_width)，例如 (100, 30, 384)
# labels: 对应的标签，形状为 (num_windows,) ，例如 (100,)

# 示例：windows 和 labels 是预先准备好的数据
file_path='combined_windows.npy'
windows = np.load(file_path)
file_path1='combined_labels.npy'
labels = np.load(file_path1)
# 训练神经网络并计算准确率和 AUC
model, accuracy, auc = train_model_with_nn(windows, labels)

print("模型准确率:", accuracy)
print("模型AUC:", auc)
