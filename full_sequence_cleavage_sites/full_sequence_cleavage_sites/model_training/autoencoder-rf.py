import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV


# 定义 Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# 训练 Autoencoder
def train_autoencoder(x_features, encoding_dim=230, epochs=50, batch_size=128, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = x_features.shape[1]
    model = Autoencoder(input_dim, encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 转换为 PyTorch Tensor
    x_tensor = torch.tensor(x_features, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(x_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练 Autoencoder
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x_batch = batch[0]
            optimizer.zero_grad()
            encoded, decoded = model(x_batch)
            loss = criterion(decoded, x_batch)  # 计算重构误差
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.6f}")

    # 提取编码后的特征
    model.eval()
    with torch.no_grad():
        X_encoded = model.encoder(x_tensor).cpu().numpy()

    return X_encoded


# 加载特征数据
def load_features():
    aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
    be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
    cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
    pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
    labels = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")

    # 加载 x 特征，并展平
    x_features = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")
    x_features = x_features.reshape(x_features.shape[0], -1)

    # 使用 Autoencoder 进行降维
    X_encoded = train_autoencoder(x_features, encoding_dim=250)

    print("降维后的数据形状:", X_encoded.shape)
    return aac_features, be_features, pssm_features, cksaap_features, X_encoded, labels


# 计算 AUC 并绘制曲线
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
    plt.savefig("/home/gpux1/CCPR/full_sequence_cleavage_sites/Picture/AUC_RF_Autoencoder.svg")


# 计算训练集和测试集的准确率
def calculate_accuracy(model, x_train, x_test, y_train, y_test):
    train_acc = accuracy_score(y_train, model.predict(x_train))
    test_acc = accuracy_score(y_test, model.predict(x_test))
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")


# 进行超参数调优
def tune_hyperparameters(x_train, y_train):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [5, 10, 20]
    }

    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=4, verbose=2)
    grid_search.fit(x_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


# 绘制学习曲线
def plot_learning_curve(model, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")

    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.title("Learning Curve for RandomForest")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("/home/gpux1/CCPR/full_sequence_cleavage_sites/Picture/Learning_Curve_RF_Autoencoder.svg")


# 主函数
def main():
    # 加载数据
    aac_features, be_features, pssm_features, cksaap_features, X_encoded, labels = load_features()

    # 合并特征
    combined_features = np.hstack([aac_features, be_features, pssm_features, cksaap_features, X_encoded])
    print(f"Combined features shape: {combined_features.shape}")

    # 数据划分
    x_train, x_val, y_train, y_val = train_test_split(combined_features, labels, test_size=0.2, stratify=labels,
                                                      random_state=42)

    # 超参数调优
    rf = tune_hyperparameters(x_train, y_train)

    # 训练最佳模型
    rf.fit(x_train, y_train)

    # 计算准确率
    calculate_accuracy(rf, x_train, x_val, y_train, y_val)

    # 绘制 AUC 曲线
    y_pred_prob = rf.predict_proba(x_val)[:, 1]
    plot_auc(y_val, y_pred_prob, title="ROC Curve for Optimized RF")

    # 只用训练数据绘制学习曲线
    plot_learning_curve(rf, x_train, y_train)


if __name__ == "__main__":
    main()
