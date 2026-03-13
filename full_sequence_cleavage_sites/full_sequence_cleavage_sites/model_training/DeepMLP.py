import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- 1. 配置与数据加载 ---

# [请修改] 这里填写你 AF2 特征的路径
af2_feature_path = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_windows.npy" 
label_path = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy"

# 设置设备 (GPU 优先)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")
X_raw = np.load(af2_feature_path, allow_pickle=True)
y = np.load(label_path)
print(f"Raw loaded X shape: {X_raw.shape}")

# === 核心修复逻辑 (保持不变，这是纯 Numpy 操作) ===
if X_raw.ndim == 1 and len(X_raw.shape) == 1:
    print("Detected 1D object array containing features. Stacking into 2D matrix...")
    try:
        X = np.vstack(X_raw).astype(np.float32)
    except Exception as e:
        print(f"Stacking failed: {e}")
        X = np.array([x for x in X_raw]).astype(np.float32)
elif X_raw.ndim == 1 and isinstance(X_raw[0], (int, float, np.number)):
    print("Detected single scalar feature. Reshaping to (N, 1)...")
    X = X_raw.reshape(-1, 1).astype(np.float32)
else:
    X = X_raw.astype(np.float32)

if X.ndim > 2:
    print(f"Flattening 3D features {X.shape} to 2D...")
    X = X.reshape(X.shape[0], -1)

print(f"Final processed X shape: {X.shape}") 
print(f"Final y shape: {y.shape}")

if X.ndim != 2:
    raise ValueError(f"Data processing failed! Expected 2D array, got {X.ndim}D.")
# === 结束数据修复 ===


# --- 定义 PyTorch 模型 ---
class AF2DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super(AF2DeepMLP, self).__init__()
        
        # 第一层: Dense(256) -> BN -> ReLU -> Dropout(0.5)
        # 注意: PyTorch中通常先做 Linear 再做 BN/ReLU
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 第二层: Dense(128) -> BN -> ReLU -> Dropout(0.5)
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 第三层: Dense(64) -> ReLU -> Dropout(0.3)
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 输出层: Dense(1) -> Sigmoid
        self.output = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

# --- 2. 交叉验证设置 ---
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
f1_scores = []
aupr_scores = []
roc_fpr = []
roc_tpr = []

# --- 3. 训练循环 ---
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nTraining fold {fold}...")

    # 数据拆分
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 转换为 PyTorch Tensor 并移至 GPU
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device) # (N, 1)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    # 计算类别权重 (模拟 class_weight='balanced')
    # PyTorch 需要我们手动处理 Loss 的权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # 将权重映射到每个样本上
    # weight_tensor[i] = weight for class of y[i]
    train_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # 创建一个与 y_train_t 形状相同的权重向量
    sample_weights = torch.where(y_train_t == 0, train_weights[0], train_weights[1])

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t, sample_weights)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 初始化模型
    input_dim = X_train.shape[1]
    model = AF2DeepMLP(input_dim).to(device)

    # 定义损失函数和优化器
    # 注意: 我们在 DataLoader 里传递了 sample_weights，所以在训练循环中手动应用
    # reduction='none' 让我们能对每个样本乘权重，然后再求平均
    criterion = nn.BCELoss(reduction='none') 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- 训练过程 ---
    epochs = 50
    model.train() # 启用 Dropout 和 Batch Norm 更新
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y, batch_w in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch_x)
            
            # 计算加权损失
            loss = criterion(outputs, batch_y)
            loss = (loss * batch_w).mean() # 应用样本权重并取平均
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

    # --- 6. 评估 ---
    model.eval() # 关闭 Dropout 和 Batch Norm 更新
    with torch.no_grad():
        y_pred_prob_t = model(X_val_t)
        # 转回 CPU numpy 数组用于 sklearn 计算
        y_pred_prob = y_pred_prob_t.cpu().numpy()

    # 计算指标
    auc = roc_auc_score(y_val, y_pred_prob)
    auc_scores.append(auc)

    y_pred = (y_pred_prob > 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    f1_scores.append(f1)

    aupr = average_precision_score(y_val, y_pred_prob)
    aupr_scores.append(aupr)

    # 记录 ROC 数据
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_fpr.append(fpr)
    roc_tpr.append(tpr)

    print(f"Fold {fold} Result -> AUC: {auc:.4f}, F1: {f1:.4f}, AUPR: {aupr:.4f}")

# --- 7. 绘图与总结 ---

plt.figure(figsize=(10, 6))
for i in range(len(roc_fpr)):
    plt.plot(roc_fpr[i], roc_tpr[i], label=f'Fold {i + 1} (AUC = {auc_scores[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - AlphaFold2 Features Only (PyTorch)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

# 打印最终平均结果
print("\n" + "="*30)
print(f'Average AUC: {np.mean(auc_scores):.4f}')
print(f'Average F1 Score: {np.mean(f1_scores):.4f}')
print(f'Average AUPR: {np.mean(aupr_scores):.4f}')
print("="*30)