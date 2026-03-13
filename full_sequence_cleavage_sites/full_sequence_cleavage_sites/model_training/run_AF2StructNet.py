import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. 核心组件: Focal Loss
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: Logits, targets: 0 或 1
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        f_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(f_loss)
        elif self.reduction == 'sum':
            return torch.sum(f_loss)
        else:
            return f_loss

# ==========================================
# 2. 核心组件: Data Loader (适配新模型)
# ==========================================
class ProteinDataset(Dataset):
    def __init__(self, pssm, af2, labels, mode='af2_only'):
        self.labels = torch.FloatTensor(labels).unsqueeze(1) # (N, 1)
        self.mode = mode
        
        # 处理 PSSM (如果是单纯跑 AF2StructNet，其实用不到 PSSM，但保留逻辑以防万一)
        if pssm is not None:
            # 自动修复维度
            if len(pssm.shape) == 2:
                # 假设展平了，尝试恢复 (N, L, 20)
                # 这里根据特征数判断长度，防止报错
                feat_dim = pssm.shape[1]
                if feat_dim % 20 == 0:
                    seq_len = feat_dim // 20
                    pssm = pssm.reshape(-1, seq_len, 20)
                else:
                    # 如果无法整除，可能是特殊情况，保持原样或报错
                    pass 
            self.pssm = torch.FloatTensor(pssm)
        else:
            self.pssm = None
            
        # 处理 AF2
        if af2 is not None:
            self.af2 = torch.FloatTensor(af2) # 预期形状 (N, L, 384)
        else:
            self.af2 = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # 模式 A: 仅使用 AF2 (对应 AF2StructNet)
        if self.mode == 'af2_only':
            # 返回形状: (L, 384)
            return self.af2[idx], label
        
        # 模式 B: PSSM + AF2 (对应以前的 Dual)
        elif self.mode == 'dual':
            pssm_data = self.pssm[idx].permute(1, 0) # (20, L) for Conv1d
            af2_data = self.af2[idx]
            return (pssm_data, af2_data), label
            
        # 模式 C: 仅 PSSM (Baseline)
        else:
            pssm_data = self.pssm[idx].permute(1, 0)
            return pssm_data, label

# ==========================================
# 3. 核心组件: 模型定义 (AF2StructNet)
# ==========================================
class AF2StructNet(nn.Module):
    def __init__(self, feature_dim=384, hidden_dim=128, dropout=0.3):
        super(AF2StructNet, self).__init__()
        
        # 1. 特征压缩与映射 (降维去噪)
        # 输入: (Batch, Length, 384)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 2. 局部特征提取 (1D ResNet Block)
        # 用卷积捕捉局部结构 motif
        # Conv1d 需要 (Batch, Channel, Length)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # 3. 全局特征注意力 (Self-Attention)
        # 捕捉长距离残基对的影响
        # batch_first=True -> (Batch, Seq_Len, Embedding)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # Output Logits
        )

    def forward(self, x):
        # x shape: (Batch, Length, 384)
        
        # A. 投影 -> (Batch, Length, 128)
        x = self.project(x) 
        
        # B. 卷积分支 (ResNet style)
        # 需要转置为 (Batch, Channel, Length) 给 Conv1d 用
        x_perm = x.permute(0, 2, 1) 
        residual = x_perm
        
        out = F.relu(self.bn1(self.conv1(x_perm)))
        out = self.bn2(self.conv2(out))
        out += residual # 残差连接
        out = F.relu(out)
        
        # 转回 (Batch, Length, Channel) 给 Attention 用
        x_conv = out.permute(0, 2, 1)
        
        # C. 注意力机制
        # Self-Attention input: (Batch, L, Hidden)
        attn_out, _ = self.attention(x_conv, x_conv, x_conv)
        x_final = x_conv + attn_out # 残差连接
        
        # D. 池化与分类
        # Global Max Pooling: 在序列长度维度(dim=1)上取最大值
        # 捕捉序列中最显著的特征信号
        embedding = torch.max(x_final, dim=1)[0] # -> (Batch, Hidden)
        
        logits = self.classifier(embedding)
        return logits

# ==========================================
# 4. 训练与验证辅助函数
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        # Move to device
        # inputs 对于 AF2StructNet 只是一个 Tensor (Batch, L, 384)
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            inputs = [i.to(device) for i in inputs]
        else:
            inputs = inputs.to(device)
            
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
    
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = [i.to(device) for i in inputs]
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            val_loss += loss.item() * labels.size(0)
            
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    mean_loss = val_loss / len(loader.dataset)
    return mean_loss, np.array(all_labels), np.array(all_preds)

# ==========================================
# 5. 数据加载 (重点修改)
# ==========================================
def load_data():
    print(">>> Loading Data...")
    PATH_AF2 = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_windows.npy"
    PATH_PSSM = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy"
    PATH_LABEL = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy"
    
    # Load
    X_pssm = np.load(PATH_PSSM).astype(np.float32) 
    X_af2_raw = np.load(PATH_AF2).astype(np.float32)
    y = np.load(PATH_LABEL).astype(np.float32)
    
    print(f"  [Data Info] Raw AF2 Shape: {X_af2_raw.shape}")
    
    # ---------------------------------------------------------
    # 关键修改: 不要在 AF2 上做 Mean Pooling，因为 AF2StructNet 需要序列信息
    # ---------------------------------------------------------
    if len(X_af2_raw.shape) == 3:
        # X_af2_raw 应该是 (N, L, 384)
        X_af2 = X_af2_raw
        print(f"  [Data Info] Keeping AF2 Sequence Dimension: {X_af2.shape}")
    else:
        # 如果原始数据已经扁平化了，代码可能无法运行卷积
        print("  [Warning] AF2 data seems flattened. AF2StructNet requires (N, L, C).")
        X_af2 = X_af2_raw

    # 标准化 (StandardScaler 需要 2D 输入，所以我们要 reshape 来回)
    # AF2: (N, L, C) -> (N*L, C) -> fit -> (N, L, C)
    N, L, C = X_af2.shape
    X_af2_flat = X_af2.reshape(-1, C)
    
    print("  [Preprocessing] Scaling AF2 features...")
    scaler = StandardScaler()
    X_af2_scaled = scaler.fit_transform(X_af2_flat)
    X_af2 = X_af2_scaled.reshape(N, L, C)
    
    return X_pssm, X_af2, y

# ==========================================
# 6. 主实验流程
# ==========================================
def run_experiment():
    # 1. 加载数据
    X_pssm, X_af2, y = load_data()
    
    # 2. 设置 KFold
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储结果
    # 这里我们只跑这一个新模型，命名为 'AF2_StructNet'
    history_scores = {'auc': [], 'aupr': []}
    
    print(f"\n{'='*40}")
    print(f"Start Training: AF2StructNet (ResNet + Attention)")
    print(f"{'='*40}")
    
    fold_aucs = []
    fold_auprs = []
        
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_af2, y)): # 注意这里 split 用 AF2 就行
        print(f"  Fold {fold+1}/5...")
        
        # Split Data (只取 AF2 和 Label)
        X_af2_train, X_af2_val = X_af2[train_idx], X_af2[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Dataset & DataLoader
        # mode='af2_only' 意味着只返回 AF2 数据
        train_dataset = ProteinDataset(None, X_af2_train, y_train, mode='af2_only')
        val_dataset = ProteinDataset(None, X_af2_val, y_val, mode='af2_only')
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Batch size 可适当调小因为参数多了
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Build Model
        model = AF2StructNet(feature_dim=384, hidden_dim=128, dropout=0.3).to(device)
        
        # Loss & Optimizer
        # 针对不平衡数据，继续使用 Focal Loss
        criterion = FocalLoss(gamma=3.0, alpha=0.25) # 尝试提高 gamma 压制简单样本
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # 使用 AdamW 防止过拟合
        
        # Training Loop
        best_val_loss = float('inf')
        patience = 8
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(50):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, _, _ = validate(model, val_loader, criterion)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        # Predict with Best Model
        model.load_state_dict(best_model_state)
        _, y_true_final, y_pred_final = validate(model, val_loader, criterion)
        
        # Metrics
        fpr, tpr, _ = roc_curve(y_true_final, y_pred_final)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true_final, y_pred_final)
        pr_auc = average_precision_score(y_true_final, y_pred_final)
        
        print(f"    -> Fold {fold+1} Result: AUC = {roc_auc:.4f}, AUPR = {pr_auc:.4f}")
        
        fold_aucs.append(roc_auc)
        fold_auprs.append(pr_auc)
        
    mean_auc = np.mean(fold_aucs)
    mean_aupr = np.mean(fold_auprs)
    
    print(f"\n{'='*40}")
    print(f"FINAL RESULT: Mean AUC = {mean_auc:.4f}, Mean AUPR = {mean_aupr:.4f}")
    print(f"{'='*40}")

if __name__ == "__main__":
    run_experiment()