import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. 定义 Dataset (CKSAAP 改为向量输入)
# ==========================================
class MultiInputDataset(Dataset):
    def __init__(self, mlp_data, cksaap_data, af2_data, labels):
        self.labels = torch.FloatTensor(labels)
        
        # 1. MLP Data (AAC + BE + PSSM)
        self.mlp_data = torch.FloatTensor(mlp_data)
        
        # 2. CKSAAP Data (改为 MLP 处理，不需要 unsqueeze 增加通道维度)
        # 原始形状 (N, 1600) -> 保持 (N, 1600)
        self.cksaap_data = torch.FloatTensor(cksaap_data) 
        
        # 3. AF2 Data (可选, 保持 CNN 处理)
        if af2_data is not None:
            # 原始形状 (N, L, 384) -> PyTorch Conv1d 需要 (N, 384, L)
            self.af2_data = torch.FloatTensor(af2_data)
            self.use_af2 = True
        else:
            self.af2_data = None
            self.use_af2 = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_mlp = self.mlp_data[idx]
        x_cksaap = self.cksaap_data[idx] # (1600,)
        y = self.labels[idx]
        
        if self.use_af2:
            # AF2: (L, C) -> (C, L)
            x_af2 = self.af2_data[idx].permute(1, 0)
            return x_mlp, x_cksaap, x_af2, y
        else:
            # 返回空 tensor 占位
            return x_mlp, x_cksaap, torch.empty(0), y

# ==========================================
# 2. 定义模型 (CKSAAP 分支改为 MLP)
# ==========================================
class HybridModel(nn.Module):
    def __init__(self, mlp_input_dim, use_af2=False, af2_channels=384, cksaap_dim=1600):
        super(HybridModel, self).__init__()
        self.use_af2 = use_af2

        # --- Branch A: MLP (AAC + BE + PSSM) ---
        self.mlp_block = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        fusion_dim = 64

        # --- Branch B: MLP (CKSAAP) [修改点] ---
        # 之前是 CNN (输出 25472 维)，现在改为 MLP 降维
        self.cksaap_block = nn.Sequential(
            nn.Linear(cksaap_dim, 512),   # 第一层压缩
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),              # 防止过拟合
            nn.Linear(512, 128),          # 第二层压缩到 128 维
            nn.ReLU()
        )
        self.cksaap_out_dim = 128
        fusion_dim += self.cksaap_out_dim

        # --- Branch C: CNN (AF2) - Optional ---
        if self.use_af2:
            # Input: (Batch, 384, Length)
            self.af2_block = nn.Sequential(
                nn.Conv1d(af2_channels, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU()
                # Global Max Pooling 在 forward 中执行
            )
            fusion_dim += 128 # AF2 输出维度

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1) # Logits
        )

    def forward(self, x_mlp, x_cksaap, x_af2=None):
        # 1. MLP Branch
        out_mlp = self.mlp_block(x_mlp)
        
        # 2. CKSAAP Branch (Now MLP)
        out_cksaap = self.cksaap_block(x_cksaap)
        
        # 3. AF2 Branch (Optional CNN)
        if self.use_af2 and x_af2 is not None:
            out_af2 = self.af2_block(x_af2)
            out_af2 = torch.max(out_af2, dim=2)[0] # Global Max Pooling
            merged = torch.cat([out_mlp, out_cksaap, out_af2], dim=1)
        else:
            merged = torch.cat([out_mlp, out_cksaap], dim=1)
            
        return self.classifier(merged)

# ==========================================
# 3. 训练与评估函数
# ==========================================
def run_cv_experiment(X_mlp, X_cksaap, X_af2, y, model_name="Model"):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tprs = []
    aucs = []
    auprs = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    precisions = []

    print(f"\n>>> Start Training: {model_name} ...")
    
    use_af2 = (X_af2 is not None)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_mlp, y), 1):
        # 1. 数据拆分
        X_mlp_train, X_mlp_val = X_mlp[train_idx], X_mlp[val_idx]
        X_cksaap_train, X_cksaap_val = X_cksaap[train_idx], X_cksaap[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        X_af2_train, X_af2_val = None, None
        if use_af2:
            X_af2_train, X_af2_val = X_af2[train_idx], X_af2[val_idx]

        # 2. 标准化 (MLP部分)
        scaler = StandardScaler()
        X_mlp_train = scaler.fit_transform(X_mlp_train)
        X_mlp_val = scaler.transform(X_mlp_val)
        
        # 3. 标准化 (AF2部分)
        if use_af2:
            N_train, L, C = X_af2_train.shape
            N_val = X_af2_val.shape[0]
            scaler_af2 = StandardScaler()
            X_af2_train = scaler_af2.fit_transform(X_af2_train.reshape(-1, C)).reshape(N_train, L, C)
            X_af2_val = scaler_af2.transform(X_af2_val.reshape(-1, C)).reshape(N_val, L, C)
            
        # 注意: CKSAAP 通常是频次特征，数值范围较好，若需要也可在这里加 StandardScaler

        # 4. 类权重
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
        
        # 5. Dataset & Loader
        train_ds = MultiInputDataset(X_mlp_train, X_cksaap_train, X_af2_train, y_train)
        val_ds = MultiInputDataset(X_mlp_val, X_cksaap_val, X_af2_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        # 6. Model
        model = HybridModel(
            mlp_input_dim=X_mlp_train.shape[1],
            use_af2=use_af2,
            af2_channels=C if use_af2 else 0,
            cksaap_dim=X_cksaap_train.shape[1] # 自动获取 1600
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 7. Training Loop
        epochs = 30
        for epoch in range(epochs):
            model.train()
            for x_m, x_c, x_a, target in train_loader:
                x_m, x_c, target = x_m.to(device), x_c.to(device), target.to(device)
                x_a = x_a.to(device) if use_af2 else None
                
                optimizer.zero_grad()
                out = model(x_m, x_c, x_a)
                loss = criterion(out, target.view(-1, 1))
                loss.backward()
                optimizer.step()

        # 8. Evaluation
        model.eval()
        probs_list, y_list = [], []
        with torch.no_grad():
            for x_m, x_c, x_a, target in val_loader:
                x_m, x_c = x_m.to(device), x_c.to(device)
                x_a = x_a.to(device) if use_af2 else None
                
                logits = model(x_m, x_c, x_a)
                probs = torch.sigmoid(logits)
                probs_list.append(probs.cpu().numpy())
                y_list.append(target.numpy())
        
        y_prob = np.concatenate(probs_list).ravel()
        y_true = np.concatenate(y_list).ravel()

        # Metrics
        auc = roc_auc_score(y_true, y_prob)
        aupr = average_precision_score(y_true, y_prob)
        aucs.append(auc)
        auprs.append(aupr)
        
        # Interpolation for Plotting
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        # Recall 降序 -> interp 需要升序 X
        interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_prec)
        
        print(f"  Fold {fold} | AUC: {auc:.4f} | AUPR: {aupr:.4f}")

    return {
        "mean_fpr": mean_fpr, "tprs": tprs, "aucs": aucs,
        "mean_recall": mean_recall, "precisions": precisions, "auprs": auprs
    }

# ==========================================
# 4. 数据加载与主程序
# ==========================================
print("Loading Data...")
# 请替换为你的真实路径
base_path = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/" 
try:
    aac = np.load(base_path + "aac.npy")
    be = np.load(base_path + "be.npy")
    pssm = np.load(base_path + "pssm.npy")
    cksaap = np.load(base_path + "cksaap_fixed.npy")
    y = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy")
    
    # AF2 Load & Reshape
    af2_path = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_windows.npy"
    af2_raw = np.load(af2_path)
    
    if len(af2_raw.shape) == 2:
        feat_dim = af2_raw.shape[1]
        seq_len = feat_dim // 384
        X_AF2 = af2_raw.reshape(-1, seq_len, 384)
    else:
        X_AF2 = af2_raw

except FileNotFoundError:
    print("Warning: Data not found. Using Mock Data.")
    N = 500
    aac = np.random.rand(N, 20)
    be = np.random.rand(N, 600)
    pssm = np.random.rand(N, 620)
    cksaap = np.random.rand(N, 1600)
    X_AF2 = np.random.rand(N, 31, 384)
    y = np.random.randint(0, 2, N)

# --- 特征准备 ---
# MLP 输入: AAC + BE + PSSM
X_MLP = np.hstack([aac, be, pssm])
print(f"MLP Input Shape: {X_MLP.shape}")

# CNN/MLP 输入 1: CKSAAP (现在是 2D，给 MLP 用)
X_CKSAAP = cksaap
print(f"CKSAAP Input Shape: {X_CKSAAP.shape}")

# CNN 输入 2: AF2 (仅用于 Extend)
print(f"AF2 Input Shape: {X_AF2.shape}")

# --- 运行对比实验 ---

# 1. Baseline: MLP(Seq) + MLP(CKSAAP)
res_base = run_cv_experiment(X_MLP, X_CKSAAP, None, y, model_name="Baseline (MLP_Seq + MLP_CKSAAP)")

# 2. Extend: MLP(Seq) + MLP(CKSAAP) + CNN(AF2)
res_ext = run_cv_experiment(X_MLP, X_CKSAAP, X_AF2, y, model_name="Extend (+AF2 CNN)")

# ==========================================
# 5. 绘图 (对比 Baseline 和 Extend)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- ROC Plot ---
ax = axes[0]
colors = ['gray', 'red']
results = [res_base, res_ext]
names = ['Baseline', 'Extend']

for i, res in enumerate(results):
    mean_auc = np.mean(res['aucs'])
    std_auc = np.std(res['aucs'])
    mean_tpr = np.mean(res['tprs'], axis=0)
    mean_tpr[-1] = 1.0
    
    ax.plot(res['mean_fpr'], mean_tpr, color=colors[i], lw=2,
            label=r'%s (AUC = %0.3f $\pm$ %0.2f)' % (names[i], mean_auc, std_auc))
    
    std_tpr = np.std(res['tprs'], axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(res['mean_fpr'], tprs_lower, tprs_upper, color=colors[i], alpha=.1)

ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_title('ROC Curve')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")

# --- PR Plot ---
ax = axes[1]
for i, res in enumerate(results):
    mean_aupr = np.mean(res['auprs'])
    std_aupr = np.std(res['auprs'])
    mean_prec = np.mean(res['precisions'], axis=0)
    
    ax.plot(res['mean_recall'], mean_prec, color=colors[i], lw=2,
            label=r'%s (AUPR = %0.3f $\pm$ %0.2f)' % (names[i], mean_aupr, std_aupr))
            
    std_prec = np.std(res['precisions'], axis=0)
    prec_upper = np.minimum(mean_prec + std_prec, 1)
    prec_lower = np.maximum(mean_prec - std_prec, 0)
    ax.fill_between(res['mean_recall'], prec_lower, prec_upper, color=colors[i], alpha=.1)

ax.set_title('Precision-Recall Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("Final_Comparison_CKSAAP_MLP.png", dpi=300)
plt.show()

print("Experiment Finished.")