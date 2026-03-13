import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import copy
import os

# ==========================================
#      实验配置 (修改这里来改变实验条件)
# ==========================================
# 可选模式: 
# 1. 'ALL'      : 使用所有特征 (Handcrafted + CKSAAP + AF2)
# 2. 'NO_AF2'   : 不使用 AF2 (仅 Handcrafted + CKSAAP)
# 3. 'ONLY_AF2' : 仅使用 AF2
EXPERIMENT_MODE = 'ONLY_AF2'  
# EXPERIMENT_MODE = 'NO_AF2'
# EXPERIMENT_MODE = 'ONLY_AF2'

print(f"Current Experiment Mode: === {EXPERIMENT_MODE} ===")

# ==========================================
#           1. 定义模型组件
# ==========================================

class PositionEmbedding(nn.Module):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = nn.Parameter(torch.randn(maxlen, embed_dim))

    def forward(self, x):
        return x + self.pos_emb

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, maxlen):
        super(TransformerEncoderBlock, self).__init__()
        self.pos_embedding = PositionEmbedding(maxlen, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(ff_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.pos_embedding(x)
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ff_output = self.ffn(out1)
        ff_output = self.dropout2(ff_output)
        out2 = self.layernorm2(out1 + ff_output)
        return torch.mean(out2, dim=1) # Global Average Pooling

class MultiModalModel(nn.Module):
    def __init__(self, input_dim_handcrafted, input_dim_cksaap, seq_len, embed_dim, mode='ALL'):
        super(MultiModalModel, self).__init__()
        self.mode = mode
        
        # --- 根据模式动态构建分支 ---
        
        # 1. Handcrafted & CKSAAP 分支 (仅在 ALL 或 NO_AF2 模式下存在)
        if mode in ['ALL', 'NO_AF2']:
            self.mlp_handcrafted = nn.Sequential(
                nn.Linear(input_dim_handcrafted, 128), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(128, 64), nn.ReLU()
            )
            self.mlp_cksaap = nn.Sequential(
                nn.Linear(input_dim_cksaap, 128), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(128, 64), nn.ReLU()
            )
            dim_hand = 64
            dim_cksaap = 64
        else:
            self.mlp_handcrafted = None
            self.mlp_cksaap = None
            dim_hand = 0
            dim_cksaap = 0
            
        # 2. Transformer 分支 (仅在 ALL 或 ONLY_AF2 模式下存在)
        if mode in ['ALL', 'ONLY_AF2']:
            self.transformer_branch = TransformerEncoderBlock(
                embed_dim=embed_dim, num_heads=4, ff_dim=128, dropout_rate=0.1, maxlen=seq_len
            )
            dim_trans = embed_dim
        else:
            self.transformer_branch = None
            dim_trans = 0
            
        # --- 融合层 ---
        # 动态计算拼接后的维度
        concat_dim = dim_hand + dim_cksaap + dim_trans
        
        self.fusion_head = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_handcrafted, x_cksaap, x_high_dim):
        features = []
        
        # 分支 1 & 2 执行
        if self.mode in ['ALL', 'NO_AF2']:
            out_hand = self.mlp_handcrafted(x_handcrafted)
            out_cksaap = self.mlp_cksaap(x_cksaap)
            features.append(out_hand)
            features.append(out_cksaap)
            
        # 分支 3 执行
        if self.mode in ['ALL', 'ONLY_AF2']:
            out_trans = self.transformer_branch(x_high_dim)
            features.append(out_trans)
            
        # 拼接
        if len(features) > 1:
            combined = torch.cat(features, dim=1)
        else:
            combined = features[0]
            
        output = self.fusion_head(combined)
        return output

# --- 早停类 (保持不变) ---
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience; self.min_delta = min_delta; self.counter = 0; self.best_loss = None; self.early_stop = False; self.best_model_weights = None
    def __call__(self, val_loss, model):
        if self.best_loss is None: self.best_loss = val_loss; self.best_model_weights = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1; 
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_loss = val_loss; self.best_model_weights = copy.deepcopy(model.state_dict()); self.counter = 0

# ==========================================
#           2. 数据加载 (保持不变)
# ==========================================
print("Loading data...")
path_features_45 = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45"
path_full_seq = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites"

try:
    aac_features = np.load(os.path.join(path_features_45, "aac.npy"))
    be_features = np.load(os.path.join(path_features_45, "be.npy"))
    pssm_features = np.load(os.path.join(path_features_45, "pssm.npy"))
    X_AAC_Be_PSSM = np.hstack([aac_features, be_features, pssm_features])
except: X_AAC_Be_PSSM = np.random.rand(100, 50).astype(np.float32) # Dummy

try: X_CKSAAP = np.load(os.path.join(path_features_45, "cksaap.npy"))
except: X_CKSAAP = np.random.rand(100, 400).astype(np.float32) # Dummy

try: 
    X_high_dim_features = np.load(os.path.join(path_full_seq, "combined_windows.npy")).astype(np.float32)
except: X_high_dim_features = np.random.rand(100, 30, 384).astype(np.float32) # Dummy

try: y = np.load(os.path.join(path_full_seq, "combined_labels.npy")).astype(np.float32)
except: y = np.random.randint(0, 2, 100).astype(np.float32) # Dummy

print("Data loaded.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
#           3. 主训练循环
# ==========================================

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, aupr_scores = [], []
BATCH_SIZE = 32; EPOCHS = 100; LR = 1e-3

for fold, (train_idx, val_idx) in enumerate(kf.split(X_AAC_Be_PSSM, y), 1):
    print(f"Training fold {fold} (Mode: {EXPERIMENT_MODE})...")
    
    # 1. 数据拆分 & 标准化
    scaler_hand = StandardScaler()
    X_hand_train = scaler_hand.fit_transform(X_AAC_Be_PSSM[train_idx])
    X_hand_val = scaler_hand.transform(X_AAC_Be_PSSM[val_idx])
    
    scaler_cksaap = StandardScaler()
    X_cksaap_train = scaler_cksaap.fit_transform(X_CKSAAP[train_idx])
    X_cksaap_val = scaler_cksaap.transform(X_CKSAAP[val_idx])
    
    X_high_train, X_high_val = X_high_dim_features[train_idx], X_high_dim_features[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 2. 转 Tensor
    t_hand_train = torch.FloatTensor(X_hand_train).to(device)
    t_hand_val = torch.FloatTensor(X_hand_val).to(device)
    t_cksaap_train = torch.FloatTensor(X_cksaap_train).to(device)
    t_cksaap_val = torch.FloatTensor(X_cksaap_val).to(device)
    t_high_train = torch.FloatTensor(X_high_train).to(device)
    t_high_val = torch.FloatTensor(X_high_val).to(device)
    t_y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    t_y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    # 3. DataLoader
    train_dataset = TensorDataset(t_hand_train, t_cksaap_train, t_high_train, t_y_train)
    val_dataset = TensorDataset(t_hand_val, t_cksaap_val, t_high_val, t_y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    
    # 4. 初始化模型 (传入实验模式)
    model = MultiModalModel(
        input_dim_handcrafted=X_hand_train.shape[1],
        input_dim_cksaap=X_cksaap_train.shape[1],
        seq_len=X_high_train.shape[1],
        embed_dim=X_high_train.shape[2],
        mode=EXPERIMENT_MODE  # <--- 关键修改
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=10)

    # 5. 训练
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_hand, batch_cksaap, batch_high, batch_y in train_loader:
            optimizer.zero_grad()
            # 无论什么模式，都把所有数据传进去，模型内部会自己决定用哪些
            outputs = model(batch_hand, batch_cksaap, batch_high)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_hand.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_hand, batch_cksaap, batch_high, batch_y in val_loader:
                outputs = model(batch_hand, batch_cksaap, batch_high)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_hand.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # 6. 最终评估
    model.load_state_dict(early_stopping.best_model_weights)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_hand, batch_cksaap, batch_high, batch_y in val_loader:
            outputs = model(batch_hand, batch_cksaap, batch_high)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    y_pred = np.array(all_preds).flatten()
    y_true = np.array(all_targets).flatten()
    
    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    auc_scores.append(auc); aupr_scores.append(aupr)
    print(f"  Fold {fold} AUC: {auc:.4f}, AUPR: {aupr:.4f}")

print(f'\n=== Result for Mode: {EXPERIMENT_MODE} ===')
print(f'Average AUC: {np.mean(auc_scores):.4f}')
print(f'Average AUPR: {np.mean(aupr_scores):.4f}')