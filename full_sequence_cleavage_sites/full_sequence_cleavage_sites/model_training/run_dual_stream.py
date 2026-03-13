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
from scipy import interp

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Define Dataset
# ==========================================
class ProteinDataset(Dataset):
    def __init__(self, pssm, af2, labels, mode='dual'):
        self.pssm = torch.FloatTensor(pssm)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        self.mode = mode
        
        if mode == 'dual':
            self.af2 = torch.FloatTensor(af2)
        else:
            self.af2 = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Permute for Conv1d: (L, C) -> (C, L)
        pssm_data = self.pssm[idx].permute(1, 0)
        label = self.labels[idx]
        
        if self.mode == 'dual':
            af2_data = self.af2[idx] # AF2 is already a vector (from Mean Pooling)
            return (pssm_data, af2_data), label
        else:
            return pssm_data, label

# ==========================================
# 2. Define Focal Loss (Keep as is)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        f_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(f_loss) if self.reduction == 'mean' else f_loss

# ==========================================
# 3. Define Model (Modified: Removed Gated Fusion)
# ==========================================
class DualNet(nn.Module):
    def __init__(self, mode='dual', seq_len=31, pssm_dim=20, af2_dim=384):
        super(DualNet, self).__init__()
        self.mode = mode
        
        # --- Branch A: PSSM (CNN) ---
        if seq_len < 5:
            k1, p1 = 3, 1
            k2, p2 = 1, 0
            self.use_pool = False
        else:
            k1, p1 = 5, 2
            k2, p2 = 3, 1
            self.use_pool = True

        self.pssm_conv = nn.Sequential(
            nn.Conv1d(pssm_dim, 64, kernel_size=k1, padding=p1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(2) if self.use_pool else nn.Identity()
        self.pssm_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=k2, padding=p2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # --- Branch B: AF2 (MLP) ---
        if mode == 'dual':
            self.af2_mlp = nn.Sequential(
                nn.Linear(af2_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            # Fusion dimension: PSSM(128) + AF2(128)
            fusion_dim = 128 + 128 
        else:
            fusion_dim = 128

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if self.mode == 'dual':
            pssm, af2 = x
        else:
            pssm = x

        # 1. PSSM Branch
        out_p = self.pssm_conv(pssm)
        out_p = self.pool(out_p)
        out_p = self.pssm_conv2(out_p)
        out_p = torch.max(out_p, dim=2)[0] # Global Max Pooling -> (B, 128)
        
        if self.mode == 'baseline':
            return self.classifier(out_p)
        
        # 2. AF2 Branch
        out_a = self.af2_mlp(af2) # -> (B, 128)
        
        # 3. Concatenate (Direct Fusion, No Gating)
        merged = torch.cat([out_p, out_a], dim=1) # (B, 256)
        
        return self.classifier(merged)

# ==========================================
# 4. Training & Evaluation Pipeline
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            inputs = [i.to(device) for i in inputs]
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = [i.to(device) for i in inputs]
            else:
                inputs = inputs.to(device)
            
            logits = model(inputs)
            preds = torch.sigmoid(logits)
            probs.extend(preds.cpu().numpy())
            targets.extend(labels.numpy())
    return np.array(targets), np.array(probs)

def run_cv_experiment(X_pssm, X_af2, y, model_name="Model", seq_len=31):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        'tprs': [], 'aucs': [], 'precisions': [], 'auprs': [],
        'mean_fpr': np.linspace(0, 1, 100),
        'mean_recall': np.linspace(0, 1, 100)
    }

    print(f"\n>>> Start Training: {model_name} (SeqLen={seq_len}) ...")
    mode = 'dual' if X_af2 is not None else 'baseline'

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_pssm, y), 1):
        # Split
        X_p_tr, X_p_val = X_pssm[train_idx], X_pssm[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        X_a_tr, X_a_val = None, None
        if mode == 'dual':
            X_a_tr, X_a_val = X_af2[train_idx], X_af2[val_idx]

        # Dataset & Loader
        train_ds = ProteinDataset(X_p_tr, X_a_tr, y_tr, mode=mode)
        val_ds = ProteinDataset(X_p_val, X_a_val, y_val, mode=mode)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
        # Model
        model = DualNet(mode=mode, seq_len=seq_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Loss
        n_pos = np.sum(y_tr == 1)
        n_neg = np.sum(y_tr == 0)
        alpha_val = n_neg / (n_pos + n_neg) 
        criterion = FocalLoss(gamma=2.0, alpha=alpha_val)

        # Training Loop
        for epoch in range(30):
            train_one_epoch(model, train_loader, criterion, optimizer)

        # Eval
        y_true, y_prob = validate(model, val_loader)

        # Metrics for Plotting
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        interp_tpr = np.interp(results['mean_fpr'], fpr, tpr)
        interp_tpr[0] = 0.0
        results['tprs'].append(interp_tpr)
        results['aucs'].append(auc(fpr, tpr))
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        interp_prec = np.interp(results['mean_recall'], recall[::-1], precision[::-1])
        results['precisions'].append(interp_prec)
        results['auprs'].append(average_precision_score(y_true, y_prob))
        
        print(f"  Fold {fold} AUC: {results['aucs'][-1]:.4f} | AUPR: {results['auprs'][-1]:.4f}")

    return results

# ==========================================
# 5. Loading Data
# ==========================================
def load_data():
    print(">>> Loading Data...")
    PATH_AF2 = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_windows.npy"
    PATH_PSSM = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy"
    PATH_LABEL = "/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy"
    
    try:
        X_pssm = np.load(PATH_PSSM).astype(np.float32) 
        X_af2_raw = np.load(PATH_AF2).astype(np.float32)
        y = np.load(PATH_LABEL).astype(np.float32)
        
        # Reshape PSSM logic
        if X_pssm.shape[1] == 60:
            X_pssm = X_pssm.reshape(-1, 3, 20)
            actual_seq_len = 3
        else:
            X_pssm = X_pssm.reshape(-1, 31, 20)
            actual_seq_len = 31

        # AF2 Mean Pooling
        if len(X_af2_raw.shape) == 3:
            X_af2 = np.mean(X_af2_raw, axis=1)
        else:
            X_af2 = X_af2_raw
            
        scaler = StandardScaler()
        X_af2 = scaler.fit_transform(X_af2)
        
        return X_pssm, X_af2, y, actual_seq_len
    except FileNotFoundError:
        print("Warning: Files not found. Using dummy data.")
        return np.random.rand(100, 31, 20).astype(np.float32), np.random.rand(100, 384).astype(np.float32), np.random.randint(0, 2, 100).astype(np.float32), 31

# ==========================================
# 6. Main & Plotting
# ==========================================
def main():
    X_pssm, X_af2, y, seq_len = load_data()
    
    # Run Baseline
    res_base = run_cv_experiment(X_pssm, None, y, model_name="Baseline (PSSM)", seq_len=seq_len)
    
    # Run Extend (Standard Concat)
    res_ext = run_cv_experiment(X_pssm, X_af2, y, model_name="Extend (PSSM+AF2)", seq_len=seq_len)
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Plot
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

    # PR Plot
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
    plt.savefig("Final_Comparison_B.png", dpi=300)
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()