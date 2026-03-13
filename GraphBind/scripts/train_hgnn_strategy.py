import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import torchmetrics

# 导入你的 1D CNN 模型和数据加载器
from model import Simple1DCNN 
from dataset import SequenceDataset, collate_fn_pad

# --- 1. [TODO] 请修改为你电脑上的真实路径！ ---
TRAIN_FEAT_DIR = "/user1/scl1/zhiqian/GraphBind/Datasets/PDNA/feature/AF2/train" # <--- 修改这里
TEST_FEAT_DIR  = "/user1/scl1/zhiqian/GraphBind/Datasets/PDNA/feature/AF2/test"  # <--- 修改这里

# --- .txt 标签文件的路径 ---
# (指向包含 .txt 文件的文件夹)
LABEL_FILE_ROOT_DIR = "/user1/scl1/zhiqian/GraphBind/Datasets/PDNA" # <--- 修改这里

# --- .txt 标签文件的完整文件名 ---
TRAIN_TXT_FILE = os.path.join(LABEL_FILE_ROOT_DIR, "DNA-new_Train.txt") # <--- 修改这里的文件名
TEST_TXT_FILE  = os.path.join(LABEL_FILE_ROOT_DIR, "DNA-129_Test.txt")


# --- 2. 匹配 HGNN 的训练超参数 ---
LEARNING_RATE = 0.00003  # 你可以调整 (HGNN 脚本使用的是 5e-5)
BATCH_SIZE = 64       # 匹配 HGNN 的 batch_size
EPOCHS = 100          # 设置一个较高的上限，让早停来决定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HGNN 策略参数
VALIDATION_SPLIT = 0.2  # 从训练集中划分 20% 作为验证集
SCHEDULER_PATIENCE = 10 # 匹配 HGNN 的 patience=10
EARLY_STOP_PATIENCE = 15 # 如果15轮没提升就停止 (HGNN 设的是 10)
BEST_MODEL_PATH = "1D_CNN_best_model.pth"

# --- 3. 验证函数 (核心) ---
def validate(model, loader, device):
    """
    在验证集上评估，并找到最佳阈值 (模仿 HGNN 的 val 函数)
    """
    model.eval()
    
    all_preds_flat = []
    all_labels_flat = []

    with torch.no_grad():
        for s_reps, labels, mask in loader:
            if s_reps is None: continue
            
            s_reps = s_reps.to(device)
            labels = labels.to(device)
            mask = mask.to(device) 
            
            # (B, 1, L) -> (B, L, 1)
            preds = model(s_reps).permute(0, 2, 1) 
            
            preds_flat = preds[mask]
            labels_flat = labels[mask].int()
            
            if preds_flat.numel() > 0:
                all_preds_flat.append(preds_flat)
                all_labels_flat.append(labels_flat)
    
    if not all_preds_flat:
        print("验证集为空或数据有误！")
        return -1, 0.5 # 返回一个糟糕的分数

    all_preds = torch.cat(all_preds_flat)
    all_labels = torch.cat(all_labels_flat)

    # --- 阈值搜索 (模仿 HGNN) ---
    best_mcc = -1.0
    best_th = 0.5
    best_f1 = 0.0

    # 初始化指标计算器 (使用 torchmetrics)
    mcc_metric = torchmetrics.classification.BinaryMatthewsCorrCoef().to(device)
    f1_metric = torchmetrics.classification.BinaryF1Score().to(device)

    # 从 0.02 遍历到 0.98
    for th in np.arange(0.02, 1.0, 0.02):
        mcc_metric.threshold = th
        f1_metric.threshold = th
        
        mcc = mcc_metric(all_preds, all_labels)
        f1 = f1_metric(all_preds, all_labels)

        if mcc > best_mcc:
            best_mcc = mcc
            best_th = th
            best_f1 = f1
    
    print(f"Validation: Best MCC={best_mcc:.4f} at Th={best_th:.2f}, F1={best_f1:.4f}")
    return best_mcc, best_th

# --- 4. 最终评估函数 ---
def evaluate(model, loader, device, threshold):
    """
    在测试集上使用 *给定* 的阈值进行评估
    """
    model.eval()
    
    # 初始化指标
    metric_auc = torchmetrics.classification.BinaryAUROC().to(device)
    metric_aupr = torchmetrics.classification.BinaryAveragePrecision().to(device)
    metric_f1 = torchmetrics.classification.BinaryF1Score(threshold=threshold).to(device)
    metric_mcc = torchmetrics.classification.BinaryMatthewsCorrCoef(threshold=threshold).to(device)
    metric_spe = torchmetrics.classification.BinarySpecificity(threshold=threshold).to(device)
    metric_rec = torchmetrics.classification.BinaryRecall(threshold=threshold).to(device)
    metric_pre = torchmetrics.classification.BinaryPrecision(threshold=threshold).to(device)

    print(f"Evaluating Test Set using Threshold = {threshold:.2f}...")
    with torch.no_grad():
        for s_reps, labels, mask in loader:
            if s_reps is None: continue
            
            s_reps = s_reps.to(device)
            labels = labels.to(device)
            mask = mask.to(device) 
            
            preds = model(s_reps).permute(0, 2, 1) # (B, L, 1)
            
            preds_flat = preds[mask]
            labels_flat = labels[mask].int()
            
            if preds_flat.numel() == 0: continue
            
            # 更新所有指标
            metric_auc.update(preds_flat, labels_flat)
            metric_aupr.update(preds_flat, labels_flat)
            metric_f1.update(preds_flat, labels_flat)
            metric_mcc.update(preds_flat, labels_flat)
            metric_spe.update(preds_flat, labels_flat)
            metric_rec.update(preds_flat, labels_flat)
            metric_pre.update(preds_flat, labels_flat)
            
    # 计算最终指标
    final_auc = metric_auc.compute()
    final_aupr = metric_aupr.compute()
    final_f1 = metric_f1.compute()
    final_mcc = metric_mcc.compute()
    final_spe = metric_spe.compute()
    final_rec = metric_rec.compute()
    final_pre = metric_pre.compute()
    
    print(f"--- Test Set Results ---")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  AUC: {final_auc:.4f}")
    print(f"  AUPR: {final_aupr:.4f}")
    print(f"  F1:  {final_f1:.4f}")
    print(f"  MCC: {final_mcc:.4f}")
    print(f"  Recall (Sen): {final_rec:.4f}")
    print(f"  Precision:  {final_pre:.4f}")
    print(f"  Specificity: {final_spe:.4f}")
    print("------------------------")
    return final_mcc # 返回 MCC 以便主函数知道

# --- 5. 主训练函数 ---
def main():
    print(f"Using device: {DEVICE}")

    # --- 3. 加载数据 ---
    print("Loading full training data...")
    # 加载 *所有* 训练 .txt 文件
    full_train_dataset = SequenceDataset(TRAIN_FEAT_DIR, TRAIN_TXT_FILE)
    
    # --- 自动分割训练集/验证集 ---
    train_size = int((1.0 - VALIDATION_SPLIT) * len(full_train_dataset))
    valid_size = len(full_train_dataset) - train_size
    train_data, valid_data = random_split(full_train_dataset, [train_size, valid_size])
    
    print(f"Split data: {len(train_data)} train, {len(valid_data)} valid")
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad)

    print("Loading Test data...")
    test_dataset = SequenceDataset(TEST_FEAT_DIR, TEST_TXT_FILE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad)

    # --- 4. 初始化模型、损失和优化器 ---
    model = Simple1DCNN(
        input_channels=384, 
        output_channels=1
    ).to(DEVICE)
    
    # --- 匹配 HGNN 策略 ---
    loss_fn = nn.BCELoss(reduction='none') # 匹配 BCELoss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 匹配 ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # 当指标停止上升时
        factor=0.6,      # 学习率 x 0.6
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-6
    )
    
    # --- 5. 训练循环 (带早停) ---
    best_val_mcc = -1.0
    best_threshold = 0.5
    early_stop_counter = 0

    print("Starting training (HGNN Strategy)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            s_reps, labels, mask = batch
            if s_reps is None: continue
                
            s_reps = s_reps.to(DEVICE)
            labels = labels.to(DEVICE)
            mask = mask.to(DEVICE)
            
            preds = model(s_reps) # (B, 1, L)
            
            # --- 掩码损失计算 (BCELoss) ---
            preds = preds.permute(0, 2, 1) # (B, L, 1)
            raw_loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1)) # (B, L)
            masked_loss = raw_loss * mask
            loss = masked_loss.sum() / (mask.sum() + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- 验证、调度和早停 (HGNN 策略) ---
        current_val_mcc, current_best_th = validate(model, valid_loader, DEVICE)
        
        # 更新学习率
        scheduler.step(current_val_mcc)
        
        if current_val_mcc > best_val_mcc:
            print(f"New best model! MCC improved from {best_val_mcc:.4f} to {current_val_mcc:.4f}.")
            best_val_mcc = current_val_mcc
            best_threshold = current_best_th
            early_stop_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model to {BEST_MODEL_PATH}")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
            
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
        
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f} | Current LR: {optimizer.param_groups[0]['lr']:.8f}")

    # --- 6. 最终评估 ---
    print("Training finished. Loading best model for final evaluation...")
    # 加载在验证集上表现最好的模型
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    
    # 在测试集上使用“最佳阈值”进行评估
    evaluate(model, test_loader, DEVICE, best_threshold)

if __name__ == "__main__":
    main()