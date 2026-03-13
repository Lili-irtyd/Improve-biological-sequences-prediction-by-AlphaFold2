import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 设置是否使用筛选后的样本（500 负样本 + 全部正样本）
use_selected_samples = False  # 设置为 True 使用筛选样本，False 使用完整数据
use_x_features = True  # 设置为 True 加入 X 特征

# 加载数据
aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
cksaap_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
labels = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")

if use_x_features:
    x_features = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")
    x_features = x_features.reshape(x_features.shape[0], -1)

if use_selected_samples:
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    np.random.seed(42)
    selected_neg_indices = np.random.choice(neg_indices, size=500, replace=False)
    selected_indices = np.concatenate([pos_indices, selected_neg_indices])
else:
    selected_indices = np.arange(len(labels))  # 选择整个数据集

# 组合特征
base_features = np.hstack([
    aac_features[selected_indices],
    be_features[selected_indices],
    pssm_features[selected_indices],
    cksaap_features[selected_indices]
])
if use_x_features:
    base_features = np.hstack([base_features, x_features[selected_indices]])

y = labels[selected_indices]

# 5 折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, f1_scores, aupr_scores = [], [], []

# 构建 MLP 模型
def build_small_nn(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

for fold, (train_idx, val_idx) in enumerate(kf.split(base_features, y), 1):
    print(f"Training fold {fold}...")
    X_train, X_val = base_features[train_idx], base_features[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model = build_small_nn(input_dim=X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=16, class_weight=class_weight_dict,
              callbacks=[early_stopping], verbose=0)

    y_pred_prob = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred_prob)
    f1 = f1_score(y_val, (y_pred_prob > 0.5).astype(int))
    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    aupr = average_precision_score(y_val, y_pred_prob)

    auc_scores.append(auc)
    f1_scores.append(f1)
    aupr_scores.append(aupr)

print(f'Average AUC: {np.mean(auc_scores):.4f}')
print(f'Average F1 Score: {np.mean(f1_scores):.4f}')
print(f'Average AUPR: {np.mean(aupr_scores):.4f}')
