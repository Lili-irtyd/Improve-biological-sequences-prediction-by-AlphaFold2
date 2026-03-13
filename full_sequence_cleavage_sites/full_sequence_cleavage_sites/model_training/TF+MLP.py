#使用4个特征拼接获得模型
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


# 创建 Transformer Encoder 模块
def transformer_encoder(inputs, num_heads=4, ff_dim=128, dropout_rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    attn_output = Add()([inputs, attn_output])
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)

    ff_output = Dense(ff_dim, activation='relu')(attn_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)

    ff_output = Add()([attn_output, ff_output])
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)
    return ff_output


# 加载数据
aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
X_AAC_Be_PSSM = np.hstack([aac_features, be_features, pssm_features])
X_CKSAAP = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
X_high_dim_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_windows.npy")
y = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy")

X_high_dim_pooled = np.mean(X_high_dim_features, axis=1)
# 拼接所有特征
X_all_features = np.hstack([X_AAC_Be_PSSM, X_CKSAAP,X_high_dim_pooled])  # 拼接 AAC+BE+PSSM 和 CKSAAP 特征
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, f1_scores = [], []
roc_fpr, roc_tpr = [], []  # 用于存储每一折的FPR和TPR
aupr_scores = []
pr_precision = []
pr_recall = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X_all_features, y), 1):
    print(f"Training fold {fold}...")

    X_all_train, X_all_val = X_all_features[train_idx], X_all_features[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_all_train = scaler.fit_transform(X_all_train)
    X_all_val = scaler.transform(X_all_val)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    input_transformer = Input(shape=(X_all_train.shape[1], 1))
    x = tf.transpose(input_transformer, perm=[0, 2, 1])
    x = transformer_encoder(x, num_heads=4, ff_dim=128)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_transformer, outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_all_train.reshape(-1, X_all_train.shape[1], 1), y_train, epochs=50, batch_size=32,
              validation_data=(X_all_val.reshape(-1, X_all_val.shape[1], 1), y_val),
              class_weight=class_weight_dict, verbose=0)

    y_pred_prob = model.predict(X_all_val.reshape(-1, X_all_val.shape[1], 1))

    auc = roc_auc_score(y_val, y_pred_prob)
    auc_scores.append(auc)

    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_fpr.append(fpr)
    roc_tpr.append(tpr)

    y_pred = (y_pred_prob > 0.5).astype(int)
    f1_scores.append(f1_score(y_val, y_pred))

    # --- AUPR ---
    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    pr_precision.append(precision)
    pr_recall.append(recall)
    aupr = average_precision_score(y_val, y_pred_prob)
    aupr_scores.append(aupr)

# 输出指标
# 输出指标
print(f'Average AUC: {np.mean(auc_scores):.4f}')
print(f'Average F1 Score: {np.mean(f1_scores):.4f}')
print(f'Average AUPR: {np.mean(aupr_scores):.4f}')


# 绘制AUC曲线
plt.figure(figsize=(10, 6))
for i in range(len(roc_fpr)):
    plt.plot(roc_fpr[i], roc_tpr[i], label=f'Fold {i + 1} (AUC = {auc_scores[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # 对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Stratified KFold Cross Validation')
plt.legend(loc='lower right')
plt.savefig(f"/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/Picture/AUC_T_5.svg", dpi=300)

# 绘制平均 PR 曲线（对每一折插值并平均）
import numpy as np
# 统一 recall 坐标轴
all_recalls = np.linspace(0, 1, 100)
mean_precision = np.zeros_like(all_recalls)

for p, r in zip(pr_precision, pr_recall):
    p_interp = np.interp(all_recalls, r[::-1], p[::-1])  # 需要逆序插值
    mean_precision += p_interp

mean_precision /= len(pr_precision)


plt.figure(figsize=(6, 5))
plt.plot(all_recalls, mean_precision, color='b', label=f'Mean PR (AUPR = {np.mean(aupr_scores):.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/Picture/AUPRT_5.svg", dpi=300)