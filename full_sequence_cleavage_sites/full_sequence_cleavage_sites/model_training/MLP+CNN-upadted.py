#4
#使用了混合精度
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate, LayerNormalization, \
    MultiHeadAttention, Add
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# 启用混合精度训练
from tensorflow.keras import mixed_precision

# 设置混合精度策略
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


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
y = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, pr_auc_scores, f1_scores = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_AAC_Be_PSSM, y), 1):
    print(f"\n===== Training Fold {fold} =====")

    # 训练集 & 验证集拆分
    X_AAC_Be_PSSM_train, X_AAC_Be_PSSM_val = X_AAC_Be_PSSM[train_idx], X_AAC_Be_PSSM[val_idx]
    X_CKSAAP_train, X_CKSAAP_val = X_CKSAAP[train_idx], X_CKSAAP[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 标准化
    scaler = StandardScaler()
    X_AAC_Be_PSSM_train = scaler.fit_transform(X_AAC_Be_PSSM_train)
    X_AAC_Be_PSSM_val = scaler.transform(X_AAC_Be_PSSM_val)

    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # MLP 处理低维特征
    input_mlp = Input(shape=(X_AAC_Be_PSSM_train.shape[1],))
    x_mlp = Dense(128, activation='relu')(input_mlp)
    x_mlp = Dropout(0.5)(x_mlp)
    x_mlp = Dense(64, activation='relu')(x_mlp)

    # CNN 处理 CKSAAP
    input_cnn = Input(shape=(X_CKSAAP_train.shape[1], 1))
    x_cnn = Conv1D(32, 3, activation='relu', padding='same')(input_cnn)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = Conv1D(64, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)

    # Transformer 处理 CNN 提取的特征
    x_cnn = transformer_encoder(x_cnn, num_heads=4, ff_dim=128)
    x_cnn = Flatten()(x_cnn)

    # 合并 MLP 和 Transformer
    x = Concatenate()([x_mlp, x_cnn])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', dtype='float32')(x)  # Explicitly set output dtype to float32

    model = Model(inputs=[input_mlp, input_cnn], outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # 使用混合精度训练
    # 将训练过程包装到 `tf.keras.mixed_precision` 中
    # 使用 GradientTape 进行训练（适用于混合精度训练）
    for epoch in range(50):
        with tf.GradientTape() as tape:
            output_train = model([X_AAC_Be_PSSM_train, X_CKSAAP_train.reshape(-1, X_CKSAAP_train.shape[1], 1)],
                                 training=True)
            loss = tf.keras.losses.binary_crossentropy(y_train, output_train)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 计算预测概率
    y_pred_prob = model.predict([X_AAC_Be_PSSM_val, X_CKSAAP_val.reshape(-1, X_CKSAAP_val.shape[1], 1)])

    # 计算 AUC-ROC
    auc = roc_auc_score(y_val, y_pred_prob)
    auc_scores.append(auc)

    # 计算 PR-AUC
    pr_auc = average_precision_score(y_val, y_pred_prob)
    pr_auc_scores.append(pr_auc)

    # 计算 F1-score
    y_pred = (y_pred_prob > 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    f1_scores.append(f1)

    print(f"Fold {fold}: AUC-ROC = {auc:.4f}, PR-AUC = {pr_auc:.4f}, F1-score = {f1:.4f}")

# 打印最终的平均分数
print("\n===== Final Results =====")
print(f'Average AUC-ROC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}')
print(f'Average PR-AUC: {np.mean(pr_auc_scores):.4f} ± {np.std(pr_auc_scores):.4f}')
print(f'Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')
