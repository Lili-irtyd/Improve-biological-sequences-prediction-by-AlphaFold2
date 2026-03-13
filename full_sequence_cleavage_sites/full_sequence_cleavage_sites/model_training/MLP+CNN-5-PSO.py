import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, average_precision_score  # AUPR 计算
from pyswarm import pso  # PSO 进行超参数优化

# 读取数据
aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")
be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")
pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
X_AAC_Be_PSSM = np.hstack([aac_features, be_features, pssm_features])
X_CKSAAP = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
X_high_dim_features = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")
y = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# PSO 目标函数
def objective_function(params):
    learning_rate, dropout_rate, units = params
    units = int(units)

    auc_scores = []
    aupr_scores = []  # 记录 AUPR

    for train_idx, val_idx in kf.split(X_AAC_Be_PSSM, y):
        # 划分数据集
        X_train, X_val = X_AAC_Be_PSSM[train_idx], X_AAC_Be_PSSM[val_idx]
        X_CKSAAP_train, X_CKSAAP_val = X_CKSAAP[train_idx], X_CKSAAP[val_idx]
        X_high_dim_train, X_high_dim_val = X_high_dim_features[train_idx], X_high_dim_features[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # 计算类权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        # MLP 处理 AAC, BE, PSSM
        input_mlp = Input(shape=(X_train.shape[1],))
        x_mlp = Dense(units, activation='relu')(input_mlp)
        x_mlp = Dropout(dropout_rate)(x_mlp)
        x_mlp = Dense(units // 2, activation='relu')(x_mlp)

        # CNN 处理 CKSAAP
        input_cnn = Input(shape=(X_CKSAAP_train.shape[1], 1))
        x_cnn = Conv1D(32, 3, activation='relu')(input_cnn)
        x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
        x_cnn = Flatten()(x_cnn)

        # CNN 处理高维特征
        input_high_dim = Input(shape=(X_high_dim_train.shape[1], X_high_dim_train.shape[2]))
        x_high_dim = Conv1D(32, 3, activation='relu')(input_high_dim)
        x_high_dim = MaxPooling1D(pool_size=2)(x_high_dim)
        x_high_dim = Flatten()(x_high_dim)

        # 合并特征
        x = Concatenate()([x_mlp, x_cnn, x_high_dim])
        x = Dense(256, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[input_mlp, input_cnn, input_high_dim], outputs=output)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        model.fit([X_train, X_CKSAAP_train.reshape(-1, X_CKSAAP_train.shape[1], 1), X_high_dim_train], y_train,
                  epochs=30, batch_size=32, class_weight=class_weight_dict, verbose=0)

        y_pred_prob = model.predict([X_val, X_CKSAAP_val.reshape(-1, X_CKSAAP_val.shape[1], 1), X_high_dim_val])

        # 计算 AUC 和 AUPR
        auc = roc_auc_score(y_val, y_pred_prob)
        aupr = average_precision_score(y_val, y_pred_prob)

        auc_scores.append(auc)
        aupr_scores.append(aupr)

    mean_auc = np.mean(auc_scores)
    mean_aupr = np.mean(aupr_scores)

    print(f"AUC: {mean_auc:.4f}, AUPR: {mean_aupr:.4f}, Params: {params}")

    return -mean_auc  # PSO 需要最小化目标函数，因此取负值，但只优化 AUC


# PSO 参数范围
lb = [0.0001, 0.2, 64]  # 最小值: learning_rate, dropout_rate, units
ub = [0.01, 0.6, 512]  # 最大值: learning_rate, dropout_rate, units

# 运行 PSO
best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=5)

# 计算最终的 mean AUC 和 mean AUPR
final_auc = -objective_function(best_params)  # 由于 PSO 最小化目标函数，需要取负值
print(f"Best Hyperparameters (Learning Rate, Dropout Rate, Units): {best_params}")
print(f"Final Mean AUC: {final_auc:.4f}")

# 计算最终 mean AUPR
learning_rate, dropout_rate, units = best_params
units = int(units)

final_aupr_scores = []
for train_idx, val_idx in kf.split(X_AAC_Be_PSSM, y):
    X_train, X_val = X_AAC_Be_PSSM[train_idx], X_AAC_Be_PSSM[val_idx]
    X_CKSAAP_train, X_CKSAAP_val = X_CKSAAP[train_idx], X_CKSAAP[val_idx]
    X_high_dim_train, X_high_dim_val = X_high_dim_features[train_idx], X_high_dim_features[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    y_pred_prob = model.predict([X_val, X_CKSAAP_val.reshape(-1, X_CKSAAP_val.shape[1], 1), X_high_dim_val])
    aupr = average_precision_score(y_val, y_pred_prob)
    final_aupr_scores.append(aupr)

final_aupr = np.mean(final_aupr_scores)
print(f"Final Mean AUPR: {final_aupr:.4f}")
