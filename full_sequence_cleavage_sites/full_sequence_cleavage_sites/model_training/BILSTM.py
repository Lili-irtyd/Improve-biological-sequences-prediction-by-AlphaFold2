from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate

# 创建StratifiedKFold对象，指定5折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用来保存每一折的AUC、F1-score
auc_scores = []
f1_scores = []
roc_fpr = []
roc_tpr = []
roc_thresholds = []

# 假设你有这些输入特征
aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")  # (26844,20)
be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")  # (26844,600)
pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
X_AAC_Be_PSSM = np.hstack([aac_features, be_features, pssm_features])
X_CKSAAP = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap.npy")
#X_high_dim_features = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_windows.npy")  # (26844, 30, 384)
y = np.load("/home/gpux1/CCPR/full_sequence_cleavage_sites/combined_labels.npy")  # (26844,)

# 交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(X_AAC_Be_PSSM, y), 1):
    print(f"Training fold {fold}...")

    # 拆分数据集
    X_AAC_Be_PSSM_train, X_AAC_Be_PSSM_val = X_AAC_Be_PSSM[train_idx], X_AAC_Be_PSSM[val_idx]
    X_CKSAAP_train, X_CKSAAP_val = X_CKSAAP[train_idx], X_CKSAAP[val_idx]
    #X_high_dim_train, X_high_dim_val = X_high_dim_features[train_idx], X_high_dim_features[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 标准化输入数据
    scaler = StandardScaler()
    X_AAC_Be_PSSM_train = scaler.fit_transform(X_AAC_Be_PSSM_train)
    X_AAC_Be_PSSM_val = scaler.transform(X_AAC_Be_PSSM_val)

    # 计算类权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # 设计模型
    # MLP部分处理AAC, BE, PSSM特征
    input_mlp = Input(shape=(X_AAC_Be_PSSM_train.shape[1],))
    x_mlp = Dense(128, activation='relu')(input_mlp)
    x_mlp = Dropout(0.5)(x_mlp)
    x_mlp = Dense(64, activation='relu')(x_mlp)

    # BiLSTM部分处理高维特征（形状：30, 384）
    input_high_dim = Input(shape=(X_high_dim_train.shape[1], X_high_dim_train.shape[2]))  # 30, 384
    x_high_dim = Bidirectional(LSTM(64, return_sequences=False))(input_high_dim)  # 双向LSTM
    x_high_dim = Dropout(0.5)(x_high_dim)  # 添加Dropout防止过拟合

    # CNN部分处理CKSAAP特征
    input_cnn = Input(shape=(X_CKSAAP_train.shape[1], 1))  # 1600维，1通道
    x_cnn = Conv1D(32, 3, activation='relu')(input_cnn)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = Conv1D(64, 3, activation='relu')(x_cnn)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = Flatten()(x_cnn)

    # 合并MLP、CNN（CKSAAP）和BiLSTM（高维特征）的输出
    x = Concatenate()([x_mlp, x_cnn, x_high_dim])

    # 分类部分
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)  # 二分类任务

    # 创建模型
    model = Model(inputs=[input_mlp, input_cnn, input_high_dim], outputs=output)

    # 编译模型
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit([X_AAC_Be_PSSM_train, X_CKSAAP_train.reshape(X_CKSAAP_train.shape[0], X_CKSAAP_train.shape[1], 1),
               X_high_dim_train], y_train, epochs=50, batch_size=32,
              validation_data=([X_AAC_Be_PSSM_val, X_CKSAAP_val.reshape(X_CKSAAP_val.shape[0], X_CKSAAP_val.shape[1], 1),
                               X_high_dim_val], y_val),
              class_weight=class_weight_dict, verbose=0)

    # 评估模型
    y_pred_prob = model.predict([X_AAC_Be_PSSM_val, X_CKSAAP_val.reshape(X_CKSAAP_val.shape[0], X_CKSAAP_val.shape[1], 1),
                                 X_high_dim_val])

    # 计算AUC
    auc = roc_auc_score(y_val, y_pred_prob)
    auc_scores.append(auc)

    # 计算F1-score
    y_pred = (y_pred_prob > 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    f1_scores.append(f1)

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
    roc_fpr.append(fpr)
    roc_tpr.append(tpr)
    roc_thresholds.append(thresholds)

# 绘制AUC曲线
plt.figure(figsize=(10, 6))
for i in range(len(roc_fpr)):
    plt.plot(roc_fpr[i], roc_tpr[i], label=f'Fold {i + 1} (AUC = {auc_scores[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # 对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Stratified KFold Cross Validation')
plt.legend(loc='lower right')
plt.show()

# 打印平均AUC和F1-score
print(f'Average AUC: {np.mean(auc_scores):.4f}')
print(f'Average F1 Score: {np.mean(f1_scores):.4f}')
