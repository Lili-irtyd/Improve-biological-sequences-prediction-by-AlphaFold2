from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# 创建StratifiedKFold对象，指定5折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用来保存每一折的AUC、F1-score
auc_scores = []
f1_scores = []
roc_fpr = []
roc_tpr = []
roc_thresholds = []
# 假设你有这些输入特征
# X_AAC_Be_PSSM: 形状 (样本数, 3*特征维度) -> 例如：AAC + BE + PSSM
# X_CKSAAP: 形状 (样本数, 1600) -> 例如：CKSAAP，1600维
# y: 标签，二分类，形状为 (样本数, 1)

import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy


aac_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/aac.npy")  # (26844,20)
be_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/be.npy")  # (26844,600)  # (26844,1600)
pssm_features = np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/pssm.npy")
X_AAC_Be_PSSM = np.hstack([aac_features, be_features,pssm_features])
X_CKSAAP=np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/features-45/cksaap_fixed.npy")
y= np.load("/user1/scl1/zhiqian/full_sequence_cleavage_sites/full_sequence_cleavage_sites/combined_labels.npy")  # (26844,)
# 交叉验证
pr_precisions = []  # 存储每个fold的precision值
pr_recalls = []  # 存储每个fold的recall值
aupr_scores = []  # 存储每个fold的AUPR
for fold, (train_idx, val_idx) in enumerate(kf.split(X_AAC_Be_PSSM, y), 1):
    print(f"Training fold {fold}...")

    # 拆分数据集
    X_AAC_Be_PSSM_train, X_AAC_Be_PSSM_val = X_AAC_Be_PSSM[train_idx], X_AAC_Be_PSSM[val_idx]
    X_CKSAAP_train, X_CKSAAP_val = X_CKSAAP[train_idx], X_CKSAAP[val_idx]
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

    # CNN部分处理CKSAAP特征
    input_cnn = Input(shape=(X_CKSAAP_train.shape[1], 1))  # 1600维，1通道
    x_cnn = Conv1D(32, 3, activation='relu')(input_cnn)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = Conv1D(64, 3, activation='relu')(x_cnn)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = Flatten()(x_cnn)

    # 合并MLP和CNN的输出
    x = Concatenate()([x_mlp, x_cnn])

    # 分类部分
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)  # 二分类任务

    # 创建模型
    model = Model(inputs=[input_mlp, input_cnn], outputs=output)

    # 编译模型
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit([X_AAC_Be_PSSM_train, X_CKSAAP_train.reshape(X_CKSAAP_train.shape[0], X_CKSAAP_train.shape[1], 1)],
              y_train, epochs=50, batch_size=32,
              validation_data=(
              [X_AAC_Be_PSSM_val, X_CKSAAP_val.reshape(X_CKSAAP_val.shape[0], X_CKSAAP_val.shape[1], 1)], y_val),
              class_weight=class_weight_dict, verbose=0)

    # 评估模型
    y_pred_prob = model.predict(
        [X_AAC_Be_PSSM_val, X_CKSAAP_val.reshape(X_CKSAAP_val.shape[0], X_CKSAAP_val.shape[1], 1)])

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

    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    pr_precisions.append(precision)
    pr_recalls.append(recall)

    # 计算AUPR
    aupr = average_precision_score(y_val, y_pred_prob)
    aupr_scores.append(aupr)

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

# 计算平均 AUPR
average_aupr = np.mean(aupr_scores)

# 打印平均 AUC、F1-score 和 AUPR
print(f'Average AUC: {np.mean(auc_scores):.4f}')
print(f'Average F1 Score: {np.mean(f1_scores):.4f}')
print(f'Average AUPR: {average_aupr:.4f}')
