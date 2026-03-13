#生成模型文件类型h5后观察模型拟合情况
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
# 加载模型
model = load_model('/home/gpux1/CCPR/full_sequence_cleavage_sites/model_training/DeepCalpain_model_with_x.h5')
model.summary()

with open("model_architecture_x.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

