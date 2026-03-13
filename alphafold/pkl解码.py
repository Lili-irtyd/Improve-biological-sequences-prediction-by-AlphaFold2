import pickle

# 加载 .pkl 文件
# with open('/home/gpux1/ccd/alphaflow/output2019_alphaflow_negative/P07318_4.pkl', 'rb') as file:
#     data = pickle.load(file)

with open('/home/gpux1/ccd/output_features_442/P20152_33/result_model_1_pred_0.pkl', 'rb') as file:
    data = pickle.load(file)
# 输出所有的键并查看每个键对应的数据类型或部分数据
if isinstance(data, dict):  # 检查数据是否为字典类型
    print("Keys in the .pkl file:")
    for key in data.keys():
        print(f"\nKey: {key}")
        # 检查每个键对应的值的类型并显示部分内容
        value = data[key]
        if isinstance(value, list):
            print(f"Type: List, Length: {len(value)}")
            print(f"First 5 items: {value[:5]}")  # 输出列表的前5个元素
        elif isinstance(value, dict):
            print(f"Type: Dictionary, Keys: {list(value.keys())[:5]}")  # 输出字典的前5个键
        elif isinstance(value, str):
            print(f"Type: String, Length: {len(value)}")
            print(f"Content: {value[:50]}")  # 输出字符串的前50个字符
        else:
            print(f"Type: {type(value)}, Value: {value}")
else:
    print("The .pkl file does not contain a dictionary.")

