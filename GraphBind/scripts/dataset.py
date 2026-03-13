import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def parse_annotation_file(txt_path):
    """
    复用你的逻辑：从 .txt (FASTA 格式) 文件中解析 序列ID 和 标签字符串。
    返回一个字典: {'1A2P_A': '000110...', ...}
    """
    seqanno = {}
    filename=os.path.basename(txt_path)
    is_train_name=False
    if "Train" in filename or "train" in filename:
        is_train_name=True
    else:
        is_train_name=False
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        # 遍历所有行，寻找FASTA头 ('>')
        for i, line in enumerate(lines):
            if line.startswith('>'):
                try:
                    query_id = line.strip()[1:]
                    # 假设标签总是在序列之后的那一行
                    # (你需要根据你的文件格式确认这里的行偏移量是 +2 还是 +3)
                    query_anno=""
                    if is_train_name:
                        if i + 3 < len(lines):
                            query_anno = lines[i + 3].strip()
                    else:
                        if i + 2 < len(lines):
                            query_anno = lines[i + 2].strip()
                    # 统一的验证和保存
                    if all(c in '01' for c in query_anno) and len(query_anno) > 10:
                        seqanno[query_id] = query_anno
                            
                except IndexError:
                    continue # 文件末尾

    except Exception as e:
        print(f"!!! 错误：无法解析标签文件 {txt_path}。")
        print(f"!!! 错误信息: {e}")
        raise
        
    print(f"成功从 {txt_path} 加载了 {len(seqanno)} 条标签。")
    return seqanno

class SequenceDataset(Dataset):
    def __init__(self, feature_dir, annotation_txt_file):
        """
        自定义数据集
        
        参数:
        feature_dir (str): A2 (s-rep) .npy 文件所在的目录 (例如 .../AF2/train)
        annotation_txt_file (str): 包含标签字符串的 .txt 文件
        """
        self.feature_dir = feature_dir
        
        # 1. 解析标签文件，构建 seqanno 字典
        self.seqanno = parse_annotation_file(annotation_txt_file)
        
        # 2. 交叉验证：只保留那些 "特征" 和 "标签" 都存在的 ID
        self.file_ids = []
        for seqid in self.seqanno.keys():
            # 假设你的 .npy 文件名就是 seqid.npy
            feature_path = os.path.join(self.feature_dir, f"{seqid}.npy")
            if os.path.exists(feature_path):
                self.file_ids.append(seqid)
            else:
                print(f"警告：标签 {seqid} 存在，但未找到特征文件 {feature_path}")

        print(f"最终用于 {os.path.basename(annotation_txt_file)} 的匹配数据：{len(self.file_ids)} 条")

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        seqid = self.file_ids[idx]
        
        # 1. 加载 s-rep 特征 (L, 384)
        feature_path = os.path.join(self.feature_dir, f"{seqid}.npy")
        try:
            s_rep = torch.from_numpy(np.load(feature_path)).float()
        except Exception as e:
            print(f"!!! 错误：加载 .npy 文件失败 {feature_path}。错误: {e}")
            return None # 返回 None，让 collate_fn 忽略
        
        # 2. 加载标签 (从字典中获取字符串 "00110..." 并转换为 Tensor)
        label_str = self.seqanno[seqid]
        label = torch.tensor([int(c) for c in label_str], dtype=torch.float32)
        
        # 3. 【关键】验证长度是否一致
        if s_rep.shape[0] != label.shape[0]:
            print(f"!!! 严重错误：{seqid} 特征与标签长度不匹配！")
            print(f"    s-rep 长度: {s_rep.shape[0]}")
            print(f"    标签长度: {label.shape[0]}")
            return None # 返回 None，让 collate_fn 忽略
            
        # 4. 调整标签维度为 (L, 1) 以便填充
        label = label.unsqueeze(-1)
            
        return s_rep, label

def collate_fn_pad(batch):
    """
    自定义的 collate_fn 来处理可变长度的序列和 None (由错误引起)。
    """
    # 1. 过滤掉 None 样本
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None # 返回空

    # 2. 解包 s-reps 和 labels
    s_reps, labels = zip(*batch)
    
    # 3. 获取每个序列的真实长度
    lengths = torch.tensor([s.shape[0] for s in s_reps])
    
    # 4. 填充 (Pad) s-reps 和 labels
    padded_s_reps = pad_sequence(s_reps, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0.0)
    
    # 5. 创建掩码 (Mask)
    max_len = padded_s_reps.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    
    # 6. 调整 s-rep 的维度以匹配 Conv1d 的输入
    # (B, L_max, C) -> (B, C, L_max)
    padded_s_reps = padded_s_reps.permute(0, 2, 1)

    return padded_s_reps, padded_labels, mask