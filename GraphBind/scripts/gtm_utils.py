import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def collate_fn_gtm(batch):
    """
    将 PyG 的 Data 对象列表转换为 GTM 需要的 Padding Tensor。
    """
    # 1. 过滤 None 数据
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None, None

    # 2. 提取数据
    # data.x: 节点特征
    # data.pos: 坐标
    # data.y: 标签
    xs = [data.x for data in batch]
    poses = [data.pos for data in batch]
    ys = [data.y for data in batch]
    
    # 3. 获取长度信息
    lengths = torch.tensor([x.shape[0] for x in xs])
    max_len = max(lengths)
    batch_size = len(xs)
    
    # 4. Padding 特征 (X) 和 标签 (Y)
    # x_pad: [Batch, Max_Len, Feat_Dim]
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    # y_pad: [Batch, Max_Len]
    y_pad = pad_sequence(ys, batch_first=True, padding_value=0.0)
    
    # 如果 y 是 [L, 1] 形状，squeeze掉最后一维
    if y_pad.dim() == 3:
        y_pad = y_pad.squeeze(-1)

    # 5. 生成 Mask (真实节点为 1，Padding 为 0)
    # mask: [Batch, Max_Len]
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    mask = mask.float() 

    # 6. 计算并 Padding 距离矩阵 (Distance Matrix)
    # dist_matrix: [Batch, Max_Len, Max_Len]
    dist_matrix = torch.zeros((batch_size, max_len, max_len), dtype=torch.float)
    
    for i, pos in enumerate(poses):
        L = lengths[i]
        # 计算欧几里得距离 (L, 3) -> (L, L)
        dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
        # 填入大矩阵
        dist_matrix[i, :L, :L] = dist

    return x_pad, dist_matrix, mask, y_pad