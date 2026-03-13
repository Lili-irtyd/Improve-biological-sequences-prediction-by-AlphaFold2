import torch
import torch.nn as nn
import numpy as np

class Self_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=4, num_neighbor=30, drop_rate=0):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_neighbor = num_neighbor
        self.dp = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,q,k,v,attention_mask=None,attention_weight=None):
        # q: bsz, protein_len, hid=heads*hidd'
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)    # q: bsz, heads, protein_len, hid'
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) 
        
        # 加上 mask (padding部分设为极小值)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # 距离权重注入
        if attention_weight is not None:
            # 简单粗暴的 Top-K 掩码
            # 注意：这里为了简化运算，直接乘上距离权重
            attention_probs = attention_probs * attention_weight
            # 重新归一化
            attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True) + 1e-8)

        outputs = torch.matmul(attention_probs, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        outputs = self.dp(outputs)
        outputs = self.ln(outputs)
        return outputs


class GTM(nn.Module):
    # --- [修改 1] 增加 cutoff_dist 参数 ---
    def __init__(self, protein_in_dim, protein_out_dim=64, target_dim=1, 
                 fc_layer_num=2, atten_layer_num=2, atten_head=4, 
                 num_neighbor=30, drop_rate1=0.2, drop_rate2=0.0, 
                 cutoff_dist=20.0): # <--- 新增参数
        super().__init__()
        
        self.cutoff_dist = cutoff_dist  # 保存阈值

        # 输入投影层
        self.input_block = nn.Sequential(
             nn.LayerNorm(protein_in_dim, elementwise_affine=True)
            ,nn.Linear(protein_in_dim, protein_out_dim)
            ,nn.LeakyReLU()
        )

        # 隐藏层
        self.hidden_block = []
        for h in range(fc_layer_num-1):
            self.hidden_block.extend([
                 nn.LayerNorm(protein_out_dim, elementwise_affine=True)
                ,nn.Dropout(drop_rate1)
                ,nn.Linear(protein_out_dim, protein_out_dim)
                ,nn.LeakyReLU()
            ])
        self.hidden_block = nn.Sequential(*self.hidden_block)

        # Attention 层
        self.layers = nn.ModuleList([Self_Attention(protein_out_dim, atten_head, num_neighbor, drop_rate2) for _ in range(atten_layer_num)])
        
        # 输出层
        self.logit = nn.Linear(protein_out_dim, target_dim)


    def forward(self, protein_node_features, protein_dist_matrix, protein_masks):
        """
        protein_node_features: [B, L, Dim]
        protein_dist_matrix: [B, L, L]
        protein_masks: [B, L] (1 for real, 0 for padding)
        """
        protein_embedding = self.input_block(protein_node_features)
        protein_embedding = self.hidden_block(protein_embedding)

        # 1. 计算 Soft Distance Weight (距离越近权重越大)
        # 修改 forward 函数中的这一行
        # 加上 1e-6 保护
        dist_weight = 1.0 / (torch.sqrt(1.0 + protein_dist_matrix) + 1e-6)
        dist_weight = dist_weight * protein_masks.unsqueeze(1) # Mask掉padding的行
        dist_weight = dist_weight.unsqueeze(1) # [B, 1, L, L]

        # --- [修改 2] 构建 Hard Mask (Padding + Distance) ---
        
        # A. Padding Mask: 处理序列长度不一
        # [B, 1, 1, L] (广播到所有 Head 和 Query)
        # 1.0 - masks: 1变成0(保留), 0变成1(屏蔽)
        padding_mask = (1.0 - protein_masks).unsqueeze(1).unsqueeze(1) * -1e9

        # B. Distance Hard Mask: 处理物理距离截断
        # [B, L, L] -> [B, 1, L, L]
        # 如果距离 > cutoff，则为 True (1.0)，需要屏蔽 (-1e9)
        # 如果距离 <= cutoff，则为 False (0.0)，保留 (0)
        is_far = (protein_dist_matrix > self.cutoff_dist).float()
        dist_hard_mask = is_far.unsqueeze(1) * -1e9
        
        # C. 合并 Mask
        # 只要是 Padding 或者 距离太远，都会变成很大的负数
        combined_mask = padding_mask + dist_hard_mask
        # ----------------------------------------------------

        for layer in self.layers:
            # 传入合并后的 Mask
            protein_embedding = layer(protein_embedding, protein_embedding, protein_embedding, 
                                      attention_mask=combined_mask, 
                                      attention_weight=dist_weight)

        y = self.logit(protein_embedding).squeeze(-1) 
        
        return torch.sigmoid(y)