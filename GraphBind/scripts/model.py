import torch
import torch.nn as nn

class Simple1DCNN(nn.Module):
    def __init__(self, input_channels=384, output_channels=1):
        super(Simple1DCNN, self).__init__()
        
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, output_channels, kernel_size=1)
        )

    def forward(self, x):
        # (B, 384, L) -> (B, 1, L)
        logits = self.conv_stack(x)
        
        # --- !! 关键修改 !! ---
        # 将 Logits 转换为 0-1 之间的概率，以匹配 BCELoss
        return torch.sigmoid(logits)
        # --- 结束修改 ---