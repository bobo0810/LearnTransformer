import torch.nn as nn
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # 多头自注意力机制
        self.self_attn = MultiheadAttention(num_heads,embed_dim)

        # 层归一化层
        self.ln1=nn.LayerNorm(embed_dim)
        self.ln2=nn.LayerNorm(embed_dim)
        
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        

        # Dropout层
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        # 多头自注意力机制
        attn_output=self.self_attn(x,x,x,mask=mask)
        # post norm   Dropout+残差
        x=x+self.dropout(attn_output)
        # 层归一化层
        x = self.ln1(x)

        # 前馈神经网络
        x = x + self.dropout(self.ffn(x))

        # 层归一化层
        x = self.ln2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x