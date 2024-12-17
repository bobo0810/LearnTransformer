import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads  # 头的数量
        self.head_dim = embed_dim // num_heads  # 每个头的维度

        # q k v
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 输出线性层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def split(self, x, batch_size):
        # 维度切分，由(batch_size, seq_len, embed_dim)变为(batch_size,seq_len, num_heads,head_dim)
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        # 转置维度，变为(batch_size, num_heads, seq_len, head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)
        query, key, value = (
            self.split(query, batch_size),
            self.split(key, batch_size),
            self.split(value, batch_size),
        )
        # transpose交换张量维度，相比permute只能交换两个
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim)
        )
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, float("-inf")
            )

        # attention_scores维度（batch_size, num_heads, seq_len, seq_len）
        # attention_probs维度（batch_size, num_heads, seq_len, seq_len）
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value)
        # output维度（batch_size, seq_len, head_dim）
        output = (
            output.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, -1, self.head_dim * self.num_heads)
        )
        output = self.out_proj(output)
        return output


if __name__ == "__main__":
    # 模拟输入
    batch_size, seq_len, embed_dim, num_heads = 2, 4, 8, 2
    x = torch.randn(batch_size, seq_len, embed_dim)
    attention_mask = torch.ones(batch_size, 1, 1, seq_len)  # 简单全1 mask

    # 实例化
    mha = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim)
    output = mha(x, attention_mask)
    print(output.shape)  # 预期输出: (batch_size, seq_len, embed_dim)
