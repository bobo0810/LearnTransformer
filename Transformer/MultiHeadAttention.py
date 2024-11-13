import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        # 确保模型维度可以被头数整除
        assert d_model % num_heads == 0

        # 计算每个注意力头的维度
        self.d_k = d_model // num_heads  # 每个头的维度
        self.num_heads = num_heads  # 注意力头的数量

        # 定义线性层用于查询（Q）、键（K）、值（V）的投影
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]  # 创建三个线性层
        )

        # 输出的线性层，用于将拼接后的多头输出映射回原始维度
        self.output_linear = nn.Linear(d_model, d_model)

        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)  # 获取批次大小

        # 1) 对输入的Q, K, V进行线性变换并拆分成多个头 分解为h * d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) 应用注意力机制   最终输出和注意力权重
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 将多个头的输出拼接起来，并通过最终的线性层
        x = (
            x.transpose(1, 2)  # 交换维度以便合并
            .contiguous()  # 确保内存连续性
            .view(batch_size, -1, self.num_heads * self.d_k)  # 合并头
        )

        return self.output_linear(x)  # 通过输出线性层返回最终结果

    def attention(self, query, key, value, mask=None, dropout=None):
        "计算'可缩放点积注意力'"
        d_k = query.size(-1)  # 获取每个头的维度  拆分后的特征维度
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # 应用掩蔽，防止对某些位置的注意力计算  对于mask为0的位置填充为负无穷-1e9
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重，使用softmax归一化
        p_attn = scores.softmax(dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)  # 应用dropout以防止过拟合

        # 计算加权值
        return torch.matmul(p_attn, value), p_attn  # 返回加权后的值和注意力权重


# 生成 mask 的示例
def generate_mask(seq_lengths, max_len):
    """
    根据序列的实际长度生成填充 mask。

    :param seq_lengths: 实际长度的列表
    :param max_len: 最大序列长度
    :return: mask张量
    """
    batch_size = len(seq_lengths)
    mask = torch.ones(batch_size, max_len)  # 初始化为全 1

    for i, length in enumerate(seq_lengths):
        mask[i, length:] = 0  # 填充位置设置为 0

    return mask.bool()  # 转换为布尔类型


# 使用示例
if __name__ == "__main__":
    d_model = 512  # 模型的维度
    num_heads = 8  # 注意力头的数量
    batch_size = 32  # 批次大小
    seq_len = 64  # 序列长度

    # # 生成 mask   假设有batch个序列，每个元素为序列的实际长度
    # seq_lengths = torch.randint(1, 65, (batch_size,)).tolist()
    # max_len = 5  # 最大序列长度
    # mask = generate_mask(seq_lengths, max_len)
    mask = None

    # 假设输入张量，形状为 [batch_size, seq_len, d_model]
    input_tensor = torch.randn(batch_size, seq_len, d_model)

    # 创建多头注意力实例
    mha = MultiHeadAttention(d_model, num_heads)

    # 进行前向传播
    output = mha(
        input_tensor, input_tensor, input_tensor, mask
    )  # 使用相同的输入作为Q, K, V  [batch_size, seq_len, d_model],即[32，64，512]
    print(output.shape)  # 应该输出: torch.Size([32, 64, 512])
