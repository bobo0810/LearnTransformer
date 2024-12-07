import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        """
        初始化交叉注意力层。
        
        :param embed_dim: 嵌入向量的维度。
        """
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        # 线性变换层，用于将输入的查询、键和值映射到相同的维度空间
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        前向传播函数。
        
        :param query: 查询序列，形状为[batch_size, query_len, embed_dim]。
        :param key: 键序列，形状为[batch_size, key_len, embed_dim]。
        :param value: 值序列，形状为[batch_size, key_len, embed_dim]。
        :param mask: 掩码矩阵，用于屏蔽掉不需要关注的部分，形状为[batch_size, query_len, key_len]。
        :return: 输出的加权值序列。
        """
        # 线性变换查询、键和值
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # 计算查询和键的点积，得到注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        
        # 如果提供了掩码，就应用掩码
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 通过softmax函数对注意力分数进行归一化，得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 使用注意力权重对值进行加权求和，得到最终的输出
        output = torch.matmul(attention_weights, value)
        
        return output

# 示例使用
batch_size, query_len, key_len, embed_dim = 2, 3, 4, 5
query = torch.randn(batch_size, query_len, embed_dim)  # 查询序列
key = torch.randn(batch_size, key_len, embed_dim)    # 键序列
value = torch.randn(batch_size, key_len, embed_dim)  # 值序列
mask = torch.ones(batch_size, query_len, key_len)     # 没有掩码

cross_attention = CrossAttention(embed_dim)
output = cross_attention(query, key, value, mask)
print(output.shape)  # 应该是[batch_size, query_len, embed_dim]
