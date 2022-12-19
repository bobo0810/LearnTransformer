import math

import torch
from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import init_rules
from titans.decorator import no_support


@no_support(['sp'])
class ViTSelfAttention(nn.Module):
    """
    多头自注意力Multi-Head Attention
    """
    def __init__(self,
                 hidden_size: int,# 隐藏层的尺寸 768
                 num_heads: int, # 多头的数量 12
                 attention_dropout: float, #自注意力的droput比例
                 dropout: float, # 线性层的droput比例
                 bias: bool = True, # 线性层是否启用偏置
                 dtype: dtype = None, #  torch的Linear的参数
                 init_method: str = 'torch'):
        super().__init__()
        #  隐藏层维度 = 头的特征长度 * 头的数量
        # 特征总长度768=12个头 * 每个头的特征维度64
        self.attention_head_size = hidden_size // num_heads # 64
        # 自注意力：输入通过线性层生成Q、K、V
        self.query_key_value = col_nn.Linear(hidden_size, # 输入特征 768维
                                             3 * hidden_size, # 输出QKV，维度均为768
                                             dtype=dtype,
                                             bias=bias,
                                             **init_rules[init_method]['transformer']) # 初始化参数的规则
        # 自注意力的droput
        self.attention_dropout = col_nn.Dropout(attention_dropout)
        # 线性层
        self.dense = col_nn.Linear(hidden_size, hidden_size, dtype=dtype, bias=True, **init_rules[init_method]['transformer'])
        # 线性层的droput
        self.dropout = col_nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # the size of x is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        # the size of qkv is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE*3)
        # x通过线性层生成qkv    [batch,197,768]->[batch,197,768*3]
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3 # 768
        num_attention_heads = all_head_size // self.attention_head_size # 头的数量 12
        # [batch,197,768*3] -> [batch,197,12,3*64] 12个头，每个头包含 维度均为64的QKV三个特征
        new_qkv_shape = qkv.shape[:-1] + \
            (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        # [batch,197,12,3*64]  -> [batch,12,197,3*64]
        qkv = qkv.permute((0, 2, 1, 3))
        # the size of q is (BATCH_SZIE, NUM_HEADS, SEQ_LEN, HIDDEN_SIZE//NUM_HEADS)
        # 按最后一维拆分为3块，得到q k v,三者尺寸相同  均为[batch,12,197,64]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # the size of x is (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        # q矩阵 和 k矩阵的转置 相乘，得到两者之间的相似度
        x = torch.matmul(q, k.transpose(-1, -2))
        # 防止方差过大，需乘一个尺度因子。使得 x的分布重新变为 均值为0，方差为1
        x = x / math.sqrt(self.attention_head_size)
        # 将结果转为 概率分布，得到自注意力权重
        x = self.softmax(x)
        x = self.attention_dropout(x)

        # the size of x after matmul is (BATCH_SZIE, NUM_HEADS, SEQ_LEN, HIDDEN_SIZE//NUM_HEADS)
        # [batch,12,197,64]
        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size,)
        # the size of x after reshape is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        x = x.reshape(new_context_layer_shape)
        # the size of x after dense is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        x = self.dense(x)
        x = self.dropout(x)

        return x
