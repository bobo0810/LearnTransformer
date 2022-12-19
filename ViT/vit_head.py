from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import init_rules


class ViTHead(nn.Module):
    """
    MLP Head 输出最终预测结果
    """
    def __init__(self,
                 hidden_size: int,
                 num_classes: int,
                 representation_size: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        if representation_size: # 特征通过线性层进行再表达
            self.representation = col_nn.Linear(hidden_size,
                                                representation_size,
                                                bias=bias,
                                                dtype=dtype,
                                                **init_rules[init_method]['head'])
        else:
            self.representation = None
            representation_size = hidden_size
        # 分类层（全连接层）
        self.dense = col_nn.Classifier(representation_size,
                                       num_classes,
                                       dtype=dtype,
                                       bias=bias,
                                       **init_rules[init_method]['head'])

    def forward(self, x):
        # the size of x is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        # 取位置0的特征即可（自注意力使得其包含了全局特征） [batch,197,768]->[batch,768]
        x = x[:, 0]
        # the size of x is (BATCH_SIZE, HIDDEN_SIZE)
        if self.representation is not None:
            x = self.representation(x)
            # the size of x after representation is (BATCH_SIZE, REPRESENTATION_SIZE)
        # [batch,768] -> [batch,1000]
        x = self.dense(x)
        # the size of x after dense is (BATCH_SIZE, NUM_CLASSES)
        return x
