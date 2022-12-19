import torch
from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import init_rules


class ViTEmbedding(nn.Module):
    """
    Construct the patch embeddings.
    线性投射层

    Args:
        img_size(int): The size of images.图像尺寸
        patch_size(int): The size of patches. Patch尺寸
        in_chans(int): The size of input channels. 图像输入通道
        embedding_dim(int): The embedding size of patches. Patch特征维度 默认768
        dropout(float): The ratio used to construct dropout modules, which indicates the percentage of parameters should be casted to zero.
        dtype (:class:`torch.dtype`): The dtype of parameters, defaults to None.
        flatten(bool): If set to ``False``, the patches will not be flatten, defaults to ``True``. True将特征拉直
        init_method(str): The initializing method used in layers, defaults to `torch`. 模型参数的初始化策略
    """

    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embedding_dim: int,
                 dropout: float,
                 dtype: dtype = None,
                 flatten: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        # 对输入图像进行分块和展平操作
        self.patch_embed = col_nn.PatchEmbedding(img_size,
                                                 patch_size,
                                                 in_chans,
                                                 embedding_dim,
                                                 dtype=dtype,
                                                 flatten=flatten,
                                                 **init_rules[init_method]['embed'])
        self.dropout = col_nn.Dropout(dropout)

    def forward(self, x):
        # the size of x before embed is (BATCH_SIZE, IN_CHAN, IMAGE_SIZE, IMAGE_SIZE)
        # the size of x after embedding is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        # 以16x16patch为例，224x224图像被分为14x14=197个pacth   每个patch（3x16x16=768）再通过线性层映射为768维特征
        # 输入[batch,3,224,224]   输出[batch,197,768]
        x = self.patch_embed(x)
        x = self.dropout(x)
        return x
