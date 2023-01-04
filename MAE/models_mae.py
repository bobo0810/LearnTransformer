# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    Masked的自编码器，ViT为主干网络
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # MAE 编码器部分
        # 图像映射为patch
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  #patch数量 通常为14x14=196

        # 可学习参数[1,1,768]   位置0，用于分类
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码参数[1，197，768]  固定参数，不更新
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # 构建编码器  transformer encoder模块，重复N次
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        # 层归一化
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # MAE 解码器部分
        # 解码器的嵌入映射 768->512
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # 可学习参数[1,1,512]   解码器中充当已mask的patch特征，被解码器重建
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # 解码器位置编码参数[1，197，512] 固定参数，不更新
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # 构建解码器 transformer encoder模块，重复N次
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        # 解码器层归一化
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # 解码器预测 将特征解码成patch  512-> 16²*3,即768
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        # 初始化模型参数
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        将整图仅切成图像块patch，用于和预测值一起计算损失
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0] # patch尺寸 默认16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0 # 确认 图像为宽高一致、且整除patch

        h = w = imgs.shape[2] // p  # 分成14x14个patch
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        对patch进行随机mask
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence [batch,196,768]
        """
        N, L, D = x.shape  # batch, length196, dim768
        len_keep = int(L * (1 - mask_ratio)) # 未mask的patch数量  49
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] 生成0~1均匀分布的随机噪声 [batch,196]
        
        # sort noise for each sample
        # 打乱的索引 [batch,196]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 重建的索引  [batch,196]
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        # 未mask的patch索引  [batch,49]
        ids_keep = ids_shuffle[:, :len_keep]
        # 仅保留未mask的patch [batch,196,768] -> [batch,49,768]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        # mask 二值矩阵，用于重建 0保留 1mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # x_masked [batch,49,768]
        # mask [batch,196]
        # ids_restore [batch,196]
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        x 输入图像 [batch,3,224,224]
        mask_ratio  patch的mask比例
        """
        # embed patches
        # 将图像映射为patch编码 [batch,3,224,224]->[batch,196,768]
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # 逐元素相加的形式，添加位置编码，排除位置0的分类token
        x = x + self.pos_embed[:, 1:, :] # [batch,196,768]

        # masking: length -> length * mask_ratio
        # 对patch进行随机mask
        # x_masked [batch,49,768]
        # mask [batch,196]
        # ids_restore [batch,196]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        # [1,1,768]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # 矩阵广播  [1,1,768]-> [batch,1,768]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # 拼接位置0的分类token  [batch,1,768] + [batch,49,768] = [batch,50,768]
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # 通过N个transformer encoder提取特征
        for blk in self.blocks:
            x = blk(x)
        # 层归一化
        x = self.norm(x)
        # x [batch,50,768]   mask[batch,196]   ids_restore[batch,196]
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        解码器  解码器通过未mask的patch特征，学习重建已mask的patch
        x [batch,50,768]
        ids_restore[batch,196]
        """


        # embed tokens
        # 解码器的嵌入映射 [batch,50,768]->[batch,50,512]
        x = self.decoder_embed(x)

        # append mask tokens to sequence [batch,147,512]  196+1-50=147（147个已mask的patch，1个位置0编码，49个未mask的patch）
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # 除了位置0，将编码器输出的特征x 和 mask token拼接合并，得到整图的嵌入特征 [batch,196,512]
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # 根据重建索引，将整图的嵌入特征 还原回 原图mask时的对应位置。
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # 补上位置0编码  [batch,196,512]->[batch,197,512]
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        # 逐元素相加的形式，添加位置编码
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        # 通过N个transformer encoder提取特征
        for blk in self.decoder_blocks:
            x = blk(x)
        # 层归一化 [batch,197,512]
        x = self.decoder_norm(x)

        # predictor projection
        # 特征映射为patch   [batch,197,512]->[batch,197,768]
        x = self.decoder_pred(x)

        # remove cls token 移除位置0,得到[batch,196,768]
        x = x[:, 1:, :]
        # [batch,196,768]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]    eg:[batch,3,224,224]
        pred: [N, L, p*p*3]   eg:[batch,196,768]
        mask: [N, L], 0 is keep, 1 is remove, eg:[batch,196]二值矩阵 0未mask  1已mask
        """
        # 将整图仅切成图像块patch [batch,196,768]
        target = self.patchify(imgs)
        # 对原图像素规范化，默认关闭
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        # MSE均方误差损失
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # 仅对 解码器重建的已mask的patch计算损失。 无需对未mask的patch计算损失
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        imgs：输入图像 [batch,3,224,224]
        mask_ratio: patch的mask比例
        """
        # 编码器latent [batch,50,768]   mask[batch,196]   ids_restore[batch,196]
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # 解码器   [batch,196,768]
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        # loss
        # pred [batch,196,768]  整图的嵌入特征
        # mask [batch,196]二值矩阵 0未mask  1已mask
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    """
    以此为例，进行注解
    patch_size：patch尺寸
    embed_dim：嵌入特征维度  通常均为768
    depth: 编码器Block的数量 （Block指 Transformer Encoder模块）
    num_heads: 编码器Block的多头数量
    decoder_embed_dim：解码器的嵌入特征维度
    decoder_depth：解码器Block的数量（Block指 Transformer Encoder模块）
    decoder_num_heads：解码器Block的多头数量
    mlp_ratio：Block模块内的多层感知器MLP 特征扩展的倍数  通常为4
             流程：
             (1)768->（全连接层+激活函数+drop）-> 768*4
             (2)768*4->（全连接层+激活函数+drop）-> 768
    norm_layer: Block模块内均采用层归一化
    """
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # 不同规格的解码器decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

# 调试
if __name__ == "__main__":
    input=torch.randn([8,3,224,224])
    model = mae_vit_base_patch16()

    # 前向
    output=model(input)