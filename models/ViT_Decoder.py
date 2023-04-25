# @Time : 2023/4/24 18:00 
# @Author : Li Jiaqi
# @Description :
from functools import partial
import torch
from dinov2_source.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from dinov2_source.vision_transformer import BlockChunk
import torch.nn as nn
from typing import Sequence, Tuple, Union, Callable
from torch.nn.init import trunc_normal_


class Decoder(nn.Module):
    def __init__(self, code_size,
                 img_size=420,
                 patch_size=14,
                 out_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 ffn_bias=True,
                 proj_bias=True,
                 drop_path_rate=0.0,
                 init_values=None,  # for layerscale: None or 0 => no layerscale
                 act_layer=nn.GELU,
                 block_fn=partial(Block, attn_class=MemEffAttention),
                 block_chunks=1, ):
        super(Decoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=Mlp,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        self.fc1 = nn.Linear(code_size, code_size)
        chunked_blocks = []
        chunksize = depth // block_chunks
        for i in range(0, depth, chunksize):
            # this is to keep the block index consistent if we chunk the block list
            chunked_blocks.append([nn.Identity()] * i + blocks_list[i: i + chunksize])
        self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        self.norm_token = norm_layer(embed_dim)
        # decoder to patch
        # self.decoder_pred = nn.Linear(code_size, patch_size ** 2 * in_chans, bias=True)
        self.decoder_pred = nn.ConvTranspose2d(code_size, out_chans, patch_size, patch_size)
        self.norm_image = norm_layer(out_chans)

        self.init_weights()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.fc1(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm_token(x)  # B,P,N
        # recovery
        x = x.view(-1, self.img_size // self.patch_size, self.img_size // self.patch_size, self.embed_dim)  # B,H,W,N
        x = x.permute(0, 3, 1, 2)  # B,N,H,W
        x = self.decoder_pred(x)
        # norm
        x = x.permute(0, 2, 3, 1)  # B,H,W,N
        x = self.norm_image(x)
        x = x.permute(0, 3, 1, 2)  # B,N,H,W
        return x

    def init_weights(self):
        named_apply(init_weights_vit_timm, self)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
