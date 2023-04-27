# @Time : 2023/4/24 18:00 
# @Author : Li Jiaqi
# @Description :
from functools import partial
import torch
from dinov2_source.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
import torch.nn as nn
from typing import Callable
from torch.nn.init import trunc_normal_
import math


class Decoder(nn.Module):
    def __init__(self,
                 img_size=(420, 420),
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
                 init_values=1.0e-05,  # for layerscale: None or 0 => no layerscale
                 act_layer=nn.GELU,
                 block_fn=partial(Block, attn_class=MemEffAttention),
                 needs_fc=False, ):
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
        if needs_fc:
            self.fc1 = nn.Linear(embed_dim, embed_dim)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.chunked_blocks = False
        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)
        self.norm_token = norm_layer(embed_dim)
        # decoder to patch
        self.decoder_pred_linear = nn.Linear(embed_dim, patch_size ** 2 * out_chans, bias=True)
        self.decoder_pred_conv = nn.ConvTranspose2d(embed_dim, out_chans, patch_size, patch_size)
        self.norm_image = norm_layer(out_chans)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.needs_fc = needs_fc

        self.init_weights()

    def forward_features(self, x):
        if self.needs_fc:
            x = self.fc1(x)
        self.norm_token(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm_token(x)  # B,P,N
        # recovery
        x = x.view(-1, self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size,
                   self.embed_dim)  # B,H,W,N
        x = x.permute(0, 3, 1, 2)  # B,N,H,W
        x = self.decoder_pred(x)
        # norm
        x = x.permute(0, 2, 3, 1)  # B,H,W,N
        x = self.norm_image(x)
        x = x.permute(0, 3, 1, 2)  # B,N,H,W
        return x

    def forward(self, x, linear=True):
        if self.needs_fc:
            x = self.fc1(x)
        x = x + self.interpolate_pos_encoding(x, self.img_size[1], self.img_size[0])
        x = self.norm_token(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm_token(x)  # B,P,N
        x = x[:, 1:, :]
        # recovery
        if linear:
            x = self.decoder_pred_linear(x)  # B,P,N
            # unpatchify
            x = x.reshape(shape=(
                x.shape[0], self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size, self.patch_size,
                self.patch_size, self.out_chans))
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(x.shape[0], self.out_chans, self.img_size[0], self.img_size[1])  # B,C,H,W
        else:
            x = x.view(-1, self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size,
                       self.embed_dim)  # B,H,W,N
            x = x.permute(0, 3, 1, 2)  # B,N,H,W
            x = self.decoder_pred_conv(x)
            # norm
            x = x.permute(0, 2, 3, 1)  # B,H,W,N
            x = self.norm_image(x)
            x = x.permute(0, 3, 1, 2)  # B,N,H,W
        return x

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)
        if self.needs_fc:
            nn.init.constant_(self.fc1.weight, val=1)
            nn.init.constant_(self.fc1.bias, val=0)
        nn.init.constant_(self.decoder_pred_linear.weight, val=1)
        nn.init.constant_(self.decoder_pred_linear.bias, val=0)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


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
