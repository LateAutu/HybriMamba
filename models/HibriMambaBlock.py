import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import trunc_normal_
from timm.layers import trunc_normal_
import numpy as np
import math
from models.mamba import *
# from mamba import *
torch.manual_seed(123)

class DWConv(nn.Module):
    def __init__(self, c_in, c_out, stride=1, trans=False):
        super().__init__()
        if trans:   
            self.op = nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, 1, 1, groups=c_in, bias=False),  # depthwise
                nn.Conv2d(c_in, c_out*4, 1, bias=False),                  # pointwise
                nn.PixelShuffle(2)                                       # 2× upsample
            )
        else:       
            self.op = nn.Sequential(
                nn.Conv2d(c_in, c_in, 4, stride, 1, groups=c_in, bias=False), # depthwise
                nn.Conv2d(c_in, c_out, 1, bias=False)                     # pointwise
            )
    def forward(self, x):
        return self.op(x)

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = DWConv(in_channel, out_channel, stride=2) 
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops_dw = H * W * self.in_channel * (4 * 4)  
        flops_pw = (H // 2) * (W // 2) * self.in_channel * self.out_channel
        total_flops = flops_dw + flops_pw
        print("Downsample FLOPs: {:.2f} G".format(total_flops / 1e9))
        return total_flops


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = DWConv(in_channel, out_channel, trans=True) 
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops_dw = H * W * self.in_channel * (3 * 3) 
        flops_pw = (H * 2) * (W * 2) * self.in_channel * (self.out_channel * 4)
        total_flops = flops_dw + flops_pw
        print("Upsample FLOPs: {:.2f} G".format(total_flops / 1e9))
        return total_flops

class Attention(nn.Module):
    """ Attention module that can take tensor with [B, N, C] or [B, C, H, W] as input. """
    def __init__(self, dim, num_heads=16, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        shape = x.shape
        B, N, C = shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class QKV(nn.Module):
    """ QKV multiplication in Attention module. """
    def __init__(self, dim, num_heads=16, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = x.unbind(0)
        B, _, _, _ = q.shape
        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """ feed-forward network(MLP) """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)

class LEM(nn.Module):
    """Local Enhancement Module"""
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.expand = nn.Conv2d(dim, dim * 2, 1, bias=True)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),  # 3×3 dw
            ChannelAttention(dim, squeeze_factor=16)  # ← CAB
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )
        self.smooth = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x4d = x.transpose(1, 2).view(B, C, H, W)
        x1, x2 = self.expand(x4d).chunk(2, dim=1)   # (B,C,H,W)
        x1 = self.ca(x1)                            
        x2 = self.gate(x2.flatten(2).transpose(1, 2)).view(B, C, H, W)  
        out = self.smooth(x1 * x2)                  # Hadamard
        return out.flatten(2).transpose(1, 2)       # (B, L, C)


class PatchEmbed(nn.Module):
    """ feature to two Embedding """
    def __init__(self, embed_dim=32, lamb=4, norm_layer=None):
        super().__init__()
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = NotImplemented
        self.conv_corse = nn.Conv2d(embed_dim,embed_dim, lamb, lamb)

    def forward(self, x):
        x_corse = self.conv_corse(x)
        x_fine = x
        x_corse = x_corse.flatten(2).transpose(1, 2)
        x_fine = x_fine.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x_corse = self.norm(x_corse)
            x_fine = self.norm(x_fine)
        return x_corse, x_fine


class HibriMamba(nn.Module):
    def __init__(self, dim, depth, config=None, mlp_ratio = 2, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Mamba(config)),
                # PreNorm(dim, FeedForward(dim, dim * mlp_ratio, dropout = dropout))  # MLP
                PreNorm(dim, LEM(dim, mlp_ratio))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class HibriMambaBlock(nn.Module):
    """ main architecture """
    def __init__(self, embed_dim=32, mlp_ratio=4, drop_rate=0.,
                 token_projection='linear'):
        super().__init__()

        self.mlp_ratio = mlp_ratio
        self.dropout = drop_rate
        self.token_projection = token_projection

        # Input/Output
        self.shallow_conv = nn.Conv2d(3, embed_dim, 3, 1, 1)
        self.shallow_act = nn.LeakyReLU(inplace=True)
        # self.out = nn.Conv2d(64, 3, 3, 1, 1)
        self.out = nn.Conv2d(2 * embed_dim, 3, 3, 1, 1)
        self.out_act = nn.LeakyReLU(inplace=True)

        cfg = dict(expand_factor=1.5, d_state=8, dt_rank='auto')  

        config_32 = MambaConfig(d_model=32, n_layers=2, **cfg)
        config_64 = MambaConfig(d_model=64, n_layers=2, **cfg)
        config_128 = MambaConfig(d_model=128, n_layers=2, **cfg)
        config_256 = MambaConfig(d_model=256, n_layers=2, **cfg)

        self.encoderlayer_0 = HibriMamba(32, 2, config=config_32)
        self.encoderlayer_1 = HibriMamba(64, 2, config=config_64)
        self.encoderlayer_2 = HibriMamba(128, 2, config=config_128)
        self.encoderlayer_bottom = HibriMamba(128, 2, config=config_128)
        self.decoderlayer_1 = HibriMamba(256, 2, config=config_256)
        self.decoderlayer_2 = HibriMamba(128, 2, config=config_128)
        self.decoderlayer_3 = HibriMamba(64, 2, config=config_64)

        self.pool0 = Downsample(32,64)
        self.pool1 = Downsample(64,128)

        self.sigmoid = nn.Sigmoid()

        self.Upsample1 = Upsample(256, 64)
        self.Upsample2 = Upsample(128, 32)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, _, _, _ = x.shape
        y = self.shallow_conv(x)
        y = self.shallow_act(y)

        y = y.flatten(2).transpose(1, 2)
        trans0 = self.encoderlayer_0(y)
        pool0 = self.pool0(trans0)

        trans1 = self.encoderlayer_1(pool0)
        pool1 = self.pool1(trans1)

        trans2 = self.encoderlayer_2(pool1)

        trans_bottom = self.encoderlayer_bottom(trans2)

        deconv1 = torch.cat([trans_bottom, trans2], -1)
        deconv1 = self.decoderlayer_1(deconv1)
        up1 = self.Upsample1(deconv1)

        deconv2 = torch.cat([up1, trans1], -1)
        deconv2 = self.decoderlayer_2(deconv2)
        up2 = self.Upsample2(deconv2)

        deconv3 = torch.cat([up2, trans0], -1)

        H_out = W_out = int(math.sqrt(deconv3.size(1)))  
        final_output = self.out(deconv3.transpose(1, 2).view(B, 64, H_out, W_out)))
        final_output = self.out_act(final_output)

        return final_output + x

    
if __name__ == "__main__":
    x = torch.randn((2, 3, 128, 128))
    net = HibriMambaBlock(embed_dim=32)
    y= net(x)
    print(y.shape)
