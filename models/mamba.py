import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import repeat
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 3
    # d_conv: int = 4
    # n_heads: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        # self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments
        self.d_inner = int(self.expand_factor * self.d_model)  

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        # x : (B, L, D)
        # output : (B, L, D)
        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class FrequencyModeling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fft_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 1)
        )

        self.dwt = DWTForward(J=1, wave='haar')
        self.idwt = DWTInverse(wave='haar')
        self.wave_conv = nn.Conv2d(d_model, d_model, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # === Fourier Transform ===
        x_fft = torch.fft.rfft2(x, norm='ortho')
        mag = self.fft_conv(torch.abs(x_fft))
        pha = torch.angle(x_fft)
        x_fourier = torch.fft.irfft2(mag * torch.exp(1j*pha), s=(H, W), norm='ortho')
        
        # === Wavelet Transform ===
        Yl, Yh = self.dwt(x)

        if Yh:
            Yh0 = Yh[0]
            processed_subbands = []
            for i in range(Yh0.size(2)):
                subband = Yh0[:, :, i, :, :]
                conv_sub = self.wave_conv(subband)
                processed_subbands.append(conv_sub.unsqueeze(2))

            Yh = [torch.cat(processed_subbands, dim=2)]

        x_wavelet = self.idwt((Yl[:, :, :H//2, :W//2], Yh))

        return x + x_fourier + x_wavelet

class LightweightSymmetryEnhancement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.symmetry_weight = nn.Parameter(torch.tensor(0.3))  
        
    def forward(self, x):
        x_flipped = torch.flip(x, [-1])
        
        symmetric_feat = (x + x_flipped) / 2
        
        weight = torch.sigmoid(self.symmetry_weight)  
        enhanced = x * (1 - weight) + symmetric_feat * weight
        
        return enhanced

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        self.conv2d = nn.Conv2d(config.d_inner, config.d_inner, config.d_conv, 
                               padding=(config.d_conv-1)//2, groups=config.d_inner, 
                               bias=config.conv_bias)
        self.freq_model = FrequencyModeling(config.d_inner)
        
        self.symmetry_enhance = LightweightSymmetryEnhancement(config.d_inner)
        
        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(config.d_inner, config.dt_rank + 2*config.d_state, bias=False).weight 
            for _ in range(4)], dim=0))
        
        self.dt_projs_weight = nn.Parameter(torch.stack([
            self.dt_init(config.dt_rank, config.d_inner).weight for _ in range(4)], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([
            self.dt_init(config.dt_rank, config.d_inner).bias for _ in range(4)], dim=0))
        
        self.A_logs = self.A_log_init(config.d_state, config.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(config.d_inner, copies=4, merge=True)
        
        self.selective_scan = selective_scan_fn
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.out_norm = nn.LayerNorm(config.d_inner)

    def forward(self, x):
        B, L, D = x.shape
        H = W = int(L ** 0.5)

        x_2d = x.view(B, H, W, D)
        xz = self.in_proj(x_2d)
        x, z = xz.chunk(2, dim=-1)

        x_freq = self.freq_model(x.permute(0, 3, 1, 2))
        x = x + x_freq.permute(0, 2, 3, 1)

        x = x.permute(0, 3, 1, 2)
        x = F.silu(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4

        y_4d = y.view(B, self.config.d_inner, H, W)

        y_enhanced_4d = self.symmetry_enhance(y_4d)

        y_enhanced = y_enhanced_4d.view(B, self.config.d_inner, L)

        y_final = y + 0.2 * y_enhanced  

        y_final = y_final.transpose(1, 2).contiguous().view(B, H, W, -1)

        y_final = self.out_norm(y_final)
        y_final = y_final * F.silu(z)
        output = self.out_proj(y_final)

        return output.view(B, L, D)

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
    
        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
    
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=2)
    
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
    
        out_y = self.selective_scan(
            xs.view(B, -1, L), dts.contiguous().view(B, -1, L),
            -torch.exp(self.A_logs.float()).view(-1, self.config.d_state),
            Bs.view(B, K, -1, L),
            Cs.view(B, K, -1, L),
            self.Ds.float().view(-1),
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True
        ).view(B, K, -1, L)
    
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj
    
    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output
    