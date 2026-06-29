"""
Triplet Deep Attention (TDA) Module.
Implements the TDA module from:
  "Transformer-based end-to-end attack on text CAPTCHAs with triplet deep attention"
  (Zhang et al., Computers & Security 146, 2024)

Based on Triplet Attention (Misra et al., WACV 2021) with ECANet-style
adaptive 1D convolution integrated into each branch.
"""
import math
import torch
import torch.nn as nn


def _get_eca_kernel_size(channels):
    """Adaptive kernel size for 1D convolution (ECANet style).
    Eq. (8): psi(c) = |log2(c)/gamma + b/gamma|, gamma=2, b=1
    """
    gamma, b = 2, 1
    k = int(abs(math.log2(channels) / gamma + b / gamma))
    k = k if k % 2 == 1 else k + 1
    k = max(k, 3)
    return k


class ZPool(nn.Module):
    """Z-pool: concatenate max and mean along dim=1, reducing it to 2.
    Eq. (3)-(4).
    """
    def forward(self, x):
        return torch.cat(
            (x.max(dim=1, keepdim=True)[0],
             x.mean(dim=1, keepdim=True)),
            dim=1
        )


class SpatialAttn(nn.Module):
    """7x7 conv spatial attention."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        return self.bn(self.conv(x))


class TDAModule(nn.Module):
    """Triplet Deep Attention Module.
    
    Three parallel branches following Eqs. (1)-(10):
    - Branch 0 (BR_CHW): spatial attention on (C, H, W) with 7x7 conv
    - Branch 1 (BR_WHC): Rot02 (swap C<->W) -> Z-pool -> 7x7 conv -> sigmoid -> 
                          ECA 1D conv -> Rot02_back
    - Branch 2 (BR_HCW): Rot01 (swap C<->H) -> Z-pool -> 7x7 conv -> sigmoid ->
                          ECA 1D conv -> Rot01_back
    
    Output = average of three branches (Eq. 10).
    """
    def __init__(self, channels):
        super().__init__()
        k = _get_eca_kernel_size(channels)
        
        # Branch 0: spatial attention (C,H,W)
        self.zpool0 = ZPool()
        self.attn0 = SpatialAttn()
        
        # Branch 1: cross-dim after Rot02 (swap C and W)
        self.zpool1 = ZPool()
        self.attn1 = SpatialAttn()
        self.eca1 = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        
        # Branch 2: cross-dim after Rot01 (swap C and H)
        self.zpool2 = ZPool()
        self.attn2 = SpatialAttn()
        self.eca2 = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        
        # ---- Branch 0: spatial attention on (B, C, H, W) ----
        x0 = self.zpool0(x)                    # (B, 2, H, W)
        x0 = self.sigmoid(self.attn0(x0))      # (B, 1, H, W)
        y0 = x * x0                            # (B, C, H, W)
        
        # ---- Branch 1: Rot02 swap C<->W ----
        x1 = x.permute(0, 3, 2, 1).contiguous()  # (B, W, H, C)
        x1_zp = self.zpool1(x1)                   # (B, 2, H, C)
        x1_attn = self.sigmoid(self.attn1(x1_zp)) # (B, 1, H, C)
        x1 = x1 * x1_attn                         # (B, W, H, C)
        # ECA on channel dim C
        gap1 = x1.mean(dim=[1, 2]).unsqueeze(1)    # (B, 1, C)
        eca1 = self.sigmoid(self.eca1(gap1))       # (B, 1, C)
        x1 = x1 * eca1.unsqueeze(2)               # (B, W, H, C)
        y1 = x1.permute(0, 3, 2, 1).contiguous()  # (B, C, H, W)
        
        # ---- Branch 2: Rot01 swap C<->H ----
        x2 = x.permute(0, 2, 1, 3).contiguous()   # (B, H, C, W)
        x2_zp = self.zpool2(x2)                    # (B, 2, C, W)
        x2_attn = self.sigmoid(self.attn2(x2_zp))  # (B, 1, C, W)
        x2 = x2 * x2_attn                          # (B, H, C, W)
        # ECA on channel dim C
        gap2 = x2.mean(dim=[1, 3]).unsqueeze(1)     # (B, 1, C)
        eca2 = self.sigmoid(self.eca2(gap2))        # (B, 1, C)
        x2 = x2 * eca2.unsqueeze(3)                # (B, H, C, W)
        y2 = x2.permute(0, 2, 1, 3).contiguous()   # (B, C, H, W)
        
        # ---- Average (Eq. 10) ----
        return (y0 + y1 + y2) / 3.0
