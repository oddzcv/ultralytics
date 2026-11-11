"""
Boundary-Guided Prototype Refinement Module for YOLO26-Seg (K1 Contribution)
Combines:
  1. Low-level P2 skip connection (40x40 spatial detail)
  2. DySample content-aware upsampling (Liu et al., ICCV 2023)
  3. Boundary Attention Map (B) for gated feature fusion
  4. Boundary prediction head for L_boundary supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.dysample import DySample


class BoundaryProto26(nn.Module):
    """
    Boundary-Guided Prototype Refinement (Proto26-K1).
    
    Args:
        ch (tuple): Channel sizes of input feature maps (P2, P3, P4, P5).
                    Default: (c_p2, c_p3, c_p4, c_p5).
        c_ (int): Intermediate prototype channel dimension (default: 256 for YOLO26, or 64/128 for Nano).
        c2 (int): Number of prototype masks (default: 32).
    """
    def __init__(self, ch: tuple = (64, 128, 256, 512), c_: int = 256, c2: int = 32):
        super().__init__()
        assert len(ch) >= 2, f"BoundaryProto26 membutuhkan minimal 2 feature maps [P2, P3, ...], didapat {len(ch)}"
        
        c_p2 = ch[0]  # Channel fitur P2 (stride 4, 40x40 pada input 160x160)
        c_p3 = ch[1]  # Channel fitur P3 (stride 8, 20x20)
        
        # 1. Penyelarasan multi-skala Neck (P4, P5 -> P3)
        self.feat_refine = nn.ModuleList(Conv(x, c_p3, k=1) for x in ch[2:])
        self.feat_fuse_p3 = Conv(c_p3, c_, k=3)

        # 2. DySample: memperbesar fitur konteks P3 (20x20) ke resolusi P2 (40x40)
        self.dysample = DySample(in_channels=c_, scale=2, groups=4)

        # 3. Penyelaras fitur P2 (Backbone -> Proto space): phi(F2)
        self.phi_p2 = Conv(c_p2, c_, k=1)

        # 4. Boundary Head: menghasilkan Boundary Attention Map B (1 x 40 x 40)
        # Nilai diaktifkan dengan Sigmoid agar berada pada rentang [0, 1]
        self.boundary_conv = nn.Sequential(
            Conv(c_, c_, k=3),
            nn.Conv2d(c_, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 5. Prototype Generation Layers: F_fused -> 32 Prototype Masks
        self.cv_fuse = Conv(c_, c_, k=3)
        self.cv_proto = Conv(c_, c2, k=1)  # 32 prototype masks

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass Boundary-Guided Prototype Refinement.
        
        Args:
            x (list[torch.Tensor]): List feature maps [P2, P3, P4, P5]
                - x[0]: P2 (B, c_p2, 40, 40)
                - x[1]: P3 (B, c_p3, 20, 20)
                - x[2]: P4 (B, c_p4, 10, 10)
                - x[3]: P5 (B, c_p5, 5, 5)
        
        Returns:
            proto (torch.Tensor): Prototype masks P' (B, 32, 40, 40)
            boundary_map (torch.Tensor): Boundary Attention Map B (B, 1, 40, 40) (saat training)
        """
        p2 = x[0]
        p3 = x[1]

        # 1. Fusi konteks level P3 (P3 + P4 + P5)
        feat_p3 = p3
        for i, f in enumerate(self.feat_refine):
            up = f(x[i + 2])
            up = F.interpolate(up, size=p3.shape[-2:], mode="nearest")
            feat_p3 = feat_p3 + up
        
        feat_p3 = self.feat_fuse_p3(feat_p3)  # Shape: (B, c_, 20, 20)

        # 2. Perbesar konteks P3 ke resolusi 40x40 via DySample
        context_40 = self.dysample(feat_p3)  # Shape: (B, c_, 40, 40)

        # 3. Ekstrak fitur detail P2: phi(F2)
        detail_40 = self.phi_p2(p2)          # Shape: (B, c_, 40, 40)

        # 4. Hasilkan Boundary Attention Map B in [0, 1]
        boundary_map = self.boundary_conv(detail_40)  # Shape: (B, 1, 40, 40)

        # 5. Boundary-Guided Fusion:
        # F_fused = DySample(P3) + B * phi(P2)
        # Menghapus noise tekstur daun di tengah dan hanya mempertegas tepi!
        fused = context_40 + (boundary_map * detail_40)

        # 6. Hasilkan 32 Prototype Masks
        proto = self.cv_proto(self.cv_fuse(fused))  # Shape: (B, 32, 40, 40)

        # Kembalikan tuple (proto, boundary_map) saat training untuk perhitungan L_boundary
        if self.training:
            return proto, boundary_map
        return proto