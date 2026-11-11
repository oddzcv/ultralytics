"""
DySample: Light-weight Dynamic Upsampling Operator
Reference: "Learning to Upsample by Learning to Sample" (ICCV 2023)
Adapted for YOLO26 Boundary-Guided Prototype Refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """
    DySample: Dynamic Content-Aware Upsampling module.
    
    Args:
        in_channels (int): Jumlah channel input.
        scale (int): Faktor perbesaran (default: 2 untuk 20x20 -> 40x40).
        groups (int): Jumlah grup sampling offset (default: 4).
        end2end (bool): Kompatibilitas inference.
    """
    def __init__(self, in_channels: int, scale: int = 2, groups: int = 4, end2end: bool = False):
        super().__init__()
        assert in_channels % groups == 0, f"in_channels ({in_channels}) harus habis dibagi groups ({groups})"
        self.scale = scale
        self.groups = groups
        self.end2end = end2end

        # Generator offset: konvolusi 1x1 ringan untuk menghasilkan deviasi koordinat (dx, dy)
        # Output channel: 2 * scale^2 * groups (2 untuk sumbu x dan y)
        out_channels = 2 * (scale ** 2) * groups
        self.offset_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Inisialisasi bobot offset ke 0 agar awal training setara dengan bilinear grid standar
        nn.init.zeros_(self.offset_conv.weight)

    def _init_grid(self, b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Membuat grid koordinat dasar seragam ter-normalisasi [-1, 1]."""
        gh = torch.linspace(-1.0, 1.0, h * self.scale, device=device, dtype=dtype)
        gw = torch.linspace(-1.0, 1.0, w * self.scale, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(gh, gw, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)  # (H*scale, W*scale, 2)
        return grid.unsqueeze(0).repeat(b * self.groups, 1, 1, 1)  # (B*groups, H*scale, W*scale, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out_h, out_w = h * self.scale, w * self.scale

        # 1. Prediksi offset dinamis
        # Shape: (B, 2 * scale^2 * groups, H, W)
        offset = self.offset_conv(x)
        
        # Reshape & rearrange offset ke dimensi target (B*groups, out_h, out_w, 2)
        # Catatan: gunakan reshape (bukan view) karena saat training tensor bisa
        # non-contiguous (mis. memory format channels_last / AMP)
        offset = offset.reshape(b, self.groups, 2, self.scale, self.scale, h, w)
        offset = offset.permute(0, 1, 5, 3, 6, 4, 2).contiguous()
        offset = offset.reshape(b * self.groups, out_h, out_w, 2)

        # Skala offset dinamis agar pergeseran koordinat halus
        offset = offset * (1.0 / max(out_h, out_w))

        # 2. Buat grid dasar dan tambahkan offset
        base_grid = self._init_grid(b, h, w, x.device, x.dtype)
        sampling_grid = (base_grid + offset).clamp(-1.0, 1.0)

        # 3. Reshape input fitur per grup dan lakukan sampling adaptif
        x_grouped = x.contiguous().reshape(b * self.groups, c // self.groups, h, w)
        
        # F.grid_sample melakukan interpolasi bilinear adaptif berdasarkan sampling_grid
        out = F.grid_sample(
            x_grouped,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # 4. Kembalikan ke shape output asli (B, C, H*scale, W*scale)
        return out.reshape(b, c, out_h, out_w)