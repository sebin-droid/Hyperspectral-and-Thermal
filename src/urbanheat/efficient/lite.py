"""Lightweight student: Member 2's HeatCNN with narrower branches and depthwise-separable residual blocks."""
import json
from pathlib import Path

import torch
from torch import nn

from urbanheat.models import cnn


class SepBlock(nn.Module):
    """cnn.ResBlock with each 3x3 conv split into depthwise 3x3 + pointwise 1x1 (same receptive field)."""

    def __init__(self, c, dilation):
        super().__init__()
        sep = lambda: [nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation, groups=c, bias=False),
                       nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c)]
        self.body = nn.Sequential(*sep(), nn.ReLU(inplace=True), *sep())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


def lite_cnn(variant="fusion_cnn", width=32, thermal_width=16, separable=True, hs_proj=None):
    """HeatCNN(variant) at the given widths; separable=True swaps every ResBlock for a SepBlock."""
    m = cnn.HeatCNN(variant, width=width, thermal_width=thermal_width, hs_proj=hs_proj)
    if separable:
        for name in ("surface", "thermal"):
            if hasattr(m, name):
                branch = getattr(m, name)
                dilations = [blk.body[0].dilation[0] for blk in branch.blocks]
                branch.blocks = nn.Sequential(*[SepBlock(branch.stem[0].out_channels, d) for d in dilations])
    return m


def load(run_dir, device="cpu"):
    """Model of a run folder: a student saved by efficient.distill.run, or one of Member 2's HeatCNN runs."""
    d = Path(run_dir)
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    sd = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    # Member 2's configs may lack "variant"; the run name holds it (longest match: fusion_pca_cnn before fusion_cnn)
    variant = cfg.get("variant") or next(v for v in sorted(cnn.VARIANTS, key=len, reverse=True) if v in d.name)
    if cfg.get("lite"):
        m = lite_cnn(variant, cfg["width"], cfg["thermal_width"], cfg["separable"], hs_proj=sd.get("hs_proj"))
    else:
        m = cnn.HeatCNN(variant, hs_proj=sd.get("hs_proj"))
    m.load_state_dict(sd)
    return m.to(device).eval()
