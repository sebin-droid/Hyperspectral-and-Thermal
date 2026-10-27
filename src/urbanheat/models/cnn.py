"""Fully convolutional heat models: per-pixel LST anomaly from image patches (or whole scenes).

Branches as in the project design (slides 10-11): a surface / spectral CNN, a thermal CNN, and concatenation
fusion of their feature maps. No pooling, so the output keeps the input's size and any scene size works;
dilated residual blocks give a receptive field of 29 px (~1.7 km at 60 m).
"""
import torch
from torch import nn

from urbanheat.contracts import SENSORS

# which inputs feed each branch, per model variant
VARIANTS = {
    "landsat_cnn": dict(surface=["ls", "env"], thermal=False),
    "spectral_cnn": dict(surface=["hs", "ls", "env"], thermal=False),
    "thermal_cnn": dict(surface=[], thermal=True),
    "fusion_cnn": dict(surface=["hs", "ls", "env"], thermal=True),
}
CHANNELS = {"hs": 122, "ls": 9, "env": 3, "thermal": 1}


class ResBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(c))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class Branch(nn.Module):
    """1x1 'spectral' reduction, then dilated residual blocks."""

    def __init__(self, c_in, width, dilations=(1, 2, 4)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(c_in, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock(width, d) for d in dilations])

    def forward(self, x):
        return self.blocks(self.stem(x))


class HeatCNN(nn.Module):
    def __init__(self, variant="fusion_cnn", width=64, thermal_width=32, sensor_dim=8, channels=CHANNELS):
        super().__init__()
        v = VARIANTS[variant]
        self.variant, self.surface_keys, self.use_thermal = variant, v["surface"], v["thermal"]
        feats = sensor_dim
        if self.surface_keys:
            self.surface = Branch(sum(channels[k] for k in self.surface_keys), width)
            feats += width
        if self.use_thermal:
            self.thermal = Branch(channels["thermal"], thermal_width)
            feats += thermal_width
        self.sensor = nn.Embedding(len(SENSORS), sensor_dim)          # learns per-sensor calibration offsets
        self.head = nn.Sequential(nn.Conv2d(feats, 64, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))

    def forward(self, batch):
        any_x = batch["ls"] if "ls" in batch else batch["thermal"]
        b, _, h, w = any_x.shape
        parts = []
        if self.surface_keys:
            parts.append(self.surface(torch.cat([batch[k] for k in self.surface_keys], 1)))
        if self.use_thermal:
            parts.append(self.thermal(batch["thermal"]))
        parts.append(self.sensor(batch["sensor"])[:, :, None, None].expand(-1, -1, h, w))
        return self.head(torch.cat(parts, 1))[:, 0]                    # concatenation fusion -> (B, H, W)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, batch):
    """Multiply-adds of all convolutions for one forward pass of `batch` (x2 = FLOPs)."""
    macs = []

    def hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups
        macs.append(out.numel() * k)

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(batch)
    for h in hs:
        h.remove()
    n = next(iter(batch.values())).shape[0]
    return 2 * sum(macs) // n
