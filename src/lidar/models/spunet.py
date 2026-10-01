"""Sparse 3D U-Net (MinkUNet-style) on spconv."""
import torch
from torch import nn

import spconv.pytorch as spconv


def _bn_relu(c):
    return [nn.BatchNorm1d(c, momentum=0.02), nn.ReLU(inplace=True)]


class BasicBlock(spconv.SparseModule):
    def __init__(self, cin, cout, key):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(cin, cout, 3, padding=1, bias=False, indice_key=key)
        self.bn1 = nn.BatchNorm1d(cout, momentum=0.02)
        self.conv2 = spconv.SubMConv3d(cout, cout, 3, padding=1, bias=False, indice_key=key)
        self.bn2 = nn.BatchNorm1d(cout, momentum=0.02)
        self.proj = nn.Identity() if cin == cout else nn.Sequential(nn.Linear(cin, cout, bias=False), nn.BatchNorm1d(cout, momentum=0.02))

    def forward(self, x):
        identity = self.proj(x.features)
        out = self.conv1(x)
        out = out.replace_feature(torch.relu(self.bn1(out.features)))
        out = self.conv2(out)
        return out.replace_feature(torch.relu(self.bn2(out.features) + identity))


class SpUNet(nn.Module):
    def __init__(self, in_channels=5, num_classes=19, base=32,
                 enc=(32, 64, 128, 256), dec=(256, 128, 96, 96),
                 enc_layers=(2, 2, 2, 2), dec_layers=(2, 2, 2, 2)):
        super().__init__()
        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base, 5, padding=2, bias=False, indice_key="stem"), *_bn_relu(base))
        self.down, self.enc = nn.ModuleList(), nn.ModuleList()
        skip_ch, prev = [], base
        for s, (c, n) in enumerate(zip(enc, enc_layers)):
            skip_ch.append(prev)
            self.down.append(spconv.SparseSequential(
                spconv.SparseConv3d(prev, c, 2, stride=2, bias=False, indice_key=f"down{s}"), *_bn_relu(c)))
            self.enc.append(spconv.SparseSequential(*[BasicBlock(c, c, f"subm{s + 1}") for _ in range(n)]))
            prev = c
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for i, (c, n) in enumerate(zip(dec, dec_layers)):
            s = len(enc) - 1 - i
            self.up.append(spconv.SparseSequential(
                spconv.SparseInverseConv3d(prev, c, 2, bias=False, indice_key=f"down{s}"), *_bn_relu(c)))
            blocks = [BasicBlock(c + skip_ch[s], c, f"subm{s}")] + [BasicBlock(c, c, f"subm{s}") for _ in range(n - 1)]
            self.dec.append(spconv.SparseSequential(*blocks))
            prev = c
        self.head = nn.Linear(prev, num_classes)

    def forward(self, feats, coords, batch_size):
        """feats (N, C) float, coords (N, 4) int32 [batch, x, y, z] -> logits (N, num_classes)."""
        shape = (coords[:, 1:].amax(0) + 1).tolist()
        x = spconv.SparseConvTensor(feats, coords, shape, batch_size)
        x = self.stem(x)
        skips = []
        for down, enc in zip(self.down, self.enc):
            skips.append(x)
            x = enc(down(x))
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = x.replace_feature(torch.cat([x.features, skips.pop().features], 1))
            x = dec(x)
        return self.head(x.features)


def build_model(width=1.0, in_channels=5, num_classes=19):
    """SpUNet with every channel count scaled by `width` (kept to multiples of 8 for fp16 kernels)."""
    c = lambda n: max(8, int(round(n * width / 8)) * 8)
    return SpUNet(in_channels, num_classes, base=c(32), enc=tuple(c(n) for n in (32, 64, 128, 256)),
                  dec=tuple(c(n) for n in (256, 128, 96, 96)))
