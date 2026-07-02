import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv  # 使用官方 Conv

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Depthwise conv: groups=in_channels，输出必须等于 in_channels
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU()
        # Pointwise conv: 用 1×1 映射到 out_channels
        self.pwconv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pwconv(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c * e)
        self.cv1 = Conv(c, c_, 1, 1)
        self.cv2 = DSConv(c_, c_)  # 输入输出通道相等
        self.shortcut = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


class C2f_DSConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((n + 2) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))