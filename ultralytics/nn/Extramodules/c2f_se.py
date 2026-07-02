import torch
import torch.nn as nn


class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super(SEAttention, self).__init__()
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpooling(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Conv(nn.Module):
    # ✅ 参数名改为 g，与 Bottleneck/C2f 中的调用一致
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, g=1):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)          # 普通卷积，不分组
        self.cv2 = Conv(c_, c2, 3, 1, g=g)     # 分组卷积，g 参数匹配
        self.se = SEAttention(c2)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        out = self.se(self.cv2(self.cv1(x)))
        return x + out if self.shortcut else out


class C2f_SE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((n + 2) * self.c, c2, 1, 1)

        # 每个 Bottleneck 输入输出都是 self.c，使用给定的 shortcut 和 g
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))