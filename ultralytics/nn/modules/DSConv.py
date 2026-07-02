import torch.nn as nn


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU()

        self.pwconv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.act2(self.bn2(self.pwconv(x)))
        return x
