import torch
import torch.nn as nn
import torch.nn.functional as F


class DCN(nn.Module):
    """
    Pure PyTorch Deformable Convolution
    No mmcv
    No torchvision deform_conv2d
    """

    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        p=None,
        bias=False
    ):
        super().__init__()

        if p is None:
            p = k // 2

        self.k = k
        self.s = s
        self.p = p
        self.c1 = c1
        self.c2 = c2

        # 学习offset
        self.offset_conv = nn.Conv2d(
            c1,
            2 * k * k,
            kernel_size=k,
            stride=s,
            padding=p
        )

        # 卷积权重
        self.weight = nn.Parameter(
            torch.randn(c2, c1, k, k)
        )

        self.bias = (
            nn.Parameter(torch.zeros(c2))
            if bias else None
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        nn.init.kaiming_normal_(
            self.weight,
            mode='fan_out',
            nonlinearity='relu'
        )

    def forward(self, x):

        B, C, H, W = x.shape

        offset = self.offset_conv(x)

        Hout = offset.shape[2]
        Wout = offset.shape[3]

        device = x.device

        # output coordinate
        yy, xx = torch.meshgrid(
            torch.arange(Hout, device=device),
            torch.arange(Wout, device=device),
            indexing='ij'
        )

        yy = yy.float() * self.s
        xx = xx.float() * self.s

        output = []

        idx = 0

        for ky in range(self.k):
            for kx in range(self.k):

                off_y = offset[:, idx * 2 + 0]
                off_x = offset[:, idx * 2 + 1]

                idx += 1

                sample_y = yy + ky - self.p + off_y
                sample_x = xx + kx - self.p + off_x

                sample_y = 2.0 * sample_y / max(H - 1, 1) - 1.0
                sample_x = 2.0 * sample_x / max(W - 1, 1) - 1.0

                grid = torch.stack(
                    (sample_x, sample_y),
                    dim=-1
                )

                sampled = F.grid_sample(
                    x,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )

                output.append(sampled)

        sampled_feat = torch.stack(
            output,
            dim=2
        )

        sampled_feat = sampled_feat.view(
            B,
            C * self.k * self.k,
            Hout,
            Wout
        )

        weight = self.weight.view(
            self.c2,
            -1,
            1,
            1
        )

        out = F.conv2d(
            sampled_feat,
            weight,
            bias=self.bias
        )

        out = self.bn(out)

        out = self.act(out)

        return out

class DCNConv(nn.Module):

    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        p=None
    ):
        super().__init__()

        self.conv = DCN(
            c1,
            c2,
            k=k,
            s=s,
            p=p
        )

    def forward(self, x):
        return self.conv(x)