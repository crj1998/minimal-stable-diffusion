
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import SelfAttention

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gn = nn.GroupNorm(32, channels)
        self.attn = SelfAttention(1, channels)
    
    def forward(self, x):
        identity = x
    
        x = self.gn(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attn(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return x + identity

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gn1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.gn2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.silu = nn.SiLU()
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x

        x = self.gn1(x)
        x = self.silu(x)
        x = self.conv1(x)

        x = self.gn2(x)
        x = self.silu(x)
        x = self.conv2(x)

        return x + self.residual(identity)


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),

            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),

            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),

            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            ResidualBlock(512, 512),

            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, chunks=2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise

        x = x * 0.18215
        return x

class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x / 0.18215
        for module in self:
            x = module(x)
        return x