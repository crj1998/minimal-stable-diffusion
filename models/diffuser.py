import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UNet

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super(TimeEmbedding, self).__init__()
        self.linear1 = nn.Linear(dim, 4 * dim)
        self.linear2 = nn.Linear(4 * dim, 4 * dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)
        return x

class FinalLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(FinalLayer, self).__init__()
        self.gn = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.gn(x)
        x = self.silu(x)
        x = self.conv(x)
        return x

class Diffuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
        self.time_embedding = TimeEmbedding(dim=320)
        self.final = FinalLayer(320, 4)

    def forward(self, latent, context, timestep):
        time_embed = self.time_embedding(timestep)
        output = self.unet(latent, context, time_embed)
        output = self.final(output)
        return output