from typing import List, Union, Tuple

import math

from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import SelfAttention
from tokenizer import Tokenizer

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class TransformerLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int, attn_mask: bool=False):
        super().__init__()
        self.attn_mask = attn_mask
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)
        self.gelu = QuickGELU()

    def forward(self, x):
        identity = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=self.attn_mask)
        x = x + identity

        identity = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = self.gelu(x)   # QuickGELU activation function
        x = self.linear_2(x)
        x = x + identity

        return x

class LanguageTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 768,
        heads: int = 12,
        layers: int = 12
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.zeros((context_length, width)))
        self.layers = nn.Sequential(*[
            TransformerLayer(heads, width, True) for _ in range(layers)  # attn mask used
        ])
        self.layernorm = nn.LayerNorm(width)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:        
        state = self.token_embedding(tokens) + self.positional_embedding
        state = self.layers(state)
        output = self.layernorm(state)
        return output


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.layers = nn.Sequential(*[
            TransformerLayer(heads, width, False) for _ in range(layers)  # attn mask unused
        ])

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding

        x = self.ln_pre(x)
        # print(x.mean())
        x = self.layers(x)
        # print(x[:, :2, :])
        # print(x.mean())
        x = self.ln_post(x[:, 0, :])
        
        x = x @ self.proj
        
        return x
    

class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        # vision
        image_resolution: int = 224,
        vision_layers: int = 24,
        vision_width: int = 1024,
        vision_patch_size: int = 14,
        # text
        context_length: int = 77,
        vocab_size: int = 49408,
        transformer_width: int = 768,
        transformer_heads: int = 12,
        transformer_layers: int = 12
    ):
        super().__init__()

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim
        )

        self.language = LanguageTransformer(
            context_length=context_length,
            vocab_size=vocab_size,
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, images: torch.FloatTensor) -> torch.FloatTensor:
        image_features = self.visual(images)
        return image_features

    def encode_text(self, texts: torch.LongTensor) -> torch.Tensor:
        text_features = self.language(texts)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_features = text_features[torch.arange(text_features.size(0)), texts.argmax(dim=-1)] @ self.text_projection

        return text_features

    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = torch.exp(self.logit_scale)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    @staticmethod
    def process(n_px=224):
        import torchvision.transforms as T
        # def _convert_image_to_rgb(image):
        #     return image.convert("RGB")
        # RGB format
        return T.Compose([
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(n_px),
            # _convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]), Tokenizer()