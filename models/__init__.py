from models.vae import Encoder, Decoder
from models.unet import UNet
from models.diffuser import Diffuser
from models.clip import CLIP, Transformer, VisionTransformer

MODELS = {
    "clip": CLIP,
    "clip.transformer": Transformer,
    "clip.visual": VisionTransformer,
    "vae.encoder": Encoder,
    "vae.decoder": Decoder,
    "unet": UNet,
    "diffuser": Diffuser
}

KWARGS = {
    "clip": dict( embed_dim = 768,  image_resolution = 224, 
        vision_layers = 24, vision_width = 1024, vision_patch_size = 14, 
        context_length = 77, vocab_size = 49408, transformer_width = 768,
        transformer_heads = 12, transformer_layers = 12 ),
    "clip.visual": dict( input_resolution=224, patch_size=14, width=1024, layers=24, heads=16, output_dim=768),
    "clip.transformer": dict(width=768, layers=12, heads=12, attn_mask=None),
    "vae.encoder": dict(),
    "vae.decoder": dict(),
    "unet": dict(),
    "diffuser": dict()
}

WEIGHTS = {}

def builder(name, device=None, pretrained=False):
    model = MODELS[name](**KWARGS[name])
    if pretrained:
        import torch
        model.load_state_dict(torch.load(WEIGHTS[name]))
    model.to(device)
    return model

def available_models():
    return list(MODELS.keys())