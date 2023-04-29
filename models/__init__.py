from models.vae import Encoder, Decoder
from models.unet import UNet
from models.diffuser import Diffuser
# from models.clip2 import CLIP, Transformer, VisionTransformer
from models.clip import LanguageTransformer, VisionTransformer, CLIP

MODELS = {
    "clip": CLIP,
    # "clip.transformer": Transformer,
    # "clip.visual": VisionTransformer,
    "vae.encoder": Encoder,
    "vae.decoder": Decoder,
    "unet": UNet,
    "diffuser": Diffuser,
    "clip.language": LanguageTransformer,
    "clip.visual": VisionTransformer
}

KWARGS = {
    "clip": dict( embed_dim = 768,  image_resolution = 224, 
        vision_layers = 24, vision_width = 1024, vision_patch_size = 14, 
        context_length = 77, vocab_size = 49408, transformer_width = 768,
        transformer_heads = 12, transformer_layers = 12 ),
    # "clip.visual": dict( input_resolution=224, patch_size=14, width=1024, layers=24, heads=16, output_dim=768),
    # "clip.transformer": dict(width=768, layers=12, heads=12, attn_mask=None),
    "vae.encoder": dict(),
    "vae.decoder": dict(),
    "unet": dict(),
    "diffuser": dict(),
    "clip.language": dict(context_length = 77, vocab_size = 49408, width = 768, heads = 12, layers = 12),
    "clip.visual": dict(input_resolution=224, patch_size=14, width=1024, layers=24, heads=16, output_dim=768)
}

WEIGHTS = {
    "vae.encoder": "/home/crj1998/workspace/minimal-stable-diffusion/weights/encoder.pt",
    "vae.decoder": "/home/crj1998/workspace/minimal-stable-diffusion/weights/decoder.pt",
    # "unet": "/home/crj1998/workspace/minimal-stable-diffusion/weights/encoder.pt",
    "diffuser": "/home/crj1998/workspace/minimal-stable-diffusion/weights/diffuser.pt",
    "clip": "/home/crj1998/workspace/minimal-stable-diffusion/weights/clip.pt",
    "clip.language": "/home/crj1998/workspace/minimal-stable-diffusion/weights/clip.language.pt",
    "clip.visual": "/home/crj1998/workspace/minimal-stable-diffusion/weights/clip.visual.pt"
}

def builder(name, device=None, pretrained=True):
    model = MODELS[name](**KWARGS[name])
    if pretrained:
        import torch
        if name == "clip":
            model.visual.load_state_dict(torch.load(WEIGHTS["clip.visual"]))
            model.language.load_state_dict(torch.load(WEIGHTS["clip.language"]))
            model.load_state_dict(torch.load(WEIGHTS["clip"]), strict=False)
        else:
            model.load_state_dict(torch.load(WEIGHTS[name]))
            print(f"{name} load weights from {WEIGHTS[name]}.")
    model.to(device)
    return model

def available_models():
    return list(MODELS.keys())