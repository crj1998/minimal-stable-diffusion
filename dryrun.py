import torch
from torchinfo import summary
from models import builder as model_builder
from samplers import builder as sample_builder

sampler = sample_builder('k_euler')

device = torch.device("cuda")

# encoder = model_builder("vae.encoder", device)
# print(f"VAE Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f} M")
# x = torch.randn(2, 3, 512, 512)
# n = torch.randn(2, 4, 64, 64)
# summary(
#     model = encoder,
#     input_data = [x, n],
#     depth = 3
# )

# decoder = model_builder("vae.decoder", device)
# print(f"VAE Decoder: {sum(p.numel() for p in decoder.parameters())/1e6:.1f} M")

# x = torch.randn(2, 4, 64, 64)
# summary(
#     model = decoder,
#     input_data = x,
#     depth = 3
# )


# unet = model_builder("unet", device)
# print(f"UNet: {sum(p.numel() for p in unet.parameters())/1e6:.1f} M")

# x = torch.randn(1, 4, 64, 64)
# c = torch.randn(1, 77, 768)
# t = torch.randn(1, 1280)
# summary(
#     model = unet,
#     input_data = (x, c, t),
#     depth = 3
# )


# clip_visual = model_builder("clip.visual", device)
# print(f"CLIP visual: {sum(p.numel() for p in clip_visual.parameters())/1e6:.1f} M")

# x = torch.randn(1, 3, 224, 224).to(device)
# summary(
#     model = clip_visual,
#     input_data = x,
#     depth = 3
# )

# clip_transformer = model_builder("clip.transformer", device)
# print(f"CLIP Transformer: {sum(p.numel() for p in clip_transformer.parameters())/1e6:.1f} M")

# x = torch.randn(77, 1, 768).to(device)
# summary(
#     model = clip_transformer,
#     input_data = x,
#     depth = 3
# )

# diffuser = model_builder("diffuser", device)
# print(f"Diffuser: {sum(p.numel() for p in diffuser.parameters())/1e6:.1f} M")

# l = torch.randn(1, 4, 64, 64).to(device)
# c = torch.randn(1, 77, 768).to(device)
# t = torch.randn(1, 320).to(device)

# summary(
#     model = diffuser,
#     input_data = (l, c, t),
#     depth = 3
# )
