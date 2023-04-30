import torch
from torchinfo import summary
from models import builder as model_builder
from samplers import builder as sample_builder

# sampler = sample_builder('k_euler')
# print(sampler.timesteps)
device = torch.device("cuda")



# encoder = model_builder("vae.encoder", device)
# print(f"VAE Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f} M")
# state_dict = torch.load("data/ckpt/encoder.pt")
# new_state_dict = {}
# for k, v in state_dict.items():
#     # print(k, v.shape)
#     k = k.replace("groupnorm_", "gn")
#     k = k.replace("groupnorm", "gn")
#     k = k.replace("conv_", "conv")
#     k = k.replace("attention", "attn")
#     k = k.replace("residual_layer", "residual")
#     new_state_dict[k] = v
# encoder.load_state_dict(new_state_dict)
# torch.save(encoder.state_dict(), "weights/encoder.pt")

# x = torch.randn(2, 3, 512, 512)
# n = torch.randn(2, 4, 64, 64)
# summary(
#     model = encoder,
#     input_data = [x, n],
#     depth = 3
# )

# decoder = model_builder("vae.decoder", device)
# print(f"VAE Decoder: {sum(p.numel() for p in decoder.parameters())/1e6:.1f} M")

# state_dict = torch.load("data/ckpt/decoder.pt")
# new_state_dict = {}
# for k, v in state_dict.items():
#     # print(k, v.shape)
#     k = k.replace("groupnorm_", "gn")
#     k = k.replace("groupnorm", "gn")
#     k = k.replace("conv_", "conv")
#     k = k.replace("attention", "attn")
#     k = k.replace("residual_layer", "residual")
#     new_state_dict[k] = v
# decoder.load_state_dict(new_state_dict)
# torch.save(decoder.state_dict(), "weights/decoder.pt")

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
# state_dict = clip_transformer.state_dict()
# for k, v in state_dict.items():
#     print(k, v.shape)
# print(5*"======")
# state_dict = torch.load("data/ckpt/clip.pt")
# new_state_dict = {}
# for k, v in state_dict.items():
#     print(k, v.shape)
    # k = k.replace("layernorm_", "ln")
#     k = k.replace("groupnorm", "gn")
#     k = k.replace("conv_", "conv")
    # k = k.replace("attention", "attn")
#     k = k.replace("residual_layer", "residual")
#     new_state_dict[k] = v
# clip_transformer.load_state_dict(new_state_dict)
# torch.save(encoder.state_dict(), "weights/encoder.pth")
# x = torch.randn(77, 1, 768).to(device)
# summary(
#     model = clip_transformer,
#     input_data = x,
#     depth = 3
# )

# diffuser = model_builder("diffuser", device)
# print(f"Diffuser: {sum(p.numel() for p in diffuser.parameters())/1e6:.1f} M")

# state_dict = torch.load("data/ckpt/diffusion.pt")
# new_state_dict = {}
# for k, v in state_dict.items():
#     # print(k, v.shape)
#     k = k.replace("groupnorm_", "gn")
#     k = k.replace("groupnorm", "gn")
#     k = k.replace("conv_", "conv")
#     k = k.replace("attention", "attn")
#     k = k.replace("residual_layer", "residual")
#     new_state_dict[k] = v
# diffuser.load_state_dict(state_dict)
# torch.save(diffuser.state_dict(), "weights/diffuser.pt")
# l = torch.randn(1, 4, 64, 64).to(device)
# c = torch.randn(1, 77, 768).to(device)
# t = torch.randn(1, 320).to(device)

# summary(
#     model = diffuser,
#     input_data = (l, c, t),
#     depth = 3
# )

# model = torch.jit.load("ViT-L-14.pt")
# state_dict = model.language.state_dict()
# clip_text = model_builder("clip.language", device)
# print(f"CLIP Language Transformer: {sum(p.numel() for p in clip_text.parameters())/1e6:.1f} M")

# state_dict = torch.load("data/ckpt/clip.pt")
# new_state_dict = {}

# mapping = {
#     "embedding.position_value": "positional_embedding",
#     "embedding.token_embedding.weight": "token_embedding.weight"
# }
# for k, v in state_dict.items():
#     new_state_dict[mapping.get(k, k)] = v
# clip_text.load_state_dict(new_state_dict)
# torch.save(clip_text.state_dict(), "weights/clip.language.pt")

model = torch.jit.load("ViT-L-14.pt")
state_dict = model.visual.state_dict()
clip_visual = model_builder("clip.visual", device)
print(f"CLIP Language Transformer: {sum(p.numel() for p in clip_visual.parameters())/1e6:.1f} M")

new_state_dict = {}
mapping = {}

for i in range(24):
    mapping[f"transformer.resblocks.{i}.ln_1.weight"] = f"layers.{i}.layernorm_1.weight"
    mapping[f"transformer.resblocks.{i}.ln_1.bias"] = f"layers.{i}.layernorm_1.bias"
    mapping[f"transformer.resblocks.{i}.ln_2.weight"] = f"layers.{i}.layernorm_2.weight"
    mapping[f"transformer.resblocks.{i}.ln_2.bias"] = f"layers.{i}.layernorm_2.bias"
    mapping[f"transformer.resblocks.{i}.attn.in_proj_weight"] = f"layers.{i}.attention.in_proj.weight"
    mapping[f"transformer.resblocks.{i}.attn.in_proj_bias"] = f"layers.{i}.attention.in_proj.bias"
    mapping[f"transformer.resblocks.{i}.attn.out_proj.weight"] = f"layers.{i}.attention.out_proj.weight"
    mapping[f"transformer.resblocks.{i}.attn.out_proj.bias"] = f"layers.{i}.attention.out_proj.bias"
    mapping[f"transformer.resblocks.{i}.mlp.c_fc.weight"] = f"layers.{i}.linear_1.weight"
    mapping[f"transformer.resblocks.{i}.mlp.c_fc.bias"] = f"layers.{i}.linear_1.bias"
    mapping[f"transformer.resblocks.{i}.mlp.c_proj.weight"] = f"layers.{i}.linear_2.weight"
    mapping[f"transformer.resblocks.{i}.mlp.c_proj.bias"] = f"layers.{i}.linear_2.bias"

for k, v in state_dict.items():
    new_state_dict[mapping.get(k, k)] = v
clip_visual.load_state_dict(new_state_dict)
torch.save(clip_visual.state_dict(), "weights/clip.visual.pt")

# model = torch.jit.load("ViT-L-14.pt")
# state_dict = model.state_dict()
# for name in list(state_dict.keys()):
#     if name not in ['text_projection', 'logit_scale']:
#         del state_dict[name]
# print(type(state_dict), state_dict)
# # print(list(state_dict.keys()))
# torch.save(state_dict, "weights/clip.pt")