from typing import Optional

import torch

from tokenizer import Tokenizer

from models import builder as model_builder
from samplers import builder as sampler_builder

def get_time_embedding(timestep):
    freqs = torch.pow(10000, - torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class StableDiffusionPipeline:
    def __init__(self,
        strength: float = 0.8,
        do_cfg: bool = True,
        cfg_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        sampler: str = "k_lms",
        n_inference_steps: int = 50,
        models: Optional[dict] = None,
        seed: Optional[int] = None
    ):
        self.strength = strength
        self.do_cfg = do_cfg
        self.cfg_scale = cfg_scale
        self.height = height
        self.width = width
        self.n_inference_steps = n_inference_steps

        device = torch.device("cuda")
        self.tokenizer = Tokenizer()
        self.clip_text_encoder = models.get('clip', model_builder('clip.text'))
        self.diffuser = models.get('diffusion', model_builder('diffuser'))
        self.vae_decoder = models.get('decoder', model_builder('vae.decoder'))
        self.sampler = sampler_builder(sampler)
        self.device = device

    def load_pretrained(self):
        return self
    
    def txt2img(self):
        pass

    def img2img(self, prompts, uncond_prompts=None):
        uncond_prompts = uncond_prompts or [""] * len(prompts)
        if self.do_cfg:
            cond_tokens = self.tokenizer.encode_batch(prompts)    # List [1, 77]
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=self.device)    # (1, 77)
            cond_context = self.clip_text_encoder(cond_tokens)    # (1, 77, 768)
            uncond_tokens = self.tokenizer.encode_batch(uncond_prompts)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=self.device)
            uncond_context = self.clip_text_encoder(uncond_tokens)    # (1, 77, 768)
            context = torch.cat([cond_context, uncond_context])    # (2, 77, 768)
        else:
            tokens = self.tokenizer.encode_batch(prompts)
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
            context = self.clip_text_encoder(tokens)

        noise_shape = (len(prompts), 4, self.height // 8, self.width // 8)

        latents = torch.randn(noise_shape, generator=self.generator, device=self.device)    # [1, 4, 64, 64]
        latents *= self.sampler.initial_scale

        for i, t in enumerate(self.sampler.timesteps):

            input_latents = latents * self.sampler.get_input_scale()

            if self.do_cfg:
                input_latents = input_latents.repeat(2, 1, 1, 1)    # [2, 4, 64, 64]

            time_embedding = get_time_embedding(t).to(self.device)    # [1, 320]
            output = self.diffuser(input_latents, context, time_embedding)    # [2, 4, 64, 64]

            if self.do_cfg:
                output_cond, output_uncond = output.chunk(chunks=2)
                output = self.cfg_scale * (output_cond - output_uncond) + output_uncond    # [1, 4, 64, 64]

            latents = self.sampler.step(latents, output)    # [1, 4, 64, 64]

        images = self.decoder(latents)    # [1, 3, 512, 512]

        # post process
