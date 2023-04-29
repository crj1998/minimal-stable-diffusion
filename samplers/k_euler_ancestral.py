
import torch
from samplers.sampler import Sampler


class KEulerAncestralSampler(Sampler):
    def __init__(self, n_inference_steps=50, n_training_steps=1000, generator=None, **kwargs):
        super().__init__(n_inference_steps, n_training_steps)
        self.generator = generator

    def step(self, latents, output):
        sigma_from = self.sigmas[self.step_count]
        sigma_to   = self.sigmas[self.step_count + 1]
        sigma_up = sigma_to * (1 - (sigma_to ** 2 / sigma_from ** 2)) ** 0.5
        sigma_down = sigma_to ** 2 / sigma_from
        latents += output * (sigma_down - sigma_from)
        noise = torch.randn( latents.shape, generator=self.generator, device=latents.device)
        self.step_count += 1
        return latents + noise * sigma_up