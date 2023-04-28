
import numpy as np

def get_alphas_cumprod(n_training_steps=1000, beta_start=0.00085, beta_end=0.0120):
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=np.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas_cumprod

class Sampler:
    def __init__(self, n_inference_steps: int = 50, n_training_steps: int = 1000):
        timesteps = np.linspace(n_training_steps - 1, 0, n_inference_steps)
        alphas_cumprod = get_alphas_cumprod(n_training_steps)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        log_sigmas = np.log(sigmas)
        log_sigmas = np.interp(timesteps, range(n_training_steps), log_sigmas)
        sigmas = np.exp(log_sigmas)
        sigmas = np.append(sigmas, 0)

        self.n_inference_steps = n_inference_steps
        self.n_training_steps = n_training_steps
        self.sigmas = sigmas
        self.initial_scale = sigmas.max()
        self.timesteps = timesteps
        self.step_count = 0

    def reset(self):
        self.step_count = 0

    def set_strength(self, strength=1):
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        timesteps = np.linspace(self.n_training_steps - 1, 0, self.n_inference_steps)
        self.timesteps = timesteps[start_step:]
        self.initial_scale = self.sigmas[start_step]
        self.step_count = start_step

    def get_input_scale(self, step_count=None):
        step_count = step_count or self.step_count
        sigma = self.sigmas[step_count]
        return 1 / (sigma ** 2 + 1) ** 0.5

    def step(self, latents, output):
        raise NotImplementedError