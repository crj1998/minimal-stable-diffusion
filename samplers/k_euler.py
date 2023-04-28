from samplers.sampler import Sampler


class KEulerSampler(Sampler):
    def __init__(self, n_inference_steps=50, n_training_steps=1000):
        super().__init__(n_inference_steps, n_training_steps)


    def step(self, latents, output):
        sigma_from = self.sigmas[self.step_count]
        sigma_to   = self.sigmas[self.step_count + 1]
        self.step_count += 1

        return latents + output * (sigma_to - sigma_from)