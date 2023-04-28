import numpy as np

from samplers.sampler import Sampler

class KLMSSampler(Sampler):
    def __init__(self, n_inference_steps=50, n_training_steps=1000, lms_order=4):
        super().__init__(n_inference_steps, n_training_steps)

        self.lms_order = lms_order
        self.outputs = []

    def step(self, latents, output):
        t = self.step_count
        self.outputs = [output] + self.outputs[:self.lms_order - 1]
        order = len(self.outputs)
        for i, output in enumerate(self.outputs):
            # Integrate polynomial by trapezoidal approx. method for 81 points.
            x = np.linspace(self.sigmas[t], self.sigmas[t + 1], 81)
            y = np.ones(81)
            for j in range(order):
                if i == j: continue
                y *= x - self.sigmas[t - j]
                y /= self.sigmas[t - i] - self.sigmas[t - j]
            lms_coeff = np.trapz(y=y, x=x)
            latents += lms_coeff * output

        self.step_count += 1
        return latents