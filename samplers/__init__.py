
from samplers.k_euler import KEulerSampler
from samplers.k_euler_ancestral import KEulerAncestralSampler
from samplers.k_lms import KLMSSampler

SAMPLERS = {
    "k_lms": KLMSSampler,
    "k_euler": KEulerSampler,
    "k_euler_ancestral": KEulerAncestralSampler
}

def builder(sampler, **kwargs):
   assert sampler in SAMPLERS.keys()
   return SAMPLERS[sampler](**kwargs)