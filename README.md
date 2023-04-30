# minimal-stable-diffusion
A minimal example for stable diffusion.

Stable Diffusion v1 refers to a specific configuration of the model architecture that uses a downsampling-factor 8 autoencoder with an 860 M UNet and CLIP ViT-L/14 text encoder for the diffusion model. The model was pretrained on 256x256 images and then finetuned on 512x512 images.

The vision and language encoder of CLIP use the same transformer layer, the only difference is that the former not use attn mask.
The openai clip tokenizer use "!" as sequence pad, while stable-diffusion use "<endoftext>" as pad.

## Prepare

download weights
[https://huggingface.co/jinseokim/stable-diffusion-pytorch-data/resolve/main/data.v20221029.tar]

"ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",

Model card: https://huggingface.co/maze/sd


## Reference
https://github.com/openai/CLIP.git
https://github.com/CompVis/stable-diffusion
