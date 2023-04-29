from PIL import Image
import torch
from models import builder as model_builder
from models.clip import CLIP


device = torch.device("cuda")
clip = model_builder('clip').to(device)
# torch.save(clip.visual.layers[0].attention.state_dict(), "attn.pt")
# exit()
transform, tokenize = CLIP.process(224)

im = Image.open("dog.jpg").convert("RGB")
# im = Image.open("CLIP.png").convert("RGB")
image = transform(im).unsqueeze(dim=0).to(device)
text = tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    # image_features = clip.encode_image(image)
    # text_features = clip.encode_text(text)
    logits_per_image, logits_per_text = clip(image, text)
    # print(logits_per_image)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print(probs)
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


