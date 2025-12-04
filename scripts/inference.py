import sys
sys.path.append("/data/xiaoxinghhh/code/ProCLIP")

from openclip import create_model_and_transforms
from PIL import Image

import torch

model, _, preprocess_val = create_model_and_transforms(
    model_name="ViT-L-14",
    pretrained="/data/xiaoxinghhh/code/ProCLIP/checkpoints/L14.pt",
    force_quick_gelu=True,
    force_custom_clip=True,
    eval=True
)


model.to("cuda")
model.eval()

imgs = preprocess_val(Image.open("assets/CLIP.png"))
imgs = imgs.unsqueeze(0).to('cuda')
# print(image.shape)

# image = image.to("cuda")
text = ["a diagram", "a dog", "a cat"]

image_features, text_features, logit_scale = model(imgs, text)

logits_per_image = logit_scale * image_features @ text_features.t()
# logits_per_text = logits_per_image.t()

print(logits_per_image)
# print(logits_per_text)
