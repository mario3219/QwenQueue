import torch
from diffusers import QwenImage21Pipeline
from PIL import Image
import os

def qwen(pipe, prompt, width, height, steps):
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    return image

def qwen_edit(pipe, prompt, img_input, steps):
    img_path = os.path.abspath(os.path.join(os.getcwd(),"input",img_input))
    input_image = Image.open(img_path)
    image = pipe(
        prompt=prompt,
        image=input_image,
        num_inference_steps=steps,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    return image
