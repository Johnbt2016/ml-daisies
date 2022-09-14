import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

value = os.getenv("HF_TOKEN")

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=value)
pipe = pipe.to(device)


def image_from_text(prompt, samples=4, scale=7.5, steps=45, seed=1024):

    generator = torch.Generator(device=device).manual_seed(seed)
    
    #If you are running locally with CPU, you can remove the `with autocast("cuda")`
    with autocast("cuda"):
        images_list = pipe(
            [prompt] * samples,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator)
    # with autocast("cuda"):
    #     image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
    images = []
    for i, image in enumerate(images_list["sample"]):
        images.append(image)
    
    return images

