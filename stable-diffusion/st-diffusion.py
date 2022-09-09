import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import streamlit as st
import os

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

value = os.getenv("HF_TOKEN")
st.write(value)

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

def st_ui():
    st.title("Stable Diffusion")
    prompt = st.text_input("Enter your prompt")

    nb_samples = st.sidebar.number_input("Number of images", 4)
    guidance = st.sidebar.number_input("Guidance", 7.5)
    steps = st.sidebar.number_input("Steps", 45)
    seed = st.sidebar.number_input("Seed", 1024)

    if st.button("Generate !"):

        images = image_from_text(prompt,samples=nb_samples, scale=guidance, steps=steps, seed=seed)

    for im in images:
        st.image(im)

if __name__ == "__main__":
    st_ui()
