import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import streamlit as st

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)


def image_from_text(prompt):
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
    
    return image

def st_ui():
    st.title("Stable Diffusion")
    prompt = st.text_input("Enter your prompt")

    image = image_from_text(prompt)

    st.image(image)

if __name__ == "__main__":
    st_ui()
