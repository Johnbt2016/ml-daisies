import streamlit as st
import pydaisi as pyd

import pydaisi as pyd
stable_diffusion = pyd.Daisi("laiglejm/Stable Diffusion")

def image_from_text(prompt, samples=4, scale=7.5, steps=45, seed=1024):
    images = stable_diffusion.image_from_text(prompt,samples=samples, scale=scale, steps=steps, seed=seed).value

    return images


def st_ui():
    st.title("Stable Diffusion")
    prompt = st.text_input("Enter your prompt")

    nb_samples = st.sidebar.number_input("Number of images", value=4)
    guidance = st.sidebar.number_input("Guidance", value=7.5)
    steps = st.sidebar.number_input("Steps", value=45)
    seed = st.sidebar.number_input("Seed", value=1024)

    if st.button("Generate !"):
        with st.spinner("Generating your images (takes a few seconds)"):
            images = image_from_text(prompt, samples=nb_samples, scale=guidance, steps=steps, seed=seed)

        for im in images:
            st.image(im)

if __name__ == "__main__":
    st_ui()