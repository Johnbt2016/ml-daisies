import streamlit as st
import pydaisi as pyd

import pydaisi as pyd
stable_diffusion = pyd.Daisi("laiglejm/stable diffusion", instance="dev3")

def st_ui():
    st.title("Stable Diffusion")
    prompt = st.text_input("Enter your prompt")

    nb_samples = st.sidebar.number_input("Number of images", 4)
    guidance = st.sidebar.number_input("Guidance", 7.5)
    steps = st.sidebar.number_input("Steps", 45)
    seed = st.sidebar.number_input("Seed", 1024)

    if st.button("Generate !"):

        images = stable_diffusion.image_from_text(prompt,samples=nb_samples, scale=guidance, steps=steps, seed=seed).value

        for im in images:
            st.image(im)

if __name__ == "__main__":
    st_ui()