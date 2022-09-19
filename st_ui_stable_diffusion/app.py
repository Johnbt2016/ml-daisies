import streamlit as st
import pydaisi as pyd
import time

import pydaisi as pyd
stable_diffusion = pyd.Daisi("laiglejm/Stable Diffusion")

def image_from_text(prompt, samples=4, scale=7.5, steps=45, seed=1024):
    '''
    Returns an array of Pillow images.
    '''

    images = stable_diffusion.image_from_text(prompt,samples=samples, scale=scale, steps=steps, seed=seed).value

    return images


def st_ui():
    st.title("Stable Diffusion")

    if stable_diffusion.workers.number == 0:
        st.info("Stable Diffusion service is currently not started. Allow 30 to 45 seconds for service to start when trigerring the first execution. Subsequent executions should complete in 4 to ~10 seconds, depending on your parameters.")
    else:
        st.info("Stable Diffusion service is currently live. Executions should complete in 4 to ~10 seconds, depending on your parameters.")

    prompt = st.text_area("Enter your prompt", value="a cat with a hat, ultra-detailed. anime, pixiv, uhd 8k cryengine, octane render")

    nb_samples = st.sidebar.number_input("Number of images", value=4)
    guidance = st.sidebar.number_input("Guidance", value=7.5)
    steps = st.sidebar.number_input("Steps", value=45)
    seed = st.sidebar.number_input("Seed", value=1024)

    if st.button("Generate !"):
        with st.spinner(f"Generating your images (takes a few seconds). {time.time(),2}"):
            images = image_from_text(prompt, samples=nb_samples, scale=guidance, steps=steps, seed=seed)

        for im in images:
            st.image(im)

if __name__ == "__main__":
    st_ui()