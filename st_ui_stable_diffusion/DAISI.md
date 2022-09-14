# Stable Diffusion running on GPUs on Daisi

This Daisi is the serverless deployment on GPUs of the latent text-to-image diffusion model Stable Diffusion
by Rombach et al., 2022. "High-Resolution Image Synthesis With Latent Diffusion Models".

This is a straight deployment of the model hosted on HugginFace using the diffusers library.

To call it in Python:

```python

import pydaisi as pyd
stable_diffusion = pyd.Daisi("laiglejm/Stable Diffusion")

res = stable_diffusion.image_from_text(prompt="a cat with a hat", 
                                        samples=4, 
                                        scale=7.5, 
                                        steps=10, 
                                        seed=1024).value
# returns an array of Pillow Images

from PIL import Image
for im in res:
    display(im)
```
