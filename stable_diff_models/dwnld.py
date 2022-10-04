# This file is part of stable-diffusion-webui (https://github.com/sd-webui/stable-diffusion-webui/).

# Copyright 2022 sd-webui team.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import os
import os.path as op
import streamlit as st
dest = "/pebble_tmp/models/stable_diffusion_models"

class Models:
    
    # Stable DiffusionV1.4
    def modelSD():
        if op.exists(f'{dest}/models/ldm/stable-diffusion-v1/model.ckpt'):
            return st.write(f"Stable Diffusion model already exists !")
        else:
            # For 4GB model
            # os.system('wget  -O models/ldm/stable-diffusion-v1/model.ckpt https://cdn-lfs.huggingface.co/repos/ab/41/ab41ccb635cd5bd124c8eac1b5796b4f64049c9453c4e50d51819468ca69ceb8/fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556?response-content-disposition=attachment%3B%20filename%3D%22model.ckpt%22')
            # os.rename('models/ldm/stable-diffusion-v1/sd-v1-4.ckpt?alt=media','models/ldm/stable-diffusion-v1/model.ckpt')
            # For 7.2GB model
            os.system(
                f'wget -O {dest}/models/ldm/stable-diffusion-v1/model.ckpt https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media')
            # os.system('wget -O models/ldm/stable-diffusion-v1/model.ckpt https://cdn-lfs.huggingface.co/repos/ab/41/ab41ccb635cd5bd124c8eac1b5796b4f64049c9453c4e50d51819468ca69ceb8/14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a?response-content-disposition=attachment%3B%20filename%3D%22modelfull.ckpt%22')
            # os.rename('models/ldm/stable-diffusion-v1/modelfull.ckpt','models/ldm/stable-diffusion-v1/model.ckpt')
            return st.write(f"Model installed successfully")

    # RealESRGAN_x4plus & RealESRGAN_x4plus_anime_6B
    def realESRGAN():
        if op.exists(f'{dest}/src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus.pth') and op.exists(f'{dest}/src/realesrgan/experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth'):
            return st.write(f"RealESRGAN already exists !")
        else:
            os.system(f'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P {dest}/src/realesrgan/experiments/pretrained_models')
            os.system(f'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {dest}/src/realesrgan/experiments/pretrained_models')
            return st.write(f"ESRGAN upscaler installed successfully !")

    # GFPGANv1.3
    def GFPGAN():
        if op.exists(f'{dest}/src/gfpgan/experiments/pretrained_models/GFPGANv1.3.pth'):
            return st.write(f"GFPGAN already exists !")
        else:
            os.system(
                f'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P {dest}/src/gfpgan/experiments/pretrained_models')
            return st.write(f"GFPGAN installed successfully !")

    # Latent Diffusion
    def modelLD():
        if op.exists(f'{dest}/src/latent-diffusion'):
            return st.write(f"Latent-Diffusion Model already exists !")
        else:
            os.system(f'cd {dest}/models/ ; git clone https://github.com/devilismyfriend/latent-diffusion.git')
            os.system(f'mv {dest}/latent-diffusion {dest}/src/latent-diffusion')
            st.write(f"Github Repository cloned !")
            if op.exists(f'{dest}/src/latent-diffusion/experiments/pretrained_models/model.ckpt'):
                st.write(f"Latent Diffusion model already exists!")
            else:
                os.mkdir(f'{dest}/src/latent-diffusion/experiments')
                os.mkdir(f'{dest}/src/latent-diffusion/experiments/pretrained_models')
                os.system(
                    f'wget -O {dest}/src/latent-diffusion/experiments/pretrained_models/project.yaml https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1')
                # os.rename('src/latent-diffusion/experiments/pretrained_models/index.html?dl=1', 'src/latent-diffusion/experiments/pretrained_models/project.yaml')
                os.system(
                    f'wget -O {dest}/src/latent-diffusion/experiments/pretrained_models/model.ckpt https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1')
                # os.rename('src/latent-diffusion/experiments/pretrained_models/index.html?dl=1', 'src/latent-diffusion/experiments/pretrained_models/model.ckpt')
                return st.write(f"Latent Diffusion successfully installed !")

    # Stable Diffusion Conecpt Library
    def SD_conLib():
        if op.exists(f'{dest}/models/custom/sd-concepts-library'):
            return st.write(f"Stable Diffusion Concept Library already exists !")
        else:
            os.system(
                f'cd {dest}/models/ ; git clone https://github.com/sd-webui/sd-concepts-library models/custom/')
            return st.write("Stable Diffusion Concept Library successfully installed !")

    # Blip Model
    def modelBlip():
        if op.exists(f'{dest}/models/custom/blip/model__base_caption.pth'):
            return st.write(f"Blip Model already exists !")
        else:
            # return st.write(f"Blip Model is to be installed !")
            os.mkdir(f'{dest}/models/custom/blip')
            os.system(f"wget -O {dest}/models/custom/blip/model__base_caption.pth https://cdn-lfs.huggingface.co/repos/cd/15/cd1551e1e53c5049819b5349e3e386c497a767dfeebb8e146ae2adb8f39c8d10/96ac8749bd0a568c274ebe302b3a3748ab9be614c737f3d8c529697139174086?response-content-disposition=attachment%3B%20filename%3D%22model__base_caption.pth%22")
            return st.write(f"Blip model successfully installed")

    # Waifu Diffusion v1.2
    def modelWD():
        if op.exists(f"{dest}/models/custom/waifu-diffusion"):
            return st.write(f"Waifu Diffusion Model already exists !")
        else:
            os.system(
                f"cd {dest}/models/ ;git clone https://huggingface.co/hakurei/waifu-diffusion models/custom/waifu-diffusion")
            return st.write(f"Waifu Diffusion model successfully installed")

    # Waifu Diffusion v1.2 Pruned
    def modelWDP():
        if op.exists(f"{dest}/models/custom/pruned-waifu-diffusion"):
            return st.write(f"Waifu Pruned Model already exists !")
        else:
            os.system(
                f"cd {dest}/models/ ;git clone https://huggingface.co/crumb/pruned-waifu-diffusion models/custom/pruned-waifu-diffusion")
            return st.write(f"Waifu Pruned model successfully installed")

    # TrinArt Stable Diffusion v2
    def modelTSD():
        if op.exists(f"{dest}/models/custom/trinart_stable_diffusion_v2"):
            return st.write(f"Trinart S.D model already exists!")
        else:
            os.system(
                f"cd {dest}/models/ ;git clone https://huggingface.co/naclbit/trinart_stable_diffusion_v2 models/custom/trinart_stable_diffusion_v2")
            return st.write(f"TrinArt successfully installed !")


def st_ui():
    st.title("Stable Diffusion models download")
    if st.button("Stable Diffusion"):
        Models.modelSD()

    if st.button("RealESRGAN"):
        Models.realESRGAN()

    if st.button("GFPGAN"):
        Models.GFPGAN()

    if st.button("Latent Diffusion"):
        Models.modelLD()

    if st.button("SD Concept Lib"):
        Models.SD_conLib()

    if st.button("Blip"):
        Models.modelBlip()

    if st.button("Waifu Diffusion"):
        Models.modelWD()

    if st.button("Waifu Pruned"):
        Models.modelLD()

    if st.button("TrinArt SD"):
        Models.modelTSD()

if __name__ == "__main__":
    st_ui()