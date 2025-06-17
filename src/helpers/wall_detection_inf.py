import argparse
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image
import numpy as np
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoint-200000/controlnet",
    )
    return parser.parse_args()

class WallDetection:
    def __init__(self, ckpt_path, stable_diffusion_ckpt = 'CompVis/stable-diffusion-v1-4'):
        controlnet = ControlNetModel.from_pretrained(ckpt_path, torch_dtype=torch.float16)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            stable_diffusion_ckpt,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe = pipe.to(self.device)

    def infer(self, image_path, prompt='A floor plan', num_images=16):
        image = Image.open(image_path)
        # infer 16 samples
        out = self.pipe(
            prompt,
            num_inference_steps=50,
            image=image,
            height=1024, width=1024,
            controlnet_conditioning_scale=1.0,
            guidance_scale=7.5,
            generator=[torch.manual_seed(s) for s in range(num_images)],
            num_images_per_prompt=num_images
        )
        # aggregate them to an average result
        I = np.stack([np.asarray(img) for img in out.images]).mean(axis=0).mean(axis=-1)
        I = np.uint8(I)
        return Image.fromarray(np.uint8((I > 127) * 255))

    def infer_pil(self, image, prompt='A floor plan', num_images=16):
        # infer 16 samples
        out = self.pipe(
            prompt,
            num_inference_steps=50,
            image=image,
            height=1536, width=1536,
            controlnet_conditioning_scale=1.0,
            guidance_scale=7.5,
            generator=[torch.manual_seed(s) for s in range(num_images)],
            num_images_per_prompt=num_images
        )
        # aggregate them to an average result
        I = np.stack([np.asarray(img) for img in out.images]).mean(axis=0).mean(axis=-1)
        I = np.uint8(I)
        return Image.fromarray(np.uint8((I > 127) * 255))