import os
import torch
import tempfile
import shutil
import numpy as np
import gc
from PIL import Image


class OmniGenNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("OMNIGEN_MODEL",),
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "default": "you only need image_1, text will auto be <img><|image_1|></img>",
                    "tooltip": "Enter your prompt text here. For images, use <img><|image_1|></img> syntax"
                }),
                "latent": ("LATENT",),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 5.0, "step": 0.1}),
                "img_guidance_scale": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 2.0, "step": 0.1}),
                "max_input_image_size": ("INT", {"default": 1024, "min": 128, "max": 2048, "step": 8}),
                "use_input_image_size_as_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically adjust output image size to match input image"
                }),
                "seed": ("INT", {"default": 42})
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gen"
    CATEGORY = "AIFSH_OmniGen"

    def save_input_img(self,image):
        with tempfile.NamedTemporaryFile(suffix=".png",delete=False,dir="tmp") as f:
            img_np = image.numpy()[0]*255
            img_pil = Image.fromarray(img_np.astype(np.uint8))
            img_pil.save(f.name)
        return f.name

    def gen(self, model, prompt_text, latent, num_inference_steps, guidance_scale,
            img_guidance_scale, max_input_image_size,
            use_input_image_size_as_output, seed, image_1=None, image_2=None, image_3=None):

        pipe, (store_in_vram, separate_cfg_infer, offload_model) = model
        pipe.to(torch.device('cuda'))

        # Get dimensions from latent
        height = latent["samples"].shape[2] * 8
        width = latent["samples"].shape[3] * 8

        input_images = []
        os.makedirs("tmp", exist_ok=True)
        if image_1 is not None:
            input_images.append(self.save_input_img(image_1))
            prompt_text = prompt_text.replace("image_1","<img><|image_1|></img>")

        if image_2 is not None:
            input_images.append(self.save_input_img(image_2))
            prompt_text = prompt_text.replace("image_2","<img><|image_2|></img>")

        if image_3 is not None:
            input_images.append(self.save_input_img(image_3))
            prompt_text = prompt_text.replace("image_3","<img><|image_3|></img>")

        if len(input_images) == 0:
            input_images = None

        print(f"\nGenerating with prompt: {prompt_text}")  # noqa: T201

        output = pipe(
            prompt=prompt_text,
            input_images=input_images,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            num_inference_steps=num_inference_steps,
            separate_cfg_infer=separate_cfg_infer,
            use_kv_cache=True,
            offload_kv_cache=True,
            offload_model=offload_model,
            use_input_image_size_as_output=use_input_image_size_as_output,
            seed=seed,
            max_input_image_size=max_input_image_size,
        )
        pipe.to(torch.device('cpu'))

        img = np.array(output[0]) / 255.0
        img = torch.from_numpy(img).float().unsqueeze(0)
        shutil.rmtree("tmp")

        # Clean up if not storing in VRAM
        if not store_in_vram:
            # del pipe
            torch.cuda.empty_cache()
            gc.collect()
        else:
            ...

        return (img,)
