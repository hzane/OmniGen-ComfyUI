import os
import torch
import folder_paths
import gc
from .OmniGen import OmniGenPipeline


class OmniGenLoader:
    current_loaded_model = None
    current_model_path = None
    current_dtype = None
    current_memory_config = None

    @classmethod
    def INPUT_TYPES(s):
        model_dirs = []
        base_path = os.path.join(folder_paths.models_dir, "OmniGen")
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                full_path = os.path.join(base_path, item)
                if os.path.isdir(full_path):
                    model_dirs.append(item)

        if not model_dirs:
            model_dirs = ["none"]

        return {"required": {
            "model_name": (model_dirs, ),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], ),
            "store_in_vram": ("BOOLEAN", {
                "default": False,
                "tooltip": "Keep model in VRAM between generations. Faster but uses more VRAM."
            }),
            "separate_cfg_infer": ("BOOLEAN", {
                "default": True,
                "tooltip": "Use separate inference process for different guidance. Reduces memory cost."
            }),
            "offload_model": ("BOOLEAN", {
                "default": False,
                "tooltip": "Offload model to CPU. Reduces VRAM usage but slower."
            })
        }}

    RETURN_TYPES = ("OMNIGEN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"

    def get_model_file(self, model_path):
        """Find any available model file in the directory"""
        for file in os.listdir(model_path):
            if file.endswith('.safetensors'):
                return os.path.join(model_path, file)
        for file in os.listdir(model_path):
            if file.endswith('.pt') or file.endswith('.ckpt'):
                return os.path.join(model_path, file)
        return None

    def load_model(self, model_name, weight_dtype, store_in_vram, separate_cfg_infer, offload_model):
        if model_name == "none":
            raise RuntimeError("No model folder found in models/OmniGen/")

        model_path = os.path.join(folder_paths.models_dir, "OmniGen", model_name)
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model folder {model_name} not found in models/OmniGen/")

        memory_config = (store_in_vram, separate_cfg_infer, offload_model)

        # Check if we need to load a new model
        if (OmniGenLoader.current_loaded_model is None or
            model_path != OmniGenLoader.current_model_path or
            weight_dtype != OmniGenLoader.current_dtype or
            memory_config != OmniGenLoader.current_memory_config):

            # Clean up existing model if it exists
            if OmniGenLoader.current_loaded_model is not None:
                del OmniGenLoader.current_loaded_model
                torch.cuda.empty_cache()
                gc.collect()

            # Convert dtype string to actual dtype
            if weight_dtype == "fp8_e4m3fn":
                dtype = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                dtype = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e5m2":
                dtype = torch.float8_e5m2
            else:
                dtype = torch.bfloat16

            pipe = OmniGenPipeline.from_pretrained(model_path)
            pipe.model = pipe.model.to(dtype)

            if store_in_vram:
                OmniGenLoader.current_loaded_model = pipe
                OmniGenLoader.current_model_path = model_path
                OmniGenLoader.current_dtype = weight_dtype
                OmniGenLoader.current_memory_config = memory_config
        else:
            pipe = OmniGenLoader.current_loaded_model

        return ((pipe, memory_config),)
