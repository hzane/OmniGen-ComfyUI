import torch

def get_vram_info():
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        r = torch.cuda.memory_reserved() / (1024**3)
        a = torch.cuda.memory_allocated() / (1024**3)
        f = t - (r + a)
        return f"VRAM: Total {t:.2f}GB | Reserved {r:.2f}GB | Allocated {a:.2f}GB | Free {f:.2f}GB"
    return "CUDA not available"