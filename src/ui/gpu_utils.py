from typing import TypedDict


class GPUStats(TypedDict):
    available: bool
    name: str
    used_gb: float
    total_gb: float


def get_gpu_stats() -> GPUStats:
    """Retorna nome, VRAM usada e total via PyTorch. Não requer dependência extra."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "name": "—", "used_gb": 0.0, "total_gb": 0.0}

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        used_gb = torch.cuda.memory_allocated(device) / (1024**3)
        total_gb = props.total_memory / (1024**3)
        return {
            "available": True,
            "name": props.name,
            "used_gb": used_gb,
            "total_gb": total_gb,
        }
    except Exception:
        return {"available": False, "name": "—", "used_gb": 0.0, "total_gb": 0.0}
