from typing import TypedDict


class GPUStats(TypedDict):
    available: bool
    name: str
    used_gb: float
    total_gb: float


def torch_cuda_available() -> bool:
    """PyTorch CUDA — usado por SileroVAD e Coqui TTS."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def ct2_cuda_available() -> bool:
    """CTranslate2 CUDA — usado por Whisper e TranslationEngine."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_device(requested: str) -> str:
    """
    Retorna 'cuda' se o device solicitado é CUDA e GPU está disponível;
    caso contrário retorna 'cpu' com aviso no log.
    Checa tanto PyTorch quanto CTranslate2.
    """
    import logging
    logger = logging.getLogger("directtranslation.gpu")

    if requested != "cuda":
        return requested

    if torch_cuda_available() or ct2_cuda_available():
        return "cuda"

    logger.warning(
        "CUDA solicitado mas nao disponivel. Usando CPU. "
        "Execute 'python check_gpu.py' para diagnostico e instrucoes de instalacao."
    )
    return "cpu"


def get_gpu_stats() -> GPUStats:
    """Retorna nome, VRAM usada e total via PyTorch."""
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
