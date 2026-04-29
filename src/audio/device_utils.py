import logging
from typing import Optional

logger = logging.getLogger("directtranslation.audio")


def resolve_input_device(device: Optional[int] = None) -> Optional[int]:
    """Valida e loga o dispositivo de entrada. Retorna None para usar o padrão do sistema."""
    import sounddevice as sd

    if device is not None:
        try:
            info = sd.query_devices(device)
            if info["max_input_channels"] < 1:
                logger.warning(
                    f"Dispositivo [{device}] '{info['name']}' nao tem canais de entrada. "
                    "Usando padrao do sistema."
                )
                return None
            logger.info(f"Dispositivo de audio: [{device}] {info['name']}")
        except Exception:
            logger.warning(f"Dispositivo {device} nao encontrado. Usando padrao do sistema.")
            return None
        return device

    try:
        default_idx = sd.default.device[0]
        info = sd.query_devices(default_idx)
        logger.info(f"Dispositivo de audio (padrao): [{default_idx}] {info['name']}")
    except Exception:
        logger.info("Dispositivo de audio: padrao do sistema")

    return None


def list_input_devices() -> str:
    """Retorna string listando todos os dispositivos de entrada disponíveis."""
    import sounddevice as sd

    lines = ["Dispositivos de entrada disponíveis:"]
    try:
        default_idx = sd.default.device[0]
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                marker = " (padrao)" if i == default_idx else ""
                lines.append(f"  [{i}] {dev['name']}{marker}")
    except Exception as e:
        lines.append(f"  Erro ao listar dispositivos: {e}")
    return "\n".join(lines)
