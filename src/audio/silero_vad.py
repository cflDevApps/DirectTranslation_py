import numpy as np
import torch
import logging
from silero_vad import load_silero_vad, get_speech_timestamps

logger = logging.getLogger("directtranslation.vad")


class SileroVAD:
    SAMPLE_RATE = 16000

    def __init__(self, device: str = "cuda", threshold: float = 0.5):
        # Verifica disponibilidade de CUDA via PyTorch
        resolved = device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("SileroVAD: CUDA indisponivel (PyTorch CPU-only). Usando CPU.")
            resolved = "cpu"

        self.device = resolved
        self.threshold = threshold
        logger.info(f"Carregando SileroVAD ({resolved})...")
        self.model = load_silero_vad()
        self.model.to(resolved)
        logger.info("SileroVAD carregado.")

    def is_speech(self, audio: np.ndarray) -> bool:
        try:
            tensor = torch.from_numpy(audio).float().to(self.device)
            timestamps = get_speech_timestamps(
                tensor,
                self.model,
                threshold=self.threshold,
                sampling_rate=self.SAMPLE_RATE,
            )
            return len(timestamps) > 0
        except Exception as e:
            logger.error(f"VAD erro (fail-open, audio sera processado): {e}", exc_info=True)
            return True
