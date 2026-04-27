import numpy as np
import torch
import logging
from silero_vad import load_silero_vad, get_speech_timestamps

logger = logging.getLogger("directtranslation.vad")


class SileroVAD:
    SAMPLE_RATE = 16000

    def __init__(self, device: str = "cuda", threshold: float = 0.5):
        self.device = device
        self.threshold = threshold
        logger.info(f"Carregando SileroVAD ({device})...")
        self.model = load_silero_vad()
        self.model.to(device)
        logger.info("SileroVAD carregado.")

    def is_speech(self, audio: np.ndarray) -> bool:
        tensor = torch.from_numpy(audio).float().to(self.device)
        timestamps = get_speech_timestamps(
            tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.SAMPLE_RATE,
        )
        return len(timestamps) > 0
