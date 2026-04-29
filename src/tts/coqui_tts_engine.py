import numpy as np
import sounddevice as sd
import logging
from queue import Queue
from threading import Thread

logger = logging.getLogger("directtranslation.tts.coqui")

SAMPLE_RATE = 22050


class CoquiTTSEngine:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/vits", device: str = "cuda"):
        from TTS.api import TTS
        import torch

        use_gpu = device == "cuda" and torch.cuda.is_available()
        if device == "cuda" and not use_gpu:
            logger.warning("Coqui TTS: CUDA indisponivel (PyTorch CPU-only). Usando CPU.")

        logger.info(f"Carregando Coqui TTS ({model_name}, gpu={use_gpu})...")
        self._tts = TTS(model_name, gpu=use_gpu)
        logger.info("Coqui TTS carregado.")

        self._queue: Queue = Queue(maxsize=5)
        self._running = True
        Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while self._running:
            text = self._queue.get()
            if text is None:
                break
            try:
                self.speak_sync(text)
            except Exception as e:
                logger.error(f"Coqui TTS error: {e}")

    def speak_sync(self, text: str):
        wav = self._tts.tts(text=text)
        audio = np.array(wav, dtype=np.float32)
        sd.play(audio, SAMPLE_RATE)
        sd.wait()

    def speak(self, text: str):
        if not text or not text.strip():
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Exception:
                pass
        self._queue.put(text)

    def stop(self):
        self._running = False
        self._queue.put(None)
