import asyncio
import logging
import numpy as np
import sounddevice as sd

from src.audio.device_utils import resolve_input_device

logger = logging.getLogger("directtranslation.capture")


class AsyncAudioCapture:
    """
    Captura áudio do microfone e injeta chunks no AsyncTranslationPipeline.

    O callback do sounddevice roda em thread nativa. A injeção no pipeline
    é feita via loop.call_soon_threadsafe para garantir thread-safety com asyncio.
    """

    def __init__(
        self,
        pipeline,
        sample_rate: int = 16000,
        device=None,
        chunk_duration: float = 2.5,
        overlap_duration: float = 0.25,
    ):
        self._pipeline = pipeline
        self._sample_rate = sample_rate
        self._device = device
        self._chunk_samples = int(sample_rate * chunk_duration)
        self._overlap_samples = int(sample_rate * overlap_duration)
        self._buffer = np.array([], dtype=np.float32)
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")

        audio = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        self._buffer = np.concatenate([self._buffer, audio.copy()])

        while len(self._buffer) >= self._chunk_samples:
            chunk = self._buffer[: self._chunk_samples].copy()
            # Mantém sobreposição para continuidade entre chunks
            self._buffer = self._buffer[self._chunk_samples - self._overlap_samples :]
            self._pipeline.feed_audio(chunk)

    def start(self):
        resolved = resolve_input_device(self._device)
        self._buffer = np.array([], dtype=np.float32)
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            device=resolved,
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()
        logger.info("AsyncAudioCapture iniciado.")

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            logger.info("AsyncAudioCapture encerrado.")
