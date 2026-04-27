import sounddevice as sd
import numpy as np
import time
import logging
from queue import Queue
from threading import Thread

logger = logging.getLogger("directtranslation.audio")


class HybridAudioPipeline:
    def __init__(
        self,
        whisper_model,
        sample_rate: int = 16000,
        chunk_duration: float = 2.5,
        overlap_duration: float = 0.25,
        energy_threshold: float = 0.002,
        silence_timeout: float = 0.8,
        device=None,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.overlap_samples = int(sample_rate * overlap_duration)
        self.energy_threshold = energy_threshold
        self.silence_timeout = silence_timeout
        self.device = device

        self.whisper = whisper_model

        self.buffer = []
        self.buffer_samples = 0
        self.last_speech_time = 0.0
        self.speaking = False

        self.raw_queue: Queue = Queue()
        self.text_queue: Queue = Queue()

        self.running = False
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        data = indata.copy()
        if data.ndim > 1:
            data = data[:, 0]
        self.raw_queue.put(data)

    def _start_stream(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            device=self.device,
            blocksize=1024,
        )
        self.stream.start()
        logger.info(f"Stream de audio iniciado (device={self.device})")

    def _worker(self):
        while self.running:
            try:
                data = self.raw_queue.get(timeout=0.1)
            except Exception:
                continue

            self.buffer.append(data)
            self.buffer_samples += len(data)

            if self.buffer_samples < self.chunk_samples:
                continue

            chunk = np.concatenate(self.buffer, axis=0)[: self.chunk_samples]

            if self.overlap_samples > 0:
                self.buffer = [chunk[-self.overlap_samples :]]
                self.buffer_samples = self.overlap_samples
            else:
                self.buffer = []
                self.buffer_samples = 0

            energy = float(np.mean(np.abs(chunk)))

            if energy > self.energy_threshold:
                self.last_speech_time = time.time()
                if not self.speaking:
                    logger.debug("Inicio da fala detectado")
                    self.speaking = True

                text = self.whisper.transcribe(chunk)
                if text:
                    self.text_queue.put(text)
            else:
                if self.speaking and (time.time() - self.last_speech_time > self.silence_timeout):
                    logger.debug("Fim da fala detectado")
                    self.speaking = False

    def start(self):
        self.running = True
        self._start_stream()
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_text(self) -> str | None:
        if not self.text_queue.empty():
            return self.text_queue.get()
        return None
