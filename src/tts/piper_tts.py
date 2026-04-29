import subprocess
import sounddevice as sd
import numpy as np
import logging
from queue import Queue
from threading import Thread

logger = logging.getLogger("directtranslation.tts.piper")

SAMPLE_RATE = 22050


class PiperTTS:
    def __init__(self, model_path: str, piper_path: str = "piper"):
        self.model_path = model_path
        self.piper_path = piper_path

        self.queue: Queue = Queue(maxsize=5)
        self.running = True

        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        while self.running:
            text = self.queue.get()
            if text is None:
                continue
            try:
                self.speak_sync(text)
            except Exception as e:
                logger.error(f"Piper TTS error: {e}")

    def speak_sync(self, text: str):
        process = subprocess.Popen(
            [self.piper_path, "--model", self.model_path, "--output_raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_data, stderr_data = process.communicate(input=text.encode("utf-8"))

        if process.returncode != 0:
            logger.error(f"Piper error: {stderr_data.decode()}")
            return

        audio = np.frombuffer(stdout_data, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, SAMPLE_RATE)
        sd.wait()

    def speak(self, text: str):
        if not text or not text.strip():
            return
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except Exception:
                pass
        self.queue.put(text)

    def stop(self):
        self.running = False
        self.queue.put(None)
