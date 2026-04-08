import sounddevice as sd
import numpy as np
import time
from queue import Queue
from threading import Thread

class HybridAudioPipeline:
    def __init__(self,
                 whisper_model,
                 sample_rate=16000,
                 chunk_duration=2.5,
                 overlap_duration=0.25,
                 energy_threshold=0.002,
                 silence_timeout=0.8,
                 device=None):

        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.overlap_samples = int(sample_rate * overlap_duration)
        self.energy_threshold = energy_threshold
        self.silence_timeout = silence_timeout
        self.device = device

        self.whisper = whisper_model

        # estado
        self.buffer = []
        self.buffer_samples = 0
        self.last_speech_time = 0
        self.speaking = False

        # filas
        self.raw_queue = Queue()
        self.text_queue = Queue()

        self.running = False
        self.stream = None

    # ---------------- CAPTURA ----------------
    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)

        data = indata.copy()

        if data.ndim > 1:
            data = data[:, 0]

        self.raw_queue.put(data)

    def start_stream(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            device=self.device,
            blocksize=1024
        )
        self.stream.start()
        print(f"🎤 Stream iniciado (device={self.device})")

    # ---------------- WORKER ----------------
    def _worker(self):
        while self.running:
            try:
                data = self.raw_queue.get(timeout=0.1)
            except:
                continue

            self.buffer.append(data)
            self.buffer_samples += len(data)

            if self.buffer_samples >= self.chunk_samples:
                chunk = np.concatenate(self.buffer, axis=0)[:self.chunk_samples]

                # overlap
                if self.overlap_samples > 0:
                    keep = chunk[-self.overlap_samples:]
                    self.buffer = [keep]
                    self.buffer_samples = len(keep)
                else:
                    self.buffer = []
                    self.buffer_samples = 0

                # ---------------- ENERGY FILTER ----------------
                energy = np.mean(np.abs(chunk))
                # print(f"Energy: {energy:.5f}")

                if energy > self.energy_threshold:
                    self.last_speech_time = time.time()

                    if not self.speaking:
                        print("🎤 Início da fala")
                        self.speaking = True

                    # ---------------- WHISPER ----------------
                    text = self.whisper.transcribe(chunk)

                    if text and text.strip():
                        self.text_queue.put(text)

                else:
                    # silêncio prolongado
                    if self.speaking and (time.time() - self.last_speech_time > self.silence_timeout):
                        print("🛑 Fim da fala")
                        self.speaking = False

    # ---------------- API ----------------
    def start(self):
        self.running = True
        self.start_stream()

        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_text(self):
        if not self.text_queue.empty():
            return self.text_queue.get()
        return None