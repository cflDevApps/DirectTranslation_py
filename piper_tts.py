# tts/piper_tts_stream.py

import subprocess
import sounddevice as sd
import numpy as np
from queue import Queue
from threading import Thread


class PiperTTS:
    def __init__(self, model_path, piper_path="piper"):
        self.model_path = model_path
        self.piper_path = piper_path

        self.queue = Queue(maxsize=5)
        self.running = True

        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    # ---------------- WORKER ----------------
    def _worker(self):
        while self.running:
            text = self.queue.get()

            if text is None:
                continue

            try:
                self._speak(text)
            except Exception as e:
                print("TTS Error:", e)

    # ---------------- TTS STREAM ----------------
    def _speak(self, text):
        process = subprocess.Popen(
            [
                self.piper_path,
                "--model", self.model_path,
                "--output_raw"   # 🔥 saída RAW PCM
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # envia texto
        stdout_data, stderr_data = process.communicate(input=text.encode("utf-8"))

        if process.returncode != 0:
            print("Piper error:", stderr_data.decode())
            return

        # ---------------- CONVERTER AUDIO ----------------
        # Piper output_raw = PCM 16-bit mono 22050Hz
        audio = np.frombuffer(stdout_data, dtype=np.int16)

        # normalizar para float32
        audio = audio.astype(np.float32) / 32768.0

        samplerate = 22050  # padrão Piper

        # tocar áudio
        sd.play(audio, samplerate)
        sd.wait()

    # ---------------- API ----------------
    def speak(self, text):
        if not text or not text.strip():
            return

        # evita backlog (tempo real)
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except:
                pass

        self.queue.put(text)

    def stop(self):
        self.running = False
        self.queue.put(None)