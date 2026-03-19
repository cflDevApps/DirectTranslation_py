import sounddevice as sd
import numpy as np
import queue

class MicrophoneStream:

    def __init__(self, sample_rate=16000, chunk_duration=2):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration

        self.chunk_samples = sample_rate * chunk_duration

        self.buffer = []
        self.audio_queue = queue.Queue()

    def callback(self, indata, frames, time, status):

        if status:
            print(status)

        self.buffer.append(indata.copy())

        current_samples = sum(len(x) for x in self.buffer)

        if current_samples >= self.chunk_samples:

            chunk = np.concatenate(self.buffer)

            chunk = chunk[:self.chunk_samples]

            self.audio_queue.put(chunk)

            self.buffer = []

    def start(self):

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.callback
        )

        self.stream.start()

    def get_chunk(self):

        return self.audio_queue.get()