import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # segundos

print("Carregando modelo Whisper...")

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

print("Modelo carregado")
print("Fale algo em português...\n")


def record_chunk():

    print("Gravando...")

    audio = sd.rec(
        int(SAMPLE_RATE * CHUNK_DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    return audio.flatten()


while True:

    audio_chunk = record_chunk()

    segments, info = model.transcribe(
        audio_chunk,
        language="en"
    )

    text = ""

    for segment in segments:
        text += segment.text

    print("\nTexto reconhecido:")
    print(text)
    print("-" * 40)