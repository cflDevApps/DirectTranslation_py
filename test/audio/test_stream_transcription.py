# test_hybrid.py

from src.audio.microphone_stream import HybridAudioPipeline
from src.asr.whisper_engine import WhisperEngine

whisper = WhisperEngine("small")

pipeline = HybridAudioPipeline(
    whisper_model=whisper,
    energy_threshold=0.0015,  # 🔥 ajustado pro seu caso
    silence_timeout=1.0,
    device=1
)

pipeline.start()

print("Fale algo...")

try:
    while True:
        text = pipeline.get_text()
        if text:
            print("🧠 Palestrante:", text)

except KeyboardInterrupt:
    pipeline.stop()