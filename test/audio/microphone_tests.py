from src.audio.microphone_stream import MicrophoneStream

mic = MicrophoneStream(
    sample_rate=16000,
    chunk_duration=2
)

mic.start()

print("Capturando áudio...")

while True:

    chunk = mic.get_chunk()

    print("Chunk recebido:", chunk.shape)