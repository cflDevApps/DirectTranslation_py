from faster_whisper import WhisperModel


class WhisperEngine:

    def __init__(self, model_size="medium"):

        print("Carregando modelo Whisper...")

        self.model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="int8_float16",
            cpu_threads = 8,
            num_workers = 1
        )

        print("Modelo carregado")

    def transcribe(self, audio):
        segments, info = self.model.transcribe(
            audio,
            language="pt"
        )
        text = ""

        for segment in segments:
            text += segment.text

        return text.strip()