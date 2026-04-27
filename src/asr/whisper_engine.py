from faster_whisper import WhisperModel
import logging

logger = logging.getLogger("directtranslation.asr")


class WhisperEngine:
    def __init__(
        self,
        model_size: str = "medium",
        source_language: str = "pt",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        cpu_threads: int = 4,
        num_workers: int = 2,
    ):
        self.source_language = source_language
        logger.info(f"Carregando modelo Whisper ({model_size})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        logger.info("Modelo Whisper carregado.")

    def transcribe(self, audio) -> str:
        segments, _ = self.model.transcribe(audio, language=self.source_language)
        return "".join(seg.text for seg in segments).strip()
