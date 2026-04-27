from src.config import AppConfig
from src.core.logging import setup_logging
from src.audio.hybrid_audio_pipeline import HybridAudioPipeline
from src.asr.whisper_engine import WhisperEngine
from src.translation.translation_engine import TranslationEngine
from piper_tts import PiperTTS

logger = setup_logging()


def start():
    config = AppConfig.from_file("config.yaml")

    whisper = WhisperEngine(
        model_size=config.asr.model_size,
        source_language=config.asr.source_language,
        device=config.asr.device,
        compute_type=config.asr.compute_type,
        cpu_threads=config.asr.cpu_threads,
        num_workers=config.asr.num_workers,
    )

    pipeline = HybridAudioPipeline(
        whisper_model=whisper,
        sample_rate=config.audio.sample_rate,
        chunk_duration=config.audio.chunk_duration,
        overlap_duration=config.audio.overlap_duration,
        energy_threshold=config.audio.energy_threshold,
        silence_timeout=config.audio.silence_timeout,
        device=config.audio.device,
    )

    translator = TranslationEngine(
        source_language=config.translation.source_language,
        target_language=config.translation.target_language,
    )

    tts = PiperTTS(
        model_path=config.tts.model_path,
        piper_path=config.tts.piper_path,
    )

    pipeline.start()
    logger.info("Fale algo...")

    try:
        while True:
            sentence = pipeline.get_text()
            if sentence:
                logger.info(f"Palestrante: {sentence}")
                translated = translator.translate(sentence)
                logger.info(f"Traducao: {translated}")
                tts.speak(translated)
    except KeyboardInterrupt:
        logger.info("Encerrando...")
        pipeline.stop()
    except Exception:
        logger.error("Erro inesperado no pipeline", exc_info=True)
        pipeline.stop()


if __name__ == "__main__":
    start()
