from src.config import AppConfig
from src.core.logging import setup_logging
from src.core.benchmark import LatencyTracker
from src.audio.hybrid_audio_pipeline import HybridAudioPipeline
from src.asr.whisper_engine import WhisperEngine
from src.translation.translation_engine import TranslationEngine

logger = setup_logging()


def _create_vad(config: AppConfig):
    if not config.vad.enabled:
        return None
    try:
        from src.audio.silero_vad import SileroVAD
        return SileroVAD(device=config.vad.device, threshold=config.vad.threshold)
    except Exception as e:
        logger.warning(f"SileroVAD indisponivel, usando VAD por energia: {e}")
        return None


def _create_tts(config: AppConfig):
    engine = config.tts.engine.lower()
    if engine == "coqui":
        try:
            from src.tts.coqui_tts_engine import CoquiTTSEngine
            return CoquiTTSEngine(
                model_name=config.tts.coqui_model,
                device=config.tts.device,
            )
        except Exception as e:
            logger.warning(f"Coqui TTS indisponivel, usando Piper: {e}")
    from src.tts.piper_tts import PiperTTS
    return PiperTTS(model_path=config.tts.model_path, piper_path=config.tts.piper_path)


def start():
    config = AppConfig.from_file("config.yaml")
    tracker = LatencyTracker()

    whisper = WhisperEngine(
        model_size=config.asr.model_size,
        source_language=config.asr.source_language,
        device=config.asr.device,
        compute_type=config.asr.compute_type,
        cpu_threads=config.asr.cpu_threads,
        num_workers=config.asr.num_workers,
    )

    vad = _create_vad(config)

    pipeline = HybridAudioPipeline(
        whisper_model=whisper,
        vad=vad,
        sample_rate=config.audio.sample_rate,
        chunk_duration=config.audio.chunk_duration,
        overlap_duration=config.audio.overlap_duration,
        energy_threshold=config.audio.energy_threshold,
        silence_timeout=config.audio.silence_timeout,
        device=config.audio.device,
    )

    translator = TranslationEngine(
        model_path=config.translation.model_path,
        source_language=config.translation.source_language,
        target_language=config.translation.target_language,
        device=config.translation.device,
    )

    tts = _create_tts(config)

    pipeline.start()
    logger.info("Fale algo...")

    try:
        while True:
            sentence = pipeline.get_text()
            if sentence:
                logger.info(f"Palestrante: {sentence}")

                tracker.start("translation")
                translated = translator.translate(sentence)
                tr_ms = tracker.end("translation")

                logger.info(f"Traducao [{tr_ms:.0f}ms]: {translated}")
                tts.speak(translated)
    except KeyboardInterrupt:
        logger.info("Encerrando...")
        pipeline.stop()
        tracker.report()
    except Exception:
        logger.error("Erro inesperado no pipeline", exc_info=True)
        pipeline.stop()


if __name__ == "__main__":
    start()
