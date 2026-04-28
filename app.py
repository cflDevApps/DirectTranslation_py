import asyncio

from src.config import AppConfig
from src.core.logging import setup_logging
from src.core.async_pipeline import AsyncTranslationPipeline
from src.audio.async_audio_capture import AsyncAudioCapture
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
    if config.tts.engine.lower() == "coqui":
        try:
            from src.tts.coqui_tts_engine import CoquiTTSEngine
            return CoquiTTSEngine(model_name=config.tts.coqui_model, device=config.tts.device)
        except Exception as e:
            logger.warning(f"Coqui TTS indisponivel, usando Piper: {e}")
    from src.tts.piper_tts import PiperTTS
    return PiperTTS(model_path=config.tts.model_path, piper_path=config.tts.piper_path)


async def main():
    config = AppConfig.from_file("config.yaml")

    asr = WhisperEngine(
        model_size=config.asr.model_size,
        source_language=config.asr.source_language,
        device=config.asr.device,
        compute_type=config.asr.compute_type,
        cpu_threads=config.asr.cpu_threads,
        num_workers=config.asr.num_workers,
    )
    vad = _create_vad(config)
    translator = TranslationEngine(
        model_path=config.translation.model_path,
        source_language=config.translation.source_language,
        target_language=config.translation.target_language,
        device=config.translation.device,
    )
    tts = _create_tts(config)

    pipeline = AsyncTranslationPipeline(asr, vad, translator, tts, config)
    capture = AsyncAudioCapture(
        pipeline=pipeline,
        sample_rate=config.audio.sample_rate,
        device=config.audio.device,
        chunk_duration=config.audio.chunk_duration,
        overlap_duration=config.audio.overlap_duration,
    )

    await pipeline.start()
    capture.start()
    logger.info("Fale algo... (Ctrl+C para encerrar)")

    try:
        # Aguarda indefinidamente; cancelado por KeyboardInterrupt via asyncio.run
        await asyncio.get_running_loop().create_future()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        logger.info("Encerrando...")
        capture.stop()
        await pipeline.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
