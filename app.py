import sys

from src.core.logging import setup_logging

logger = setup_logging()


def _run_gui():
    from PySide6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("DirectTranslation")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def _run_cli():
    import asyncio
    from src.config import AppConfig
    from src.core.async_pipeline import AsyncTranslationPipeline
    from src.audio.async_audio_capture import AsyncAudioCapture
    from src.asr.whisper_engine import WhisperEngine
    from src.translation.translation_engine import TranslationEngine

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

        vad = None
        if config.vad.enabled:
            try:
                from src.audio.silero_vad import SileroVAD
                vad = SileroVAD(device=config.vad.device, threshold=config.vad.threshold)
            except Exception as e:
                logger.warning(f"SileroVAD indisponivel: {e}")

        translator = TranslationEngine(
            model_path=config.translation.model_path,
            source_language=config.translation.source_language,
            target_language=config.translation.target_language,
            device=config.translation.device,
        )

        if config.tts.engine.lower() == "coqui":
            try:
                from src.tts.coqui_tts_engine import CoquiTTSEngine
                tts = CoquiTTSEngine(model_name=config.tts.coqui_model, device=config.tts.device)
            except Exception as e:
                logger.warning(f"Coqui TTS indisponivel: {e}")
                from src.tts.piper_tts import PiperTTS
                tts = PiperTTS(model_path=config.tts.model_path, piper_path=config.tts.piper_path)
        else:
            from src.tts.piper_tts import PiperTTS
            tts = PiperTTS(model_path=config.tts.model_path, piper_path=config.tts.piper_path)

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
            await asyncio.get_running_loop().create_future()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            logger.info("Encerrando...")
            capture.stop()
            await pipeline.stop()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if "--cli" in sys.argv:
        _run_cli()
    else:
        _run_gui()
