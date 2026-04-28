import asyncio
import threading
import logging

from PySide6.QtCore import QObject, Signal, Slot

from src.config import AppConfig

logger = logging.getLogger("directtranslation.worker")


class PipelineWorker(QObject):
    """
    Executa AsyncTranslationPipeline em um event loop asyncio dedicado
    rodando em uma thread Python separada. Comunica resultados com a UI
    via Qt Signals — emissão de signal de thread não-Qt é thread-safe no Qt.
    """

    # Sinais emitidos conforme cada estágio completa
    transcription_ready = Signal(str, float)              # text, asr_ms
    translation_ready   = Signal(str, str, float, float)  # original, translated, asr_ms, tr_ms
    tts_complete        = Signal(float)                   # tts_ms

    pipeline_started = Signal()
    pipeline_stopped = Signal()
    error_occurred   = Signal(str)

    def __init__(self, config: AppConfig):
        super().__init__()
        self._config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._thread: threading.Thread | None = None

    # ── Controle público (chamado da thread Qt principal) ────────────────

    @Slot()
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="pipeline-asyncio"
        )
        self._thread.start()

    @Slot()
    def stop(self):
        if self._loop is None:
            return

        def _set():
            if self._stop_event is not None:
                self._stop_event.set()

        self._loop.call_soon_threadsafe(_set)

    # ── Loop asyncio (thread separada) ───────────────────────────────────

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            logger.error("Erro no pipeline", exc_info=True)
            self.error_occurred.emit(str(e))
        finally:
            self._loop.close()
            self._loop = None
            self.pipeline_stopped.emit()

    async def _main(self):
        from src.asr.whisper_engine import WhisperEngine
        from src.translation.translation_engine import TranslationEngine
        from src.core.async_pipeline import AsyncTranslationPipeline
        from src.audio.async_audio_capture import AsyncAudioCapture

        cfg = self._config

        asr = WhisperEngine(
            model_size=cfg.asr.model_size,
            source_language=cfg.asr.source_language,
            device=cfg.asr.device,
            compute_type=cfg.asr.compute_type,
            cpu_threads=cfg.asr.cpu_threads,
            num_workers=cfg.asr.num_workers,
        )
        vad = self._create_vad()
        translator = TranslationEngine(
            model_path=cfg.translation.model_path,
            source_language=cfg.translation.source_language,
            target_language=cfg.translation.target_language,
            device=cfg.translation.device,
        )
        tts = self._create_tts()

        pipeline = AsyncTranslationPipeline(
            asr, vad, translator, tts, cfg,
            on_transcription=lambda text, ms: self.transcription_ready.emit(text, ms),
            on_translation=lambda orig, tr, asr_ms, tr_ms: self.translation_ready.emit(
                orig, tr, asr_ms, tr_ms
            ),
            on_tts_complete=lambda tts_ms: self.tts_complete.emit(tts_ms),
        )
        capture = AsyncAudioCapture(
            pipeline=pipeline,
            sample_rate=cfg.audio.sample_rate,
            device=cfg.audio.device,
            chunk_duration=cfg.audio.chunk_duration,
            overlap_duration=cfg.audio.overlap_duration,
        )

        self._stop_event = asyncio.Event()

        await pipeline.start()
        capture.start()
        self.pipeline_started.emit()
        logger.info("Pipeline UI iniciado.")

        try:
            await self._stop_event.wait()
        finally:
            capture.stop()
            await pipeline.stop()

    def _create_vad(self):
        if not self._config.vad.enabled:
            return None
        try:
            from src.audio.silero_vad import SileroVAD
            return SileroVAD(
                device=self._config.vad.device,
                threshold=self._config.vad.threshold,
            )
        except Exception as e:
            logger.warning(f"SileroVAD indisponivel: {e}")
            return None

    def _create_tts(self):
        cfg = self._config.tts
        if cfg.engine.lower() == "coqui":
            try:
                from src.tts.coqui_tts_engine import CoquiTTSEngine
                return CoquiTTSEngine(model_name=cfg.coqui_model, device=cfg.device)
            except Exception as e:
                logger.warning(f"Coqui TTS indisponivel: {e}")
        from src.tts.piper_tts import PiperTTS
        return PiperTTS(model_path=cfg.model_path, piper_path=cfg.piper_path)
