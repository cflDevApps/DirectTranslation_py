import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger("directtranslation.pipeline")


@dataclass
class _TranscribedItem:
    text: str
    asr_ms: float


@dataclass
class _TranslatedItem:
    original: str
    translated: str
    asr_ms: float          # carregado desde o estágio ASR
    translation_ms: float


class AsyncTranslationPipeline:
    """
    Três workers assíncronos independentes conectados por asyncio.Queue.

    Fluxo:
        AudioCapture --[audio_q]--> ASR worker --[text_q]--> Translation worker
                     --[translated_q]--> TTS worker --> Speaker

    Operações GPU rodam em ThreadPoolExecutor para não bloquear o event loop.
    Shutdown propagado via sentinel None através das filas.
    Callbacks opcionais permitem integração com a UI sem acoplamento direto.
    """

    def __init__(
        self,
        asr,
        vad,
        translator,
        tts,
        config,
        on_transcription: Optional[Callable[[str, float], None]] = None,
        on_translation: Optional[Callable[[str, str, float, float], None]] = None,
        on_tts_complete: Optional[Callable[[float], None]] = None,
    ):
        self._asr = asr
        self._vad = vad
        self._translator = translator
        self._tts = tts

        # Callbacks para a UI (chamados das threads do executor — thread-safe com Qt signals)
        self._on_transcription = on_transcription   # (text, asr_ms)
        self._on_translation = on_translation       # (original, translated, asr_ms, tr_ms)
        self._on_tts_complete = on_tts_complete     # (tts_ms,)

        self._energy_threshold: float = config.audio.energy_threshold

        p = config.pipeline
        self._audio_q: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue(
            maxsize=p.audio_queue_size
        )
        self._text_q: asyncio.Queue[Optional[_TranscribedItem]] = asyncio.Queue(
            maxsize=p.text_queue_size
        )
        self._translated_q: asyncio.Queue[Optional[_TranslatedItem]] = asyncio.Queue(
            maxsize=p.translated_queue_size
        )

        self._executor = ThreadPoolExecutor(
            max_workers=p.gpu_pool_workers,
            thread_name_prefix="gpu",
        )
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._latency: dict[str, deque] = {
            "asr": deque(maxlen=20),
            "translation": deque(maxlen=20),
            "tts": deque(maxlen=20),
        }

    # ── Thread-safe: chamado do callback do sounddevice ──────────────────

    def feed_audio(self, audio: np.ndarray):
        def _put():
            try:
                self._audio_q.put_nowait(audio)
            except asyncio.QueueFull:
                logger.debug("Audio queue cheia — chunk descartado")

        self._loop.call_soon_threadsafe(_put)

    # ── Worker 1: VAD + ASR ──────────────────────────────────────────────

    async def _asr_worker(self):
        loop = asyncio.get_running_loop()

        while self._running:
            audio = await self._audio_q.get()
            if audio is None:
                await self._text_q.put(None)
                break

            try:
                energy = float(np.sqrt(np.mean(audio ** 2)))
                if energy < self._energy_threshold:
                    continue

                if self._vad is not None:
                    has_speech = await loop.run_in_executor(
                        self._executor, self._vad.is_speech, audio
                    )
                    if not has_speech:
                        continue

                t0 = time.perf_counter()
                text = await loop.run_in_executor(self._executor, self._asr.transcribe, audio)
                asr_ms = (time.perf_counter() - t0) * 1000
                self._latency["asr"].append(asr_ms)

                if text:
                    logger.info(f"[ASR {asr_ms:.0f}ms] {text}")
                    if self._on_transcription:
                        self._on_transcription(text, asr_ms)
                    await self._text_q.put(_TranscribedItem(text=text, asr_ms=asr_ms))
            except Exception as e:
                logger.error(f"ASR worker erro (chunk ignorado): {e}", exc_info=True)

    # ── Worker 2: Tradução ───────────────────────────────────────────────

    async def _translation_worker(self):
        loop = asyncio.get_running_loop()

        while self._running:
            item = await self._text_q.get()
            if item is None:
                await self._translated_q.put(None)
                break

            try:
                t0 = time.perf_counter()
                translated = await loop.run_in_executor(
                    self._executor, self._translator.translate, item.text
                )
                tr_ms = (time.perf_counter() - t0) * 1000
                self._latency["translation"].append(tr_ms)

                logger.info(f"[TR {tr_ms:.0f}ms] {translated}")
                if self._on_translation:
                    self._on_translation(item.text, translated, item.asr_ms, tr_ms)
                await self._translated_q.put(
                    _TranslatedItem(
                        original=item.text,
                        translated=translated,
                        asr_ms=item.asr_ms,
                        translation_ms=tr_ms,
                    )
                )
            except Exception as e:
                logger.error(f"Translation worker erro (item ignorado): {e}", exc_info=True)

    # ── Worker 3: TTS ────────────────────────────────────────────────────

    async def _tts_worker(self):
        loop = asyncio.get_running_loop()

        while self._running:
            item = await self._translated_q.get()
            if item is None:
                break

            try:
                t0 = time.perf_counter()
                await loop.run_in_executor(self._executor, self._tts.speak_sync, item.translated)
                tts_ms = (time.perf_counter() - t0) * 1000
                self._latency["tts"].append(tts_ms)

                logger.info(f"[TTS {tts_ms:.0f}ms]")
                if self._on_tts_complete:
                    self._on_tts_complete(tts_ms)
            except Exception as e:
                logger.error(f"TTS worker erro (item ignorado): {e}", exc_info=True)

    # ── Ciclo de vida ────────────────────────────────────────────────────

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._tasks = [
            asyncio.create_task(self._asr_worker(),         name="asr"),
            asyncio.create_task(self._translation_worker(), name="translator"),
            asyncio.create_task(self._tts_worker(),         name="tts"),
        ]
        logger.info("Pipeline assíncrono iniciado (3 workers)")

    async def stop(self):
        self._running = False
        await self._audio_q.put(None)
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._executor.shutdown(wait=False)
        self._report_latency()
        logger.info("Pipeline encerrado.")

    def _report_latency(self):
        logger.info("── Latência média ──────────────────────────")
        for stage, samples in self._latency.items():
            if samples:
                avg = sum(samples) / len(samples)
                logger.info("  %-12s %5.0f ms  (%d amostras)", stage, avg, len(samples))
