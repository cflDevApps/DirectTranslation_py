import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from typing import Optional

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
    translation_ms: float


class AsyncTranslationPipeline:
    """
    Três workers assíncronos independentes conectados por asyncio.Queue.

    Fluxo:
        AudioCapture --[audio_q]--> ASR worker --[text_q]--> Translation worker
                     --[translated_q]--> TTS worker --> Speaker

    Operações GPU rodam em ThreadPoolExecutor para não bloquear o event loop.
    Shutdown propagado via sentinel None através das filas.
    """

    def __init__(self, asr, vad, translator, tts, config):
        self._asr = asr
        self._vad = vad
        self._translator = translator
        self._tts = tts

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

        # Acumuladores de latência por estágio
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
                logger.debug("Audio queue cheia — chunk descartado (pipeline congestionado)")

        self._loop.call_soon_threadsafe(_put)

    # ── Worker 1: VAD + ASR ──────────────────────────────────────────────

    async def _asr_worker(self):
        loop = asyncio.get_running_loop()

        while self._running:
            audio = await self._audio_q.get()
            if audio is None:
                await self._text_q.put(None)
                break

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
                await self._text_q.put(_TranscribedItem(text=text, asr_ms=asr_ms))

    # ── Worker 2: Tradução ───────────────────────────────────────────────

    async def _translation_worker(self):
        loop = asyncio.get_running_loop()

        while self._running:
            item = await self._text_q.get()
            if item is None:
                await self._translated_q.put(None)
                break

            t0 = time.perf_counter()
            translated = await loop.run_in_executor(
                self._executor, self._translator.translate, item.text
            )
            tr_ms = (time.perf_counter() - t0) * 1000
            self._latency["translation"].append(tr_ms)

            logger.info(f"[TR {tr_ms:.0f}ms] {translated}")
            await self._translated_q.put(
                _TranslatedItem(
                    original=item.text,
                    translated=translated,
                    translation_ms=tr_ms,
                )
            )

    # ── Worker 3: TTS ────────────────────────────────────────────────────

    async def _tts_worker(self):
        loop = asyncio.get_running_loop()

        while self._running:
            item = await self._translated_q.get()
            if item is None:
                break

            t0 = time.perf_counter()
            await loop.run_in_executor(self._executor, self._tts.speak_sync, item.translated)
            tts_ms = (time.perf_counter() - t0) * 1000
            self._latency["tts"].append(tts_ms)

            logger.info(f"[TTS {tts_ms:.0f}ms]")

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
        await self._audio_q.put(None)  # sentinel dispara encerramento em cascata
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
