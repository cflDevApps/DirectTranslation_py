import asyncio
import pytest
import numpy as np
from unittest.mock import MagicMock


def _mock_config():
    cfg = MagicMock()
    cfg.pipeline.audio_queue_size = 4
    cfg.pipeline.text_queue_size = 8
    cfg.pipeline.translated_queue_size = 8
    cfg.pipeline.gpu_pool_workers = 2
    return cfg


@pytest.mark.asyncio
async def test_pipeline_routes_audio_to_tts():
    from src.core.async_pipeline import AsyncTranslationPipeline

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = "Olá mundo"

    mock_translator = MagicMock()
    mock_translator.translate.return_value = "Hello world"

    mock_tts = MagicMock()
    mock_tts.speak_sync.return_value = None

    pipeline = AsyncTranslationPipeline(
        asr=mock_asr,
        vad=None,
        translator=mock_translator,
        tts=mock_tts,
        config=_mock_config(),
    )
    await pipeline.start()

    pipeline.feed_audio(np.zeros(16000, dtype=np.float32))
    await asyncio.sleep(0.3)
    await pipeline.stop()

    mock_asr.transcribe.assert_called_once()
    mock_translator.translate.assert_called_once_with("Olá mundo")
    mock_tts.speak_sync.assert_called_once_with("Hello world")


@pytest.mark.asyncio
async def test_pipeline_skips_empty_transcription():
    from src.core.async_pipeline import AsyncTranslationPipeline

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ""  # ASR retorna vazio

    mock_translator = MagicMock()
    mock_tts = MagicMock()

    pipeline = AsyncTranslationPipeline(
        asr=mock_asr,
        vad=None,
        translator=mock_translator,
        tts=mock_tts,
        config=_mock_config(),
    )
    await pipeline.start()

    pipeline.feed_audio(np.zeros(16000, dtype=np.float32))
    await asyncio.sleep(0.3)
    await pipeline.stop()

    mock_translator.translate.assert_not_called()
    mock_tts.speak_sync.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_vad_filters_non_speech():
    from src.core.async_pipeline import AsyncTranslationPipeline

    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = False  # VAD rejeita o chunk

    mock_asr = MagicMock()
    mock_translator = MagicMock()
    mock_tts = MagicMock()

    pipeline = AsyncTranslationPipeline(
        asr=mock_asr,
        vad=mock_vad,
        translator=mock_translator,
        tts=mock_tts,
        config=_mock_config(),
    )
    await pipeline.start()

    pipeline.feed_audio(np.zeros(16000, dtype=np.float32))
    await asyncio.sleep(0.3)
    await pipeline.stop()

    mock_asr.transcribe.assert_not_called()
    mock_translator.translate.assert_not_called()
