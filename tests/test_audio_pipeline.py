import pytest
import numpy as np
from unittest.mock import MagicMock
from src.audio.hybrid_audio_pipeline import HybridAudioPipeline


def _make_pipeline(**kwargs) -> tuple[HybridAudioPipeline, MagicMock]:
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = "texto"
    pipeline = HybridAudioPipeline(whisper_model=mock_asr, **kwargs)
    return pipeline, mock_asr


def test_get_text_returns_none_when_empty():
    pipeline, _ = _make_pipeline()
    assert pipeline.get_text() is None


def test_get_text_returns_queued_item():
    pipeline, _ = _make_pipeline()
    pipeline.text_queue.put("Olá mundo")
    assert pipeline.get_text() == "Olá mundo"
    assert pipeline.get_text() is None


def test_chunk_samples_calculated_from_duration():
    pipeline, _ = _make_pipeline(sample_rate=16000, chunk_duration=2.5, overlap_duration=0.25)
    assert pipeline.chunk_samples == 40000
    assert pipeline.overlap_samples == 4000


def test_vad_stored_when_provided():
    mock_vad = MagicMock()
    pipeline, _ = _make_pipeline(vad=mock_vad)
    assert pipeline.vad is mock_vad


def test_vad_none_by_default():
    pipeline, _ = _make_pipeline()
    assert pipeline.vad is None


def test_energy_threshold_default():
    pipeline, _ = _make_pipeline()
    assert pipeline.energy_threshold == 0.002
