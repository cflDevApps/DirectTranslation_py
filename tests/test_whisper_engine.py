import pytest
import numpy as np
from unittest.mock import MagicMock, patch


def _make_segment(text: str):
    seg = MagicMock()
    seg.text = text
    return seg


@patch("faster_whisper.WhisperModel")
def test_transcribe_joins_segments(MockModel):
    from src.asr.whisper_engine import WhisperEngine

    MockModel.return_value.transcribe.return_value = (
        [_make_segment("Olá "), _make_segment("mundo")],
        MagicMock(),
    )
    engine = WhisperEngine(model_size="tiny", device="cpu", compute_type="int8")
    result = engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert result == "Olá mundo"


@patch("faster_whisper.WhisperModel")
def test_transcribe_strips_whitespace(MockModel):
    from src.asr.whisper_engine import WhisperEngine

    MockModel.return_value.transcribe.return_value = (
        [_make_segment("  texto com espacos  ")],
        MagicMock(),
    )
    engine = WhisperEngine(model_size="tiny", device="cpu", compute_type="int8")
    result = engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert result == "texto com espacos"


@patch("faster_whisper.WhisperModel")
def test_transcribe_empty_segments_returns_empty(MockModel):
    from src.asr.whisper_engine import WhisperEngine

    MockModel.return_value.transcribe.return_value = ([], MagicMock())
    engine = WhisperEngine(model_size="tiny", device="cpu", compute_type="int8")
    result = engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert result == ""


@patch("faster_whisper.WhisperModel")
def test_source_language_passed_to_model(MockModel):
    from src.asr.whisper_engine import WhisperEngine

    MockModel.return_value.transcribe.return_value = ([], MagicMock())
    engine = WhisperEngine(model_size="tiny", device="cpu", compute_type="int8", source_language="es")
    engine.transcribe(np.zeros(16000, dtype=np.float32))
    _, kwargs = MockModel.return_value.transcribe.call_args
    assert kwargs.get("language") == "es"
