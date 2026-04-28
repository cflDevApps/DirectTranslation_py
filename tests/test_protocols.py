import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.core.protocols import ASREngine, TranslatorEngine, TTSEngine, VADEngine


@patch("faster_whisper.WhisperModel")
def test_whisper_satisfies_asr_protocol(MockModel):
    from src.asr.whisper_engine import WhisperEngine
    engine = WhisperEngine(model_size="tiny", device="cpu", compute_type="int8")
    assert isinstance(engine, ASREngine)


def test_translation_engine_satisfies_translator_protocol():
    with patch("os.path.isdir", return_value=False):
        from src.translation.translation_engine import TranslationEngine
        engine = TranslationEngine(
            model_path="nonexistent",
            source_language="pt",
            target_language="en",
        )
    assert isinstance(engine, TranslatorEngine)


def test_piper_tts_satisfies_tts_protocol():
    with patch("threading.Thread"):
        from src.tts.piper_tts import PiperTTS
        tts = PiperTTS(model_path="model.onnx")
    assert isinstance(tts, TTSEngine)


def test_coqui_tts_satisfies_tts_protocol():
    with patch("TTS.api.TTS"), patch("threading.Thread"):
        from src.tts.coqui_tts_engine import CoquiTTSEngine
        tts = CoquiTTSEngine(model_name="model", device="cpu")
    assert isinstance(tts, TTSEngine)


def test_silero_vad_satisfies_vad_protocol():
    with patch("silero_vad.load_silero_vad", return_value=MagicMock()):
        from src.audio.silero_vad import SileroVAD
        vad = SileroVAD(device="cpu")
    assert isinstance(vad, VADEngine)
