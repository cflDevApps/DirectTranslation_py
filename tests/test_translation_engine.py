import pytest
from unittest.mock import patch, MagicMock
from src.translation.translation_engine import TranslationEngine


def _make_engine_fallback() -> TranslationEngine:
    with patch("os.path.isdir", return_value=False):
        return TranslationEngine(
            model_path="nonexistent",
            source_language="pt",
            target_language="en",
        )


def test_translate_empty_string_returns_empty():
    engine = _make_engine_fallback()
    assert engine.translate("") == ""


def test_translate_whitespace_only_returns_empty():
    engine = _make_engine_fallback()
    assert engine.translate("   ") == ""


def test_translate_uses_argostranslate_fallback():
    engine = _make_engine_fallback()
    with patch("argostranslate.translate.translate", return_value="Hello") as mock_tr:
        result = engine.translate("Olá")
    assert result == "Hello"
    mock_tr.assert_called_once_with("Olá", "pt", "en")


def test_translate_fallback_when_model_dir_missing():
    engine = _make_engine_fallback()
    assert engine._ct2 is False


def test_translate_ct2_falls_back_on_exception():
    with patch("os.path.isdir", return_value=True):
        with patch("ctranslate2.Translator") as MockCT2:
            with patch("transformers.AutoTokenizer.from_pretrained") as MockTok:
                engine = TranslationEngine(
                    model_path="fake/path",
                    source_language="pt",
                    target_language="en",
                )
                engine._ct2 = True
                engine._translator = MockCT2.return_value
                # Força exceção no translator
                MockCT2.return_value.translate_batch.side_effect = RuntimeError("GPU error")
                engine._tokenizer = MockTok.return_value
                MockTok.return_value.encode.return_value = [1, 2, 3]
                MockTok.return_value.convert_ids_to_tokens.return_value = ["a", "b", "c"]

    with patch("argostranslate.translate.translate", return_value="fallback") as mock_tr:
        result = engine.translate("Olá mundo")

    assert result == "fallback"
    mock_tr.assert_called_once_with("Olá mundo", "pt", "en")
