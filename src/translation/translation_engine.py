import argostranslate.translate
import logging

logger = logging.getLogger("directtranslation.translation")


class TranslationEngine:
    def __init__(self, source_language: str, target_language: str):
        self.source_language = source_language
        self.target_language = target_language
        logger.info(f"TranslationEngine: {source_language} -> {target_language}")

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        return argostranslate.translate.translate(text, self.source_language, self.target_language)
