import os
import logging
import argostranslate.translate

logger = logging.getLogger("directtranslation.translation")


class TranslationEngine:
    def __init__(
        self,
        model_path: str,
        source_language: str,
        target_language: str,
        device: str = "cuda",
    ):
        self.source_language = source_language
        self.target_language = target_language
        self._ct2 = False

        if os.path.isdir(model_path):
            try:
                import ctranslate2
                from transformers import AutoTokenizer

                # Verifica e ajusta device se CUDA indisponível
                resolved_device = device
                if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
                    logger.warning(
                        "TranslationEngine: CUDA indisponivel para CTranslate2. Usando CPU."
                    )
                    resolved_device = "cpu"

                logger.info(
                    f"Carregando CTranslate2 de '{model_path}' ({resolved_device})..."
                )
                self._translator = ctranslate2.Translator(
                    model_path, device=resolved_device, inter_threads=2
                )
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._ct2 = True
                logger.info(
                    f"TranslationEngine (CTranslate2/{resolved_device}): "
                    f"{source_language} -> {target_language}"
                )
                return
            except Exception as e:
                logger.warning(f"CTranslate2 indisponivel: {e}. Usando argostranslate.")
        else:
            logger.warning(
                f"Modelo nao encontrado em '{model_path}'. "
                "Execute install_model.py para gerar. Usando argostranslate."
            )

        logger.info(
            f"TranslationEngine (argostranslate/CPU): {source_language} -> {target_language}"
        )

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        if not self._ct2:
            return argostranslate.translate.translate(
                text, self.source_language, self.target_language
            )

        try:
            token_ids = self._tokenizer.encode(text)
            input_tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
            results = self._translator.translate_batch([input_tokens])
            output_tokens = results[0].hypotheses[0]
            output_ids = self._tokenizer.convert_tokens_to_ids(output_tokens)
            return self._tokenizer.decode(output_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Erro CTranslate2, usando fallback: {e}")
            return argostranslate.translate.translate(
                text, self.source_language, self.target_language
            )
