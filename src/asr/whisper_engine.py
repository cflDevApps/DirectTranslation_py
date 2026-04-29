from faster_whisper import WhisperModel
import logging

logger = logging.getLogger("directtranslation.asr")

_COMPUTE_TYPE_CPU = "int8"
_CUDA_DLL_KEYWORDS = ("dll", "cublas", "cannot be loaded", "cudnn", "cufft", "curand")


def _is_cuda_runtime_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(kw in msg for kw in _CUDA_DLL_KEYWORDS)


class WhisperEngine:
    def __init__(
        self,
        model_size: str = "medium",
        source_language: str = "pt",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        cpu_threads: int = 4,
        num_workers: int = 2,
    ):
        self.source_language = source_language
        self._model_size = model_size
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers

        resolved_device = device
        resolved_compute = compute_type
        if device == "cuda":
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() == 0:
                    raise RuntimeError("Nenhuma GPU CUDA detectada pelo CTranslate2.")
            except Exception as e:
                logger.warning(f"Whisper: CUDA indisponivel ({e}). Usando CPU.")
                resolved_device = "cpu"
                resolved_compute = _COMPUTE_TYPE_CPU

        logger.info(f"Carregando Whisper ({model_size}, {resolved_device}, {resolved_compute})...")
        self.model = WhisperModel(
            model_size,
            device=resolved_device,
            compute_type=resolved_compute,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        logger.info("Whisper carregado.")

    def _reload_cpu(self):
        """Recria o modelo em CPU quando CUDA runtime falha na primeira inferência."""
        logger.warning(
            "Whisper: CUDA DLL ausente (cublas/cudnn). "
            "Instale o CUDA Toolkit 12.x para GPU. Recarregando em CPU..."
        )
        self.model = WhisperModel(
            self._model_size,
            device="cpu",
            compute_type=_COMPUTE_TYPE_CPU,
            cpu_threads=self._cpu_threads,
            num_workers=self._num_workers,
        )
        logger.info("Whisper recarregado em CPU.")

    def transcribe(self, audio) -> str:
        try:
            segments, _ = self.model.transcribe(audio, language=self.source_language)
            return "".join(seg.text for seg in segments).strip()
        except RuntimeError as e:
            if _is_cuda_runtime_error(e):
                self._reload_cpu()
                segments, _ = self.model.transcribe(audio, language=self.source_language)
                return "".join(seg.text for seg in segments).strip()
            raise
