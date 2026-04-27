import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    device: Optional[int]
    sample_rate: int
    chunk_duration: float
    overlap_duration: float
    energy_threshold: float
    silence_timeout: float


@dataclass
class ASRConfig:
    model_size: str
    source_language: str
    device: str
    compute_type: str
    cpu_threads: int
    num_workers: int


@dataclass
class TranslationConfig:
    source_language: str
    target_language: str


@dataclass
class TTSConfig:
    model_path: str
    piper_path: str


@dataclass
class GPUConfig:
    device: str
    fallback_to_cpu: bool


@dataclass
class AppConfig:
    audio: AudioConfig
    asr: ASRConfig
    translation: TranslationConfig
    tts: TTSConfig
    gpu: GPUConfig

    @classmethod
    def from_file(cls, path: str = "config.yaml") -> "AppConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            audio=AudioConfig(**data["audio"]),
            asr=ASRConfig(**data["asr"]),
            translation=TranslationConfig(**data["translation"]),
            tts=TTSConfig(**data["tts"]),
            gpu=GPUConfig(**data["gpu"]),
        )
