from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class ASREngine(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


@runtime_checkable
class TranslatorEngine(Protocol):
    def translate(self, text: str) -> str: ...


@runtime_checkable
class TTSEngine(Protocol):
    def speak(self, text: str) -> None: ...
    def speak_sync(self, text: str) -> None: ...
    def stop(self) -> None: ...


@runtime_checkable
class VADEngine(Protocol):
    def is_speech(self, audio: np.ndarray) -> bool: ...
