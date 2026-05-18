import threading
from datetime import datetime


class TranslationLog:
    """
    Writes translated text to a file in the exact order it was spoken.
    Thread-safe: write() is called from the TTS worker thread.
    """

    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._file = open(path, "w", encoding="utf-8", buffering=1)

    @staticmethod
    def timestamped_path(prefix: str = "translation") -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{stamp}.txt"

    def write(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._file.write(f"[{timestamp}] {text}\n")

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
