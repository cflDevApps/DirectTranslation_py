import time
import logging
from collections import deque

logger = logging.getLogger("directtranslation.benchmark")


class LatencyTracker:
    def __init__(self, window: int = 10):
        self._samples: dict[str, deque] = {}
        self._starts: dict[str, float] = {}
        self._window = window

    def start(self, stage: str):
        self._starts[stage] = time.perf_counter()

    def end(self, stage: str) -> float:
        if stage not in self._starts:
            return 0.0
        ms = (time.perf_counter() - self._starts.pop(stage)) * 1000
        if stage not in self._samples:
            self._samples[stage] = deque(maxlen=self._window)
        self._samples[stage].append(ms)
        return ms

    def report(self):
        if not self._samples:
            return
        logger.info("── Latência média (últimas %d amostras) ──", self._window)
        for stage, samples in self._samples.items():
            avg = sum(samples) / len(samples)
            logger.info("  %-15s %5.0f ms", stage, avg)
