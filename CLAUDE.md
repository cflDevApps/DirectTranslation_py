# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Real-time, fully offline voice translation (Portuguese → English by default). It captures microphone audio, transcribes with Whisper (faster-whisper), translates with CTranslate2/OPUS-MT, and synthesizes speech with Coqui VITS or Piper TTS — all locally using GPU acceleration.

## Commands

### Run the application
```powershell
# GUI mode (default)
python app.py

# CLI mode
python app.py --cli
```

### One-time model setup (required before first run)
```powershell
python install_model.py
```

### Diagnostics
```powershell
python check_gpu.py          # Verify CUDA/PyTorch/CTranslate2 setup
python list_microphones.py   # List available audio input devices
```

### Tests
```powershell
pytest                        # Run all tests
pytest tests/test_async_pipeline.py                                    # Single file
pytest tests/test_async_pipeline.py::TestAsyncPipeline::test_routing  # Single test
```

## Architecture

### Data Flow

```
Microphone
  → AsyncAudioCapture (sounddevice callback → asyncio.Queue)
  → AsyncTranslationPipeline (3 parallel asyncio workers)
      Worker 1: SileroVAD filter → WhisperEngine.transcribe()
      Worker 2: TranslationEngine.translate()
      Worker 3: TTSEngine.generate() [GPU executor]
                ↘ _play_audio()      [playback executor — runs concurrently with next generate()]
  → Speaker
  → TranslationLog (appends translated text in audio-output order)
```

### Key Source Files

- **`src/core/async_pipeline.py`** — `AsyncTranslationPipeline`: three asyncio workers connected by `asyncio.Queue`. GPU-bound calls run in `_executor` (ThreadPoolExecutor); audio playback runs in a separate `_playback_executor` (1 thread) so TTS generation of chunk N+1 overlaps with playback of chunk N. Queues use `None` as a shutdown sentinel. Optional callbacks (`on_transcription`, `on_translation`, `on_tts_complete`) bridge to the UI.

- **`src/core/protocols.py`** — Runtime-checkable `Protocol` definitions for all engines. `TTSEngine` requires `generate(text) -> np.ndarray` (inference only, no playback), `speak_sync`, `speak`, and `stop`. Tests in `tests/test_protocols.py` verify conformance.

- **`src/asr/whisper_engine.py`** — Wraps `faster-whisper`. Uses `beam_size=1` (greedy decoding), `without_timestamps=True`, and `condition_on_previous_text=False` for minimum latency. Falls back from CUDA to CPU automatically.

- **`src/translation/translation_engine.py`** — Tries CTranslate2 + OPUS-MT (GPU int8, `inter_threads=1`) first; falls back to `argostranslate` (CPU) if the model directory is missing or init fails.

- **`src/tts/`** — Both `CoquiTTSEngine` (CUDA VITS) and `PiperTTS` (subprocess ONNX, CPU) expose a `generate(text) -> np.ndarray` method (inference only) and `SAMPLE_RATE: int = 22050` as a class attribute. `speak_sync` composes `generate` + `sd.play/wait`. The pipeline calls `generate` directly and handles playback itself via `_play_audio`.

- **`src/translation_log.py`** — `TranslationLog`: thread-safe file writer. `write(text)` is called from the TTS worker only after `sd.wait()` returns, guaranteeing order matches audio output. Each session creates a timestamped file (`translation_YYYYMMDD_HHMMSS.txt`). Use `TranslationLog.timestamped_path()` to generate the filename.

- **`src/audio/async_audio_capture.py`** — `sounddevice` callback injects audio into the asyncio loop via `loop.call_soon_threadsafe()`. Chunks overlap by 0.1 s; audio queue uses `put_nowait` (drops on full for backpressure).

- **`src/ui/`** — `PySide6` GUI. `PipelineWorker` runs the asyncio event loop in a `QThread` and bridges it to Qt Signals.

- **`src/config.py`** — `AppConfig` dataclass loaded from `config.yaml` via `AppConfig.from_file()`. Each engine receives its relevant config section at init.

### TTS Pipeline: generate vs. playback

The TTS worker uses two separate executors to overlap GPU inference with audio playback:

```
chunk N:   [generate 80ms] ──────────────── wait ── done
                            [play ~1500ms ──────────────]
chunk N+1:                  [generate 80ms] ─ wait ──── done
                                              [play ~1500ms ──...]
```

`generate()` runs in the shared GPU `_executor`. `_play_audio()` runs in `_playback_executor` (single thread, enforces serial playback). The worker awaits the previous playback future only when the next audio is already ready, so the speaker starts again with zero gap.

### Fallback Hierarchy

Each component independently resolves its device:

| Layer | Primary | Fallback |
|---|---|---|
| ASR | faster-whisper CUDA | faster-whisper CPU |
| Translation | CTranslate2 GPU int8 | argostranslate CPU |
| TTS | Coqui VITS CUDA | Piper ONNX CPU |

### Test Design

Tests mock all model/hardware calls with `unittest.mock`. No GPU or model files are needed to run the test suite. Async tests use `@pytest.mark.asyncio` (mode `auto` via `pytest.ini`).

## Configuration

Edit `config.yaml` to change audio device, language pair, model sizes, VAD threshold, and queue sizes. Key latency-sensitive parameters:

| Parameter | Value | Effect |
|---|---|---|
| `audio.chunk_duration` | 1.5 s | First-response latency floor |
| `audio.overlap_duration` | 0.1 s | Context continuity at chunk boundaries |
| `asr.num_workers` | 1 | Single stream needs only one worker |
| `pipeline.audio_queue_size` | 2 | Drops stale chunks early under load |

The `src/models/opus-mt-pt-en/` directory (created by `install_model.py`) holds the CTranslate2 translation model.
