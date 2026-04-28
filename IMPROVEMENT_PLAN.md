# Plano de Melhorias — DirectTranslation

> **Análise por:** Senior Software Engineer  
> **Data:** 2026-04-28  
> **Escopo:** Melhorias arquiteturais e de tecnologia, mantendo operação 100% local (offline), com aceleração NVIDIA GPU

---

## Progresso Geral

> Atualizar os checkboxes conforme cada item for concluído.

### Legenda
- `[x]` Concluído — `[ ]` Pendente — `[~]` Em progresso

---

### Fundacao — MVP Existente
- [x] Pipeline funcional ponta a ponta (Mic → ASR → Translate → TTS → Speaker)
- [x] WhisperEngine com CUDA (`int8_float16`)
- [x] Worker thread no PiperTTS (fila + daemon thread)
- [x] Back-pressure na fila do TTS (`put_nowait` com descarte)
- [x] VAD por energia como fallback básico

---

### Fase 1 — Estabilização
- [x] Criar `config.yaml` com todos os parâmetros hardcoded extraídos
- [x] Criar `src/config.py` (`AppConfig` dataclass)
- [x] Corrigir discrepância `small` vs `medium` no Whisper
- [x] Adicionar logging estruturado (`src/core/logging.py`)
- [x] Adicionar tratamento de exceção no loop principal de `app.py`
- [x] Implementar `src/translation/translation_engine.py` encapsulando `argostranslate`

---

### Fase 2 — GPU Completo
- [x] Integrar `SileroVAD` com CUDA (`src/audio/silero_vad.py`)
- [x] Substituir VAD por energia pelo SileroVAD no `HybridAudioPipeline`
- [x] Converter modelo OPUS-MT para CTranslate2 (script em `install_model.py`)
- [x] Migrar `TranslationEngine` para CTranslate2 + CUDA
- [x] Implementar `CoquiTTSEngine` com GPU (`src/tts/coqui_tts_engine.py`)
- [x] Adicionar `speak_sync()` ao engine TTS (para uso com `run_in_executor`)
- [x] Mover `piper_tts.py` para `src/tts/piper_tts.py` como fallback
- [x] Benchmark latência ponta a ponta (antes vs depois)

---

### Fase 3 — Arquitetura Limpa
- [x] Criar `src/core/protocols.py` com interfaces `Protocol` (ASR, Translator, TTS, VAD)
- [x] Implementar detecção automática de device de áudio (`src/audio/device_utils.py`)
- [x] Criar `src/core/async_pipeline.py` com 3 workers assíncronos
- [x] Criar `src/audio/async_audio_capture.py` (bridge `sounddevice` → `asyncio`)
- [x] Refatorar `app.py` para `asyncio.run(main())`
- [x] Integrar tamanhos de fila e parâmetros async no `config.yaml`
- [x] Adicionar métricas de latência por estágio no logger
- [x] Testar shutdown graceful (`Ctrl+C` cancela tasks sem travar)
- [x] Testes unitários dos componentes isolados (ASR, Translator, TTS, pipeline async)

---

### Fase 4 — UI com PySide6
- [ ] Criar `src/ui/main_window.py` com layout básico (wireframe da seção 4.5)
- [ ] Seleção de dispositivo de áudio via combo box
- [ ] Seleção dinâmica de idioma origem / destino
- [ ] Painel de texto: Palestrante e Tradução em tempo real
- [ ] Painel de métricas: latência por estágio + uso de VRAM
- [ ] Botões Iniciar / Parar com controle do pipeline
- [ ] Integrar UI com `AsyncTranslationPipeline` via signals PySide6

---

## 1. Diagnóstico do MVP

### Pontos positivos
- Pipeline funcional ponta a ponta (Mic → ASR → Translate → TTS → Speaker)
- Uso de CUDA no Whisper (`int8_float16`) — boa decisão de performance
- Worker thread no TTS evita bloqueio do loop principal
- VAD por energia funciona como fallback simples

### Débitos técnicos críticos
| Problema | Impacto | Prioridade |
|---|---|---|
| Translation feita diretamente no `app.py` sem abstração | Dificulta troca de motor | Alta |
| Idiomas hardcoded (`pt` → `en`) | Sem flexibilidade | Alta |
| Device de áudio hardcoded (`device=2`) | Quebra em outras máquinas | Alta |
| VAD por energia simples | Alta taxa de falsos positivos/negativos | Alta |
| Piper spawnado como subprocess por chamada | Latência alta, sem GPU | Média |
| Sem configuração externa (YAML/JSON) | Parâmetros espalhados no código | Média |
| Módulos `translation/`, `ui/` vazios | Estrutura prometida não implementada | Média |
| Sem logging estruturado | Debug impossível em produção | Média |
| Discrepância de modelo: `small` em app.py vs `medium` em install_model.py | Comportamento inesperado | Baixa |
| Sem tratamento de exceção no loop principal | Crash silencioso | Baixa |

---

## 2. Mapa de Uso da GPU (estado atual vs meta)

```
COMPONENTE          ATUAL           META
──────────────────────────────────────────────────────
VAD                 CPU (energia)   GPU (Silero VAD via Torch)
Whisper ASR         GPU ✅          GPU ✅ (manter + tuning)
Tradução            CPU ❌          GPU (CTranslate2 com CUDA)
TTS                 CPU ❌          GPU (Coqui XTTS / VITS)
Audio I/O           CPU ✅          CPU ✅ (sounddevice, correto)
```

---

## 3. Melhorias de Tecnologia

### 3.1 VAD — Silero VAD no GPU (dependência já instalada)

O `requirements.txt` já tem `silero-vad>=6.2.1` mas não está sendo usado.  
Silero VAD é um modelo PyTorch que roda em CUDA, muito mais preciso que threshold de energia.

```python
# src/audio/silero_vad.py (novo)
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

class SileroVAD:
    def __init__(self, device="cuda"):
        self.model = load_silero_vad()
        self.model.to(device)
        self.device = device
        self.sample_rate = 16000

    def is_speech(self, audio_np: np.ndarray) -> bool:
        tensor = torch.from_numpy(audio_np).float().to(self.device)
        timestamps = get_speech_timestamps(tensor, self.model, sampling_rate=self.sample_rate)
        return len(timestamps) > 0
```

**Ganho:** Elimina falsos positivos de ruído de fundo, reduz envio de áudio sem fala para o Whisper.

---

### 3.2 ASR — Faster-Whisper com otimizações GPU

O modelo já usa CUDA mas pode ser melhorado:

```python
# src/asr/whisper_engine.py (melhorado)
class WhisperEngine:
    def __init__(self, model_size="medium", source_lang="pt"):
        self.source_lang = source_lang
        self.model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="int8_float16",  # mantém — ótimo custo/benefício
            cpu_threads=4,                # reduzir, GPU faz o trabalho pesado
            num_workers=2,                # aumentar para paralelismo de batches
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.source_lang,
            beam_size=3,               # beam_size=1 para latência mínima
            vad_filter=True,           # VAD embutido do faster-whisper como 2ª camada
            vad_parameters={"min_silence_duration_ms": 400},
        )
        return "".join(seg.text for seg in segments).strip()
```

**Ganho:** `beam_size=3` melhora qualidade sem custo relevante; `vad_filter=True` como segunda camada de filtragem.

---

### 3.3 Tradução — CTranslate2 com CUDA (substituir argostranslate direto)

`argostranslate` usa `CTranslate2` internamente, mas não expõe controle de device.  
Usar `CTranslate2` diretamente com modelos OPUS-MT ou NLLB garante execução em GPU.

**Opção A — Helsinki OPUS-MT (mais leve, PT→EN excelente):**

```bash
# instalar conversor
pip install ctranslate2 transformers sentencepiece

# converter modelo para CTranslate2 uma vez
ct2-opus-mt-converter --model Helsinki-NLP/opus-mt-tc-big-pt-en \
                      --output_dir src/models/opus-mt-pt-en \
                      --quantization int8
```

```python
# src/translation/translation_engine.py (novo)
import ctranslate2
from transformers import AutoTokenizer

class TranslationEngine:
    def __init__(self, model_path: str, src_lang: str, tgt_lang: str, device="cuda"):
        self.translator = ctranslate2.Translator(model_path, device=device, inter_threads=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def translate(self, text: str) -> str:
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(text)
        )
        result = self.translator.translate_batch([tokens])
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(result[0].hypotheses[0]),
            skip_special_tokens=True
        )
```

**Opção B — NLLB-200 (suporta 200 idiomas, ideal para expansão futura):**
Usar `facebook/nllb-200-distilled-600M` convertido via CTranslate2 — boa qualidade, ~1.2GB.

**Recomendação:** Opção A para PT→EN com máxima qualidade; Opção B se quiser suporte multilíngue futuro.

**Ganho:** Tradução passa de CPU para GPU, latência cai de ~300ms para ~30ms.

---

### 3.4 TTS — Substituir Piper subprocess por Coqui XTTS2 (GPU)

Piper é excelente mas não suporta GPU. XTTS2 da Coqui roda nativamente em CUDA com qualidade superior.

```python
# src/tts/coqui_tts_engine.py (novo)
from TTS.api import TTS
import torch
import sounddevice as sd

class CoquiTTSEngine:
    def __init__(self, model_name="tts_models/en/ljspeech/vits", device="cuda"):
        self.tts = TTS(model_name).to(device)
        self.sample_rate = 22050
        self.queue = Queue(maxsize=5)
        self._start_worker()

    def _start_worker(self):
        Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            text = self.queue.get()
            if text is None:
                break
            wav = self.tts.tts(text=text)
            sd.play(np.array(wav, dtype=np.float32), self.sample_rate)
            sd.wait()

    def speak(self, text: str):
        if not text or not text.strip():
            return
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except:
                pass
        self.queue.put(text)
```

**Alternativa leve:** Manter Piper mas como servidor persistente (daemon) em vez de subprocess por chamada:
```bash
# Piper como servidor HTTP local — elimina overhead de spawn
piper-server --model src/models/en_US-lessac-medium.onnx --port 5003
```

**Ganho:** Latência TTS cai de ~800ms (spawn + execução) para ~50-100ms (GPU).

---

## 4. Melhorias Arquiteturais

### 4.1 Sistema de Configuração (YAML)

Centralizar todos os parâmetros que hoje estão hardcoded:

```yaml
# config.yaml
audio:
  device: null          # null = padrão do sistema
  sample_rate: 16000
  energy_threshold: 0.0015
  silence_timeout: 1.0
  chunk_seconds: 2.5
  overlap_seconds: 0.25

asr:
  model_size: medium
  source_language: pt
  beam_size: 3

translation:
  model_path: src/models/opus-mt-pt-en
  source_language: pt
  target_language: en
  device: cuda

tts:
  engine: coqui          # coqui | piper
  model: tts_models/en/ljspeech/vits
  device: cuda
  piper_model: src/models/en_US-lessac-medium.onnx

gpu:
  device: cuda
  fallback_to_cpu: true  # fallback se CUDA não disponível
```

```python
# src/config.py
import yaml
from dataclasses import dataclass

@dataclass
class AppConfig:
    audio: dict
    asr: dict
    translation: dict
    tts: dict
    gpu: dict

    @classmethod
    def from_file(cls, path="config.yaml") -> "AppConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

---

### 4.2 Abstração de Pipeline (Interface Protocol)

Definir contratos claros entre os componentes:

```python
# src/core/protocols.py
from typing import Protocol
import numpy as np

class ASREngine(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...

class TranslatorEngine(Protocol):
    def translate(self, text: str) -> str: ...

class TTSEngine(Protocol):
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...
```

Isso permite trocar qualquer componente sem alterar o pipeline principal.

---

### 4.3 Pipeline Principal Refatorado

```python
# app.py (refatorado)
from src.config import AppConfig
from src.core.pipeline import TranslationPipeline

def start():
    config = AppConfig.from_file("config.yaml")
    pipeline = TranslationPipeline(config)
    pipeline.run()

# src/core/pipeline.py
class TranslationPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self._setup_components()

    def _setup_components(self):
        device = self.config.gpu["device"]
        
        self.asr = WhisperEngine(
            model_size=self.config.asr["model_size"],
            source_lang=self.config.asr["source_language"],
        )
        self.translator = TranslationEngine(
            model_path=self.config.translation["model_path"],
            src_lang=self.config.translation["source_language"],
            tgt_lang=self.config.translation["target_language"],
            device=device,
        )
        self.tts = CoquiTTSEngine(device=device)
        self.audio = HybridAudioPipeline(
            asr=self.asr,
            vad=SileroVAD(device=device),
            **self.config.audio,
        )

    def run(self):
        self.audio.start()
        logger.info("Pipeline iniciado. Fale algo...")
        try:
            while True:
                text = self.audio.get_text()
                if text:
                    logger.info(f"Palestrante: {text}")
                    translated = self.translator.translate(text)
                    logger.info(f"Tradução: {translated}")
                    self.tts.speak(translated)
        except KeyboardInterrupt:
            self.audio.stop()
            logger.info("Pipeline encerrado.")
```

---

### 4.4 Logging Estruturado

```python
# src/core/logging.py
import logging
import sys

def setup_logging(level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("directtranslation")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
```

---

### 4.5 UI com PySide6 (módulo já no requirements)

Interface mínima mas funcional:

```
┌─────────────────────────────────────────────┐
│  DirectTranslation                    [×]   │
├─────────────────────────────────────────────┤
│  🎤 Dispositivo: [Microfone padrão    ▼]    │
│  Idioma entrada: [Português  ▼]  →  [Inglês ▼]│
├─────────────────────────────────────────────┤
│  Palestrante:                               │
│  ┌─────────────────────────────────────┐    │
│  │ "Como você está hoje?"              │    │
│  └─────────────────────────────────────┘    │
│  Tradução:                                  │
│  ┌─────────────────────────────────────┐    │
│  │ "How are you today?"                │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│  GPU: NVIDIA RTX 4070 | VRAM: 4.2/8.0 GB   │
│  Latência: ASR 120ms | TR 28ms | TTS 85ms  │
│               [■ PARAR]  [▶ INICIAR]        │
└─────────────────────────────────────────────┘
```

---

## 5. Roadmap de Implementação

### Fase 1 — Estabilização (1–2 dias)
- [ ] Criar `config.yaml` com todos os parâmetros hardcoded
- [ ] Corrigir discrepância `small` vs `medium` no Whisper
- [ ] Adicionar logging estruturado
- [ ] Adicionar tratamento de exceção no loop principal
- [ ] Implementar `src/translation/translation_engine.py` com argostranslate encapsulado

### Fase 2 — GPU Completo (2–3 dias)
- [ ] Integrar Silero VAD com CUDA (substituir VAD por energia)
- [ ] Migrar tradução para CTranslate2 + OPUS-MT em GPU
- [ ] Implementar Coqui XTTS2 ou Piper daemon para TTS em GPU
- [ ] Benchmark latência ponta a ponta antes e depois

### Fase 3 — Arquitetura Limpa (2–3 dias)
- [ ] Definir `Protocol` interfaces em `src/core/protocols.py`
- [ ] Refatorar `app.py` usando `TranslationPipeline`
- [ ] Implementar detecção automática de device de áudio
- [ ] Testes unitários dos componentes isolados

### Fase 4 — UI (3–5 dias)
- [ ] Implementar GUI PySide6 conforme wireframe
- [ ] Painel de métricas em tempo real (latência, VRAM)
- [ ] Seleção dinâmica de idiomas via UI
- [ ] Seleção de dispositivo de áudio via UI

---

## 6. Estimativa de Latência (meta pós-melhorias)

| Componente | Atual (estimado) | Meta |
|---|---|---|
| VAD | ~10ms CPU | ~3ms GPU |
| Whisper ASR (small) | ~200ms GPU | ~120ms GPU (medium + tuning) |
| Tradução | ~300ms CPU | ~25ms GPU |
| TTS | ~800ms (subprocess) | ~80ms GPU |
| **Total ponta a ponta** | **~1,3s** | **~230ms** |

---

## 7. Dependências a Adicionar

```txt
# requirements.txt — adições
TTS>=0.22.0          # Coqui XTTS2 (se adotar GPU TTS)
ctranslate2>=4.3     # já existe — confirmar uso direto
transformers>=4.40   # tokenizers OPUS-MT / NLLB
pyyaml>=6.0          # config.yaml
```

**Nenhuma dependência requer internet em produção** — todos os modelos são baixados uma vez pelo `install_model.py` e usados localmente.

---

## 8. Estrutura de Diretórios Final

```
DirectTranslation/
├── app.py                    # entry point enxuto
├── config.yaml               # todos os parâmetros configuráveis
├── install_model.py          # setup único (baixa modelos)
├── requirements.txt
├── src/
│   ├── core/
│   │   ├── pipeline.py       # orquestrador principal
│   │   ├── protocols.py      # interfaces Protocol
│   │   └── logging.py        # configuração de log
│   ├── asr/
│   │   └── whisper_engine.py # faster-whisper + CUDA
│   ├── audio/
│   │   ├── hybrid_audio_pipeline.py
│   │   └── silero_vad.py     # VAD neural no GPU
│   ├── translation/
│   │   └── translation_engine.py  # CTranslate2 + CUDA
│   ├── tts/
│   │   ├── coqui_tts_engine.py    # Coqui XTTS2 GPU
│   │   └── piper_tts.py      # manter como fallback
│   ├── ui/
│   │   └── main_window.py    # PySide6 GUI
│   ├── config.py             # dataclass de configuração
│   └── models/               # binários dos modelos (local)
│       ├── en_US-lessac-medium.onnx
│       └── opus-mt-pt-en/    # modelo CTranslate2
└── IMPROVEMENT_PLAN.md
```

---

## 9. Pipeline Assíncrono (Entrada → Tradução → Saída)

### 9.1 Problema do fluxo síncrono atual

O loop atual em `app.py` é **sequencial bloqueante**. Cada etapa espera a anterior terminar antes de começar:

```
Mic ──► [VAD+ASR ~200ms] ──► [Translate ~300ms] ──► [TTS ~800ms] ──► Speaker
         BLOQUEIA              BLOQUEIA               BLOQUEIA
```

Enquanto o Whisper processa, o microfone não escuta. Enquanto a tradução roda, o Whisper não transcreve. O resultado é que o pipeline opera no **pior caso acumulativo** de todas as latências.

---

### 9.2 Modelo Produtor-Consumidor com Filas Assíncronas

A solução é separar cada estágio em um **worker assíncrono independente**, conectados por filas (`asyncio.Queue`) com tamanho limitado (back-pressure):

```
┌──────────────┐     Queue(4)     ┌──────────────┐     Queue(4)     ┌──────────────┐     Queue(4)     ┌──────────────┐
│  AudioCapture│ ──── audio[] ──► │  ASR Worker  │ ──── text ─────► │  Translator  │ ──── text ─────► │  TTS Worker  │
│  (Producer)  │                  │  (Consumer/  │                  │  (Consumer/  │                  │  (Consumer)  │
│  sounddevice │                  │   Producer)  │                  │   Producer)  │                  │  GPU synth   │
└──────────────┘                  └──────────────┘                  └──────────────┘                  └──────────────┘
    Thread                          ThreadPool                        ThreadPool                         ThreadPool
    callback                        (GPU bound)                       (GPU bound)                        (GPU bound)
```

**Princípio-chave:** operações de GPU são bloqueantes (`faster-whisper`, `ctranslate2`, TTS síntese). Para não bloquear o event loop do `asyncio`, cada chamada GPU roda em `ThreadPoolExecutor` via `run_in_executor`.

---

### 9.3 Implementação — `src/core/async_pipeline.py`

```python
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger("directtranslation.pipeline")


@dataclass
class AudioChunk:
    data: np.ndarray
    energy: float


@dataclass
class TranscribedText:
    text: str
    latency_ms: float


@dataclass
class TranslatedText:
    original: str
    translated: str
    latency_ms: float


class AsyncTranslationPipeline:
    """
    Três workers independentes ligados por asyncio.Queue com back-pressure.
    Operações GPU delegadas a ThreadPoolExecutor — nunca bloqueiam o event loop.
    """

    def __init__(self, asr, translator, tts, vad, config: dict):
        self._asr = asr
        self._translator = translator
        self._tts = tts
        self._vad = vad
        self._config = config

        # Filas com tamanho máximo = back-pressure automático
        self._audio_queue: asyncio.Queue[Optional[AudioChunk]] = asyncio.Queue(maxsize=4)
        self._text_queue: asyncio.Queue[Optional[TranscribedText]] = asyncio.Queue(maxsize=8)
        self._translated_queue: asyncio.Queue[Optional[TranslatedText]] = asyncio.Queue(maxsize=8)

        # Pool dedicado para operações GPU (evita contenção com I/O threads)
        self._gpu_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gpu")

        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ─── Injeção de áudio (chamado pelo callback do sounddevice) ───────────

    def feed_audio(self, audio: np.ndarray):
        """Thread-safe: sounddevice callback injeta chunks na fila async."""
        chunk = AudioChunk(data=audio, energy=float(np.sqrt(np.mean(audio ** 2))))
        try:
            # put_nowait: descarta se a fila estiver cheia (back-pressure real-time)
            self._audio_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            logger.debug("Audio queue cheia — chunk descartado (pipeline congestionado)")

    # ─── Worker 1: ASR ─────────────────────────────────────────────────────

    async def _asr_worker(self):
        """Consome AudioChunk, produz TranscribedText."""
        loop = asyncio.get_running_loop()

        while self._running:
            chunk = await self._audio_queue.get()
            if chunk is None:
                await self._text_queue.put(None)  # propaga shutdown
                break

            # Filtro VAD rápido antes de chamar GPU
            if not await loop.run_in_executor(self._gpu_executor, self._vad.is_speech, chunk.data):
                self._audio_queue.task_done()
                continue

            import time
            t0 = time.perf_counter()
            text = await loop.run_in_executor(self._gpu_executor, self._asr.transcribe, chunk.data)
            latency = (time.perf_counter() - t0) * 1000

            if text:
                logger.info(f"[ASR {latency:.0f}ms] {text}")
                await self._text_queue.put(TranscribedText(text=text, latency_ms=latency))

            self._audio_queue.task_done()

    # ─── Worker 2: Tradução ────────────────────────────────────────────────

    async def _translation_worker(self):
        """Consome TranscribedText, produz TranslatedText."""
        loop = asyncio.get_running_loop()

        while self._running:
            item = await self._text_queue.get()
            if item is None:
                await self._translated_queue.put(None)  # propaga shutdown
                break

            import time
            t0 = time.perf_counter()
            translated = await loop.run_in_executor(
                self._gpu_executor, self._translator.translate, item.text
            )
            latency = (time.perf_counter() - t0) * 1000

            logger.info(f"[TR {latency:.0f}ms] {translated}")
            await self._translated_queue.put(
                TranslatedText(original=item.text, translated=translated, latency_ms=latency)
            )
            self._text_queue.task_done()

    # ─── Worker 3: TTS ────────────────────────────────────────────────────

    async def _tts_worker(self):
        """Consome TranslatedText, sintetiza e toca áudio."""
        loop = asyncio.get_running_loop()

        while self._running:
            item = await self._translated_queue.get()
            if item is None:
                break

            import time
            t0 = time.perf_counter()
            # síntese GPU + playback em executor (sd.wait() bloqueia)
            await loop.run_in_executor(self._gpu_executor, self._tts.speak_sync, item.translated)
            latency = (time.perf_counter() - t0) * 1000

            logger.info(f"[TTS {latency:.0f}ms] '{item.translated}'")
            self._translated_queue.task_done()

    # ─── Ciclo de vida ────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._tasks = [
            asyncio.create_task(self._asr_worker(),         name="asr"),
            asyncio.create_task(self._translation_worker(), name="translator"),
            asyncio.create_task(self._tts_worker(),         name="tts"),
        ]
        logger.info("Pipeline assíncrono iniciado (3 workers ativos)")

    async def stop(self):
        self._running = False
        await self._audio_queue.put(None)   # sentinel: encerra cadeia
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._gpu_executor.shutdown(wait=False)
        logger.info("Pipeline encerrado.")
```

---

### 9.4 Integração com `sounddevice` e `asyncio`

`sounddevice` usa callbacks em thread nativa — não é compatível diretamente com `asyncio`. O padrão correto usa `loop.call_soon_threadsafe`:

```python
# src/audio/async_audio_capture.py
import sounddevice as sd
import asyncio
import numpy as np


class AsyncAudioCapture:
    def __init__(self, pipeline: AsyncTranslationPipeline, sample_rate=16000, device=None):
        self._pipeline = pipeline
        self._sample_rate = sample_rate
        self._device = device
        self._loop: asyncio.AbstractEventLoop = None
        self._stream: sd.InputStream = None
        self._buffer = np.array([], dtype=np.float32)
        self._chunk_samples = int(sample_rate * 2.5)   # 2.5s por chunk

    def _callback(self, indata: np.ndarray, frames, time_info, status):
        audio = indata[:, 0].copy()
        self._buffer = np.concatenate([self._buffer, audio])

        if len(self._buffer) >= self._chunk_samples:
            chunk = self._buffer[:self._chunk_samples]
            self._buffer = self._buffer[int(self._chunk_samples * 0.9):]  # 10% overlap
            # injeta no pipeline de forma thread-safe
            self._loop.call_soon_threadsafe(self._pipeline.feed_audio, chunk)

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
```

---

### 9.5 Entry Point Assíncrono — `app.py` refatorado

```python
# app.py
import asyncio
from src.config import AppConfig
from src.core.async_pipeline import AsyncTranslationPipeline
from src.audio.async_audio_capture import AsyncAudioCapture
from src.asr.whisper_engine import WhisperEngine
from src.translation.translation_engine import TranslationEngine
from src.tts.coqui_tts_engine import CoquiTTSEngine
from src.audio.silero_vad import SileroVAD
from src.core.logging import setup_logging

logger = setup_logging()


async def main():
    config = AppConfig.from_file("config.yaml")
    device = config.gpu["device"]

    asr        = WhisperEngine(**config.asr, device=device)
    translator = TranslationEngine(**config.translation, device=device)
    tts        = CoquiTTSEngine(**config.tts, device=device)
    vad        = SileroVAD(device=device)

    pipeline = AsyncTranslationPipeline(asr, translator, tts, vad, config)
    capture  = AsyncAudioCapture(pipeline, **config.audio)

    await pipeline.start()
    capture.start(asyncio.get_running_loop())

    logger.info("Fale algo... (Ctrl+C para encerrar)")

    try:
        await asyncio.Event().wait()   # aguarda indefinidamente
    except asyncio.CancelledError:
        pass
    finally:
        capture.stop()
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

---

### 9.6 Diagrama de Concorrência

```
Thread: sounddevice callback
  │
  │ loop.call_soon_threadsafe(feed_audio)
  ▼
asyncio event loop (thread principal)
  │
  ├─ audio_queue ──────────────────────────────────────────────┐
  │                                                            │
  │  Task: asr_worker                                         │
  │  ├─ await audio_queue.get()        ← espera não bloqueante│
  │  └─ run_in_executor(GPU: Whisper)  ← GPU em thread pool   │
  │         │                                                  │
  ├─────────▼─────────────────────────────────────────────────┘
  │  text_queue
  │
  │  Task: translation_worker
  │  ├─ await text_queue.get()
  │  └─ run_in_executor(GPU: CTranslate2)
  │         │
  ├─────────▼
  │  translated_queue
  │
  │  Task: tts_worker
  │  ├─ await translated_queue.get()
  │  └─ run_in_executor(GPU: TTS synth + sd.play)

GPU ThreadPoolExecutor (max_workers=3)
  ├─ Thread 0: Whisper (exclusivo para ASR)
  ├─ Thread 1: CTranslate2 (exclusivo para tradução)
  └─ Thread 2: TTS synthesis + playback
```

**Por que `max_workers=3` e não mais?** Cada thread corresponde a um estágio que usa a GPU. Com mais threads concorrentes na mesma GPU, ocorre contenção de memória VRAM e troca de contexto CUDA — degradando performance em vez de melhorar.

---

### 9.7 Back-pressure e Descarte

| Fila | Tamanho máximo | Comportamento quando cheia |
|---|---|---|
| `audio_queue` | 4 chunks (~10s) | Descarta chunk mais novo (`put_nowait`) — microfone não trava |
| `text_queue` | 8 itens | `await put()` — ASR espera (pressão para trás) |
| `translated_queue` | 8 itens | `await put()` — tradução espera (pressão para trás) |

A `audio_queue` usa descarte para manter o sistema em tempo real: se o Whisper estiver congestionado, áudio antigo é descartado em vez de acumular. As demais filas usam back-pressure para não perder texto já transcrito.

---

### 9.8 Métricas de Latência — Ganho do Async

Com o pipeline assíncrono, os estágios rodam **em paralelo** para frases consecutivas:

```
Frase A:  [─── ASR 120ms ───][── TR 25ms ──][─────── TTS 80ms ───────]
Frase B:              [─────── ASR 120ms ───][── TR 25ms ──][─── TTS 80ms ───]
Frase C:                          [──── ASR 120ms ───][── TR 25ms ──][─ TTS 80ms ─]
          ◄──────────────────────────────────────────────────────────────────────►
          Throughput: ~1 frase/125ms  (vs ~1 frase/1300ms no pipeline síncrono)
```

**Latência da primeira frase:** ~225ms (igual ao pipeline síncrono)  
**Throughput sustentado:** ~8x maior com frases consecutivas

---

### 9.9 Checklist de Implementação — Fase Async

- [ ] Criar `src/core/async_pipeline.py` com os 3 workers
- [ ] Criar `src/audio/async_audio_capture.py` com bridge thread→asyncio
- [ ] Adicionar `speak_sync()` ao `CoquiTTSEngine` (versão bloqueante para `run_in_executor`)
- [ ] Refatorar `app.py` para `asyncio.run(main())`
- [ ] Integrar com `config.yaml` (tamanhos de fila configuráveis)
- [ ] Adicionar métricas de latência por estágio no logger
- [ ] Testar shutdown graceful com `Ctrl+C` (cancelamento de tasks)

---

*Todas as melhorias mantêm a premissa central: operação 100% offline, sem chamadas externas em runtime.*
