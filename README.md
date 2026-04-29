# DirectTranslation

Tradução de voz em tempo real, 100% offline, com aceleração NVIDIA GPU.

Captura áudio do microfone, transcreve com Whisper, traduz com CTranslate2 e sintetiza a voz traduzida — tudo localmente, sem nenhuma chamada à internet em runtime.

```
Microfone → VAD (Silero) → ASR (Whisper) → Tradução (CTranslate2) → TTS (Coqui/Piper) → Caixa de som
```

---

## Pré-requisitos

| Requisito | Versão mínima |
|---|---|
| Python | 3.11+ |
| CUDA Toolkit | 11.8+ |
| Driver NVIDIA | compatível com CUDA instalado |
| VRAM recomendada | 4 GB+ |

> O app funciona em CPU, mas a latência aumenta significativamente.
> Para forçar CPU, ajuste `device: cpu` nos campos `asr.device`, `vad.device`, `translation.device` e `tts.device` em `config.yaml`.

---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/cflDevApps/DirectTranslation_py.git
cd DirectTranslation_py
```

### 2. Criar e ativar ambiente virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Instalar PyTorch com suporte CUDA

> **Este passo é obrigatório para usar a GPU.** `pip install torch` instala a versão CPU-only por padrão.

Escolha o comando conforme a versão do seu driver NVIDIA (`nvidia-smi` mostra na coluna "CUDA Version"):

| CUDA do driver | Comando |
|---|---|
| 12.4+ | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| 12.1 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

Não sabe qual versão usar? Execute o diagnóstico após a instalação (passo 5).

### 4. Instalar demais dependências

```bash
pip install -r requirements.txt
```

> **Nota sobre Coqui TTS:** se `pip install TTS` falhar, use a versão mantida pela comunidade:
> ```bash
> pip install coqui-tts
> ```

### 5. Verificar GPU (opcional mas recomendado)

```bash
python check_gpu.py
```

A saída mostra o status de cada componente e o comando exato de instalação caso algo esteja errado:

```
GPU          : NVIDIA GeForce RTX 4070
CUDA (driver): 12.4
PyTorch CUDA : True
CTranslate2  : 1 GPU(s) detectada(s)
```

### 6. Baixar e preparar os modelos (execução única)

```bash
python install_model.py
```

Este script realiza automaticamente:

| Etapa | O que faz |
|---|---|
| Argostranslate PT→EN | Baixa pacote de tradução (fallback CPU) |
| Whisper medium | Baixa modelo de transcrição |
| OPUS-MT PT→EN | Baixa e converte para CTranslate2 int8 (GPU) |
| Tokenizer | Salva arquivos do tokenizer em `src/models/opus-mt-pt-en/` |
| SileroVAD | Faz cache do modelo de detecção de voz |
| Coqui TTS | Baixa modelo `tts_models/en/ljspeech/vits` |

> Os modelos são salvos localmente. Após este passo, o app funciona **sem internet**.

---

## Configuração

Todos os parâmetros ficam em `config.yaml`. Os principais:

```yaml
audio:
  device: 2        # null = padrão do sistema; use o índice (ex: 2) para outro microfone

asr:
  model_size: medium  # tiny | base | small | medium | large

translation:
  source_language: pt  # idioma de entrada
  target_language: en  # idioma de saída

tts:
  engine: coqui        # coqui (GPU) | piper (CPU, mais leve)

vad:
  enabled: true        # false = usa apenas filtro de energia
  threshold: 0.5       # 0.0–1.0, maior = mais seletivo
```

### Listar dispositivos de áudio disponíveis

```python
from src.audio.device_utils import list_input_devices
print(list_input_devices())
```

---

## Como usar

### Modo GUI (padrão)

```bash
python app.py
```

A interface abre com os seguintes controles:

```
┌──────────────────────────────────────────────────┐
│  DirectTranslation                               │
├──────────────────────────────────────────────────┤
│  Dispositivo de áudio: [Padrão do sistema ▼]     │
│  Idioma: [Português ▼]  →  [English ▼]           │
├──────────────────────────────────────────────────┤
│  Palestrante:                                    │
│  ┌──────────────────────────────────────────┐    │
│  │ "Como você está hoje?"                   │    │
│  └──────────────────────────────────────────┘    │
│  Tradução:                                       │
│  ┌──────────────────────────────────────────┐    │
│  │ "How are you today?"                     │    │
│  └──────────────────────────────────────────┘    │
├──────────────────────────────────────────────────┤
│  GPU: NVIDIA RTX 4070 | VRAM: 3.8 / 8.0 GB      │
│  ASR: 118 ms  |  TR: 24 ms  |  TTS: 82 ms       │
│                   [▶ Iniciar]  [■ Parar]         │
└──────────────────────────────────────────────────┘
```

**Passos:**
1. Selecione o microfone e os idiomas desejados
2. Clique em **▶ Iniciar** — o pipeline carrega os modelos na GPU
3. Fale normalmente — a transcrição e a tradução aparecem em tempo real
4. Clique em **■ Parar** para encerrar

### Modo terminal (CLI)

```bash
python app.py --cli
```

Pressione `Ctrl+C` para encerrar. O relatório de latência média é exibido no terminal ao fechar.

---

## Pipeline assíncrono

O app usa três workers `asyncio` independentes em paralelo:

```
Worker 1 (ASR)         Worker 2 (Tradução)     Worker 3 (TTS)
──────────────         ───────────────────     ──────────────
await audio_q.get()    await text_q.get()      await translated_q.get()
  VAD (GPU)              CTranslate2 (GPU)       Coqui TTS (GPU)
  Whisper (GPU)                                  sd.play()
await text_q.put()     await translated_q.put()
```

Operações GPU rodam em `ThreadPoolExecutor` para não bloquear o event loop.
Frases consecutivas são processadas em paralelo — throughput ~8× maior que pipeline síncrono.

---

## Engines disponíveis

### ASR (transcrição)

| Engine | Aceleração | Qualidade | Configuração |
|---|---|---|---|
| faster-whisper | CUDA | Alta | `asr.model_size: medium` |

### Tradução

| Engine | Aceleração | Ativa quando |
|---|---|---|
| CTranslate2 + OPUS-MT | CUDA (int8) | Modelo existe em `src/models/opus-mt-pt-en/` |
| argostranslate | CPU | Fallback automático |

### TTS (síntese de voz)

| Engine | Aceleração | Configuração |
|---|---|---|
| Coqui VITS | CUDA | `tts.engine: coqui` |
| Piper | CPU | `tts.engine: piper` |

---

## Estrutura do projeto

```
DirectTranslation/
├── app.py                          # Entrada: GUI (padrão) ou --cli
├── config.yaml                     # Todos os parâmetros configuráveis
├── install_model.py                # Setup único — baixa e converte modelos
├── requirements.txt
├── pytest.ini
│
├── src/
│   ├── config.py                   # AppConfig (dataclasses tipadas)
│   │
│   ├── core/
│   │   ├── async_pipeline.py       # 3 workers asyncio + ThreadPoolExecutor GPU
│   │   ├── protocols.py            # Interfaces Protocol (ASR, Translator, TTS, VAD)
│   │   ├── benchmark.py            # LatencyTracker
│   │   └── logging.py              # setup_logging()
│   │
│   ├── asr/
│   │   └── whisper_engine.py       # faster-whisper + CUDA
│   │
│   ├── audio/
│   │   ├── async_audio_capture.py  # sounddevice → asyncio (thread-safe)
│   │   ├── hybrid_audio_pipeline.py # pipeline síncrono (legado / testes)
│   │   ├── silero_vad.py           # SileroVAD no GPU
│   │   └── device_utils.py         # detecção e listagem de dispositivos
│   │
│   ├── translation/
│   │   └── translation_engine.py   # CTranslate2 GPU + fallback argostranslate
│   │
│   ├── tts/
│   │   ├── coqui_tts_engine.py     # Coqui VITS GPU
│   │   └── piper_tts.py            # Piper CPU (fallback)
│   │
│   ├── ui/
│   │   ├── main_window.py          # MainWindow PySide6
│   │   ├── pipeline_worker.py      # QObject + asyncio em thread separada
│   │   └── gpu_utils.py            # métricas VRAM via torch.cuda
│   │
│   └── models/
│       ├── en_US-lessac-medium.onnx        # modelo Piper (TTS fallback)
│       └── opus-mt-pt-en/                  # modelo CTranslate2 (gerado por install_model.py)
│
└── tests/
    ├── test_translation_engine.py
    ├── test_whisper_engine.py
    ├── test_audio_pipeline.py
    ├── test_protocols.py
    └── test_async_pipeline.py
```

---

## Testes

```bash
pytest
```

Os testes usam mocks e **não requerem GPU nem modelos instalados**.

```
tests/test_translation_engine.py  — fallback argostranslate, vazio, exceção CT2
tests/test_whisper_engine.py      — join de segmentos, strip, language param
tests/test_audio_pipeline.py      — fila, chunk size, VAD injection
tests/test_protocols.py           — verifica Protocol em todos os componentes
tests/test_async_pipeline.py      — roteamento completo, VAD filter, skip vazio
```

---

## Latência esperada (GPU NVIDIA RTX 40xx)

| Componente | Latência aproximada |
|---|---|
| SileroVAD | ~3 ms |
| Whisper medium (CUDA) | ~120 ms |
| CTranslate2 OPUS-MT (int8) | ~25 ms |
| Coqui VITS (CUDA) | ~80 ms |
| **Pipeline completo (1ª frase)** | **~230 ms** |

Com frases consecutivas e os 3 workers em paralelo, o throughput sustentado é de aproximadamente **1 frase a cada 125 ms**.

---

## Licença

MIT License
