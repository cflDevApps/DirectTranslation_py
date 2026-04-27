import os
import argostranslate.package
from faster_whisper import WhisperModel

# ── Argostranslate (PT→EN) ───────────────────────────────────────────────────
print("Instalando pacote argostranslate (pt->en)...")
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(lambda x: x.from_code == "pt" and x.to_code == "en", available_packages)
)
argostranslate.package.install_from_path(package_to_install.download())
print("Argostranslate instalado.")

# ── Whisper (medium) ─────────────────────────────────────────────────────────
print("Baixando modelo Whisper (medium)...")
WhisperModel("medium")
print("Whisper instalado.")

# ── OPUS-MT PT→EN via CTranslate2 ────────────────────────────────────────────
OUTPUT_DIR = "src/models/opus-mt-pt-en"
HF_MODEL = "Helsinki-NLP/opus-mt-pt-en"

if os.path.isdir(OUTPUT_DIR):
    print(f"Modelo CTranslate2 ja existe em '{OUTPUT_DIR}', pulando conversao.")
else:
    print(f"Convertendo {HF_MODEL} para CTranslate2 (int8)...")
    converted = False
    try:
        import ctranslate2.converters as ct2_conv

        ConverterClass = getattr(ct2_conv, "OpusMTConverter", None) or getattr(
            ct2_conv, "TransformersConverter", None
        )
        if ConverterClass is None:
            raise ImportError("Nenhum conversor CTranslate2 encontrado.")

        ConverterClass(HF_MODEL).convert(OUTPUT_DIR, quantization="int8", force=True)
        converted = True
        print(f"Modelo CTranslate2 salvo em '{OUTPUT_DIR}'.")
    except Exception as e:
        print(f"Falha na conversao CTranslate2: {e}")
        print("Traducao usara argostranslate como fallback em runtime.")

    if converted:
        try:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(HF_MODEL).save_pretrained(OUTPUT_DIR)
            print("Tokenizer salvo.")
        except Exception as e:
            print(f"Aviso: falha ao salvar tokenizer: {e}")

# ── SileroVAD (cache local) ───────────────────────────────────────────────────
print("Fazendo cache do SileroVAD...")
try:
    from silero_vad import load_silero_vad

    load_silero_vad()
    print("SileroVAD em cache.")
except Exception as e:
    print(f"Aviso: SileroVAD nao disponivel: {e}")

# ── Coqui TTS (download do modelo) ───────────────────────────────────────────
print("Baixando modelo Coqui TTS (tts_models/en/ljspeech/vits)...")
try:
    from TTS.api import TTS

    TTS("tts_models/en/ljspeech/vits")
    print("Coqui TTS instalado.")
except ImportError:
    print("Pacote TTS nao instalado. Instale com: pip install TTS")
    print("O Piper sera usado como fallback.")
except Exception as e:
    print(f"Aviso: erro ao baixar Coqui TTS: {e}")

print("\nSetup concluido.")
