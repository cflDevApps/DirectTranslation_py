"""
Diagnóstico de GPU para DirectTranslation.
Execute: python check_gpu.py
"""
import subprocess
import sys


def _section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def check_nvidia_driver():
    _section("Driver NVIDIA")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,cuda_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                name, driver, mem, cuda = [x.strip() for x in line.split(",")]
                print(f"  GPU          : {name}")
                print(f"  Driver       : {driver}")
                print(f"  VRAM total   : {mem}")
                print(f"  CUDA (driver): {cuda}")
        else:
            print("  ERRO: nvidia-smi falhou.")
            print("  Verifique se o driver NVIDIA está instalado.")
    except FileNotFoundError:
        print("  ERRO: nvidia-smi não encontrado.")
        print("  Driver NVIDIA não instalado ou não está no PATH.")


def check_pytorch():
    _section("PyTorch")
    try:
        import torch
        print(f"  Versão    : {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA ok   : {cuda_ok}")

        if cuda_ok:
            print(f"  CUDA ver  : {torch.version.cuda}")
            print(f"  GPU       : {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  VRAM      : {vram:.1f} GB")
        else:
            print()
            print("  PROBLEMA IDENTIFICADO:")
            print("  PyTorch instalado sem suporte CUDA (versão CPU-only).")
            print()
            _suggest_torch_install()

    except ImportError:
        print("  ERRO: PyTorch não instalado. Execute: pip install -r requirements.txt")


def _suggest_torch_install():
    """Sugere o comando de instalação correto com base na versão CUDA do driver."""
    cuda_ver = _detect_driver_cuda()
    if cuda_ver is None:
        print("  Não foi possível detectar a versão CUDA do driver.")
        print("  Verifique em: https://pytorch.org/get-started/locally/")
        return

    major = int(cuda_ver.split(".")[0])
    minor = int(cuda_ver.split(".")[1]) if "." in cuda_ver else 0

    if major >= 12 and minor >= 4:
        wheel = "cu124"
    elif major >= 12 and minor >= 1:
        wheel = "cu121"
    elif major >= 11 and minor >= 8:
        wheel = "cu118"
    else:
        print(f"  CUDA {cuda_ver} pode ser muito antigo. Recomenda-se CUDA 11.8+.")
        wheel = "cu118"

    print(f"  CUDA do driver detectado: {cuda_ver}")
    print(f"  Execute o comando abaixo para reinstalar PyTorch com CUDA {wheel[2:]}:")
    print()
    print(f"  pip install torch --index-url https://download.pytorch.org/whl/{wheel}")
    print()


def _detect_driver_cuda() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


def check_ctranslate2():
    _section("CTranslate2 (Whisper + Tradução)")
    try:
        import ctranslate2
        print(f"  Versão        : {ctranslate2.__version__}")
        n = ctranslate2.get_cuda_device_count()
        print(f"  GPUs CUDA     : {n}")
        if n == 0:
            print()
            print("  PROBLEMA IDENTIFICADO:")
            print("  CTranslate2 não detectou GPU CUDA.")
            print("  Solução: pip install ctranslate2[cuda12] (ou cuda11 dependendo da versão)")
    except ImportError:
        print("  ERRO: CTranslate2 não instalado.")


def check_faster_whisper():
    _section("faster-whisper (ASR)")
    try:
        import faster_whisper
        print(f"  Versão : {faster_whisper.__version__}")
        print("  Status : instalado")
    except ImportError:
        print("  ERRO: faster-whisper não instalado.")


def check_silero_vad():
    _section("SileroVAD")
    try:
        import silero_vad
        print("  Status : instalado")
        import torch
        if torch.cuda.is_available():
            print("  Dispositivo: CUDA (GPU)")
        else:
            print("  Dispositivo: CPU (PyTorch sem CUDA)")
    except ImportError:
        print("  ERRO: silero-vad não instalado.")


def print_summary():
    _section("Resumo e próximos passos")
    try:
        import torch
        import ctranslate2

        torch_ok = torch.cuda.is_available()
        ct2_ok = ctranslate2.get_cuda_device_count() > 0

        if torch_ok and ct2_ok:
            print("  Tudo OK — GPU disponível para todos os componentes.")
            return

        print("  Componentes com suporte GPU:")
        print(f"    PyTorch (SileroVAD, Coqui TTS) : {'OK' if torch_ok else 'CPU-only'}")
        print(f"    CTranslate2 (Whisper, Tradução) : {'OK' if ct2_ok else 'CPU-only'}")
        print()

        if not torch_ok:
            print("  1. Reinstale PyTorch com CUDA:")
            _suggest_torch_install()

        if not ct2_ok:
            print("  2. Reinstale CTranslate2 com CUDA:")
            print("     pip install \"ctranslate2[cuda12]\"")
            print()

    except ImportError:
        pass


if __name__ == "__main__":
    print("=" * 50)
    print("  DirectTranslation — Diagnóstico GPU")
    print(f"  Python {sys.version.split()[0]}")
    print("=" * 50)

    check_nvidia_driver()
    check_pytorch()
    check_ctranslate2()
    check_faster_whisper()
    check_silero_vad()
    print_summary()
    print()
