import argostranslate.package
import argostranslate.translate
from faster_whisper import WhisperModel

# atualizar lista de modelos
argostranslate.package.update_package_index()

available_packages = argostranslate.package.get_available_packages()

# procurar pacote en->pt
package_to_install = next(
    filter(lambda x: x.from_code == "pt" and x.to_code == "en", available_packages)
)

# baixar
download_path = package_to_install.download()

# instalar
argostranslate.package.install_from_path(download_path)

print("Modelo do argostranslate instalados!")

print("Iniciando instalcao de modelos do OpenAI Whisper (medium)")
model = WhisperModel("medium")

if model is not None:
    print("Modelo instalado: ", model)
else:
    print("Modelo nao instalado!")
