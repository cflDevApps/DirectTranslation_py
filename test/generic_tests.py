# import sounddevice as sd
#
# devices = sd.query_devices()
#
# for i, device in enumerate(devices):
#     if device["max_input_channels"] > 0:
#         print(i, device["name"])

import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    if status:
        print("Status:", status)
    print("MIN:", np.min(indata), "MAX:", np.max(indata))

print(sd.query_devices())

with sd.InputStream(
    samplerate=16000,
    channels=1,
    callback=callback,
    device=1,
):
    print("Fale algo...")
    sd.sleep(5000)