import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

# /usr/local/bin/python3.9 "/Users/christan065/venv/demucs39/Signal Processing.py"

file_path = '/Users/christan065/fixed_18_and_life.wav'
duration_seconds = 20
save_dir = "/Users/christan065/Instras"
os.makedirs(save_dir, exist_ok=True)

# Read audio using scipy for spectrogram
srate, data = wavfile.read(file_path)
if len(data.shape) == 2:
    data = data.astype(np.float32).mean(axis=1)
data = data[:srate * duration_seconds]

# Spectrogram (full mix)
FRQ, TMs, Sxx = spectrogram(data, srate, nperseg=1024)
Sxx_dB = 10 * np.log10(Sxx + 1e-10)

# Demucs separation
model = get_model('htdemucs_ft')
instra, srate2 = torchaudio.load(file_path)
assert srate == srate2, "Sample rates do not match!"
if instra.shape[0] == 1:
    instra = instra.repeat(2, 1)
instra = instra.float()

# instra = instra[:, :srate2 * duration_seconds]

sources = apply_model(model, instra.unsqueeze(0), device='cpu')[0]

# Save separated tracks
components = ['drums', 'bass', 'vocals', 'other']
for i, name in enumerate(components):
    source = sources[i].unsqueeze(0) if sources[i].dim() == 1 else sources[i]
    torchaudio.save(os.path.join(save_dir, f"{name}.wav"), source, srate2)

# Load "other" instrument and create spectrogram
instr, srate_instr = torchaudio.load(os.path.join(save_dir, "other.wav"))
if instr.shape[0] > 1:
    instr = instr.mean(dim=0, keepdim=True)
instr = instr[0].numpy()

FRQ_2, TMs_2, Sxx_2 = spectrogram(instr, srate_instr, nperseg=1024)
Sxx_dB_2 = 10 * np.log10(Sxx_2 + 1e-10)

plt.figure(figsize=(12, 6))
plt.pcolormesh(TMs_2, FRQ_2, Sxx_dB_2, shading='gouraud')
plt.title("Spectrogram of Instrument (Guitar)")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Intensity [dB]")
plt.tight_layout()
plt.show()