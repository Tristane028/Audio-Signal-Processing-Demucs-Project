# Audio Signal Processing — Demucs Source Separation

**Author:** Tristan Toshiharu Endo  
**Institution:** University of California, San Diego  
**Year:** 2025

## Overview

This project uses Demucs, a deep learning model developed by Meta Research, to separate a song into its individual instrument components. We take a raw WAV file as input and split it into 4 stems: vocals, drums, bass, and other instruments. The goal of this project was to isolate the individual components of a song and examine the frequency content of each stem through spectrogram analysis.

## How It Works

The script loads a WAV file and runs it through the `htdemucs_ft` model from Demucs. The model separates the audio into 4 components and saves each one as its own WAV file. A spectrogram is then generated for the isolated instrument track to visualize the frequency content over time.

## Uploaded Files

The following WAV files are the output of running the separation on `fixed_18_and_life.wav`:

- `vocals.wav` — isolated vocal track
- `drums.wav` — isolated drum track
- `bass.wav` — isolated bass track
- `other.wav` — isolated instrument track (guitar and other instruments)

## Dependencies

- Python 3.9
- `demucs` 4.0.1
- `torchaudio`
- `scipy`
- `numpy`
- `matplotlib`

To install dependencies, activate your virtual environment and run:

```bash
pip install demucs torchaudio scipy numpy matplotlib
```

## Running the Script

Activate your virtual environment and run:

```bash
source ~/venvs/demucs39/bin/activate
python "Signal Processing.py"
```

Make sure to update the `file_path` variable in the script to point to your own WAV file.

## References

- Alexandre Défossez et al. *Hybrid Transformers for Music Source Separation.* Meta Research, 2022.
- Demucs GitHub: https://github.com/facebookresearch/demucs
