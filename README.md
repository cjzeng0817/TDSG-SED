# CRNN + TDSG

This repository contains the CRNN + TDSG implementation for *Time-Decoupled Spectral Modeling: Wide-Band Recalibration and Adaptive Fusion for Sound Event Detection*.

## Run

In `configs/config.yaml`, set the URBAN-SED dataset paths for your machine and choose an unused `generals.save_folder`. From the repository root, run

```bash
python main.py --config configs/config.yaml --gpu 0
```
