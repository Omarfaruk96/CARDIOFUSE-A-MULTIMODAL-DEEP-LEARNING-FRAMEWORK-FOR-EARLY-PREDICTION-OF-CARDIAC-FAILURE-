# CardioFuse

**Multimodal deep learning for early prediction of cardiac failure.**

CardioFuse fuses three data modalities:

| Modality | Source | Encoder |
|---|---|---|
| EHR (20 clinical features) | Erbil Cardiovascular Health Dataset (ECHD v2), 334 patients | MLP |
| 12-lead ECG images | Synthetic, label-conditioned | ResNet-18 |
| Wearable heart-rate series | Synthetic, label-conditioned | Bi-LSTM |

## Compared architectures

1. **`cnn_only`** – ResNet-18 on ECG images (baseline)
2. **`multimodal`** – EHR-MLP + ResNet-18 + LSTM, concatenation fusion (no attention)
3. **`multimodal_attn`** – same encoders with multi-head self-attention for dynamic modality weighting

Training uses a stratified **64 / 16 / 20** train/validation/test split, data augmentation
(ECG shifts/brightness/noise, HR scaling/jitter, EHR Gaussian noise), class-weighted loss,
AdamW, LR scheduling, and early stopping on validation AUC. EHR preprocessing is fit on
the training split only to avoid leakage.

## Quick start

```bash
git clone https://github.com/<your-username>/CardioFuse.git
cd CardioFuse
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Smoke test with a fully synthetic EHR table
python cardiofuse.py --demo --epochs 5

# With the real dataset (place the CSV in data/)
python cardiofuse.py --data data/ECHD_v2.csv --target <label_column>
```

Options: `--epochs`, `--batch-size`, `--lr`, `--img-size`, `--pretrained`, `--seed`, `--out`.

## Outputs (`results/`)

- `comparison.csv`, `metrics.json` – accuracy, precision, recall, F1, AUC, confusion matrices
- `roc_curves.png` – ROC comparison of all three models
- `modality_attention.png` – average attention weight per modality
- `*.pt` – trained weights (git-ignored)

## Important notes

- ECG images and wearable sequences are **synthetic** and generated to correlate with the
  label, so reported performance demonstrates the pipeline, **not clinical validity**.
- With only 334 patients, results have high variance; consider repeated/cross-validated runs.
- This is a research prototype and **not a medical device**.
- Obtain ECHD v2 from its original source and respect its license; the raw CSV is not
  included in this repository.

## License


MIT
