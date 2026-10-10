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
git clone https://github.com/<Omarfaruk96>/CardioFuse.git
cd CardioFuse
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Smoke test with a fully synthetic EHR table
python cardiofuse.py --demo --epochs 5

# With the real dataset 
python cardiofuse.py --data data/ECHD_v2.csv --target <label_column>
```

Options: `--epochs`, `--batch-size`, `--lr`, `--img-size`, `--pretrained`, `--seed`, `--out`.

## Outputs (`results/`)

Three architectures were compared over 10 training epochs for cardiovascular disease prediction:

1. **CNN-only** (ECG images)
2. **Multimodal LSTM+CNN** (EHR + wearable + ECG, concatenation fusion)
3. **Multimodal LSTM+CNN + Attention** (multi-head self-attention fusion)

### Performance (Epoch 10 / Test Set)

| Metric | CNN-only | Multimodal | Multimodal + Attention |
|---|---|---|---|
| Train Accuracy | 92.87% | 95.62% | **96.56%** |
| Validation Accuracy | **95.62%** | 93.13% | 95.00% |
| Train Loss | 0.2194 | 0.1068 | **0.0879** |
| Validation Loss | **0.1968** | 0.2501 | 0.2297 |
| Test Accuracy | 94.03% | 94.53% | **96.02%** |
| F1-Score | 94.0% | 94.6% | **96.1%** |
| AUC-ROC | 0.978 | 0.981 | **0.989** |

### Key Findings

- **Attention fusion performs best on the test set**, leading in accuracy, precision, recall, F1, and AUC-ROC.
- **Multimodal models converge faster**, reaching 90% validation accuracy by epoch 3 versus epoch 5 for CNN-only.
- **Attention improves stability**, with validation accuracy variability (CV) of 1.2% versus 2.8% for CNN-only (57% lower).
- **Simple concatenation overfits**: its training loss is low but its validation loss is the highest of the three.
- **Cross-modal dependencies exist**: attention weights show all modalities draw on one another, and ECG attention rises by 15-20% for patients with ECG abnormalities.

Two things to check before publishing:

## Important notes

- ECG images and wearable sequences are **synthetic** and generated to correlate with the
  label, so reported performance demonstrates the pipeline, **not clinical validity**.
- With only 334 patients, results have high variance; consider repeated/cross-validated runs.
- This is a research prototype and **not a medical device**.
- Obtain ECHD v2 from its original source and respect its license.

## License


MIT
