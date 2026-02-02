# Data Validation Layer

This repo includes a reusable validation and cleaning layer for the fraud/urgency dataset. It validates schema, detects data quality issues, adds non-destructive feature flags, and produces Markdown/JSON reports.

## Run Validation

```bash
python scripts/validate_data.py \
  --train fraud_data/train.csv \
  --test fraud_data/test.csv \
  --out_dir runs \
  --mode warn
```

Strict mode (non-zero exit on failures):

```bash
python scripts/validate_data.py \
  --train fraud_data/train.csv \
  --test fraud_data/test.csv \
  --out_dir runs \
  --mode strict
```

Optional: write cleaned outputs and validation features.

```bash
python scripts/validate_data.py \
  --train fraud_data/train.csv \
  --test fraud_data/test.csv \
  --out_dir runs \
  --mode warn \
  --write_cleaned
```

## How to Interpret the Report

- `runs/validation_report.md` is a human-readable summary.
- `runs/validation_report.json` is machine-readable with full check details.
- **Failed checks** indicate strict violations (e.g., negative amounts, invalid categories).
- **Warnings** indicate risks worth addressing (e.g., outliers, balance inconsistencies, drift).
- **Model Impact** links each detected issue to likely macro F1 degradation and suggested fixes.

## Integrate Into Training

1) Run validation before training. Fail fast in strict mode for data regressions.
2) Merge `validation_features.csv` into your training set by `id` and use flags as model inputs.
3) Use drift results to pick robust features and encoding strategies.
4) Track class imbalance and apply class weighting or resampling.
