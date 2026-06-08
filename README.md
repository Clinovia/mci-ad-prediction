# Predicting MCI-to-AD Progression with Minimal Clinical Features

A machine learning pipeline for predicting 24-month progression from 
mild cognitive impairment (MCI) to Alzheimer's disease (AD) using 
routinely obtainable neuropsychological measures — without neuroimaging, 
fluid biomarkers, or genetic testing.

## Key findings

- A six-feature XGBoost model (age, sex, education, RAVLT Immediate, 
  MMSE, EcogSPTotal) achieves **AUC = 0.922**, within 0.002 of a 
  nine-feature full model (AUC = 0.924)
- APOE4 genotype contributes **no incremental predictive value** once 
  standard neuropsychological measures are available (ΔAUC = −0.001 
  at the full minimal feature set)
- The minimal model outperforms CDR-SB threshold classification on 
  all metrics (AUC 0.922 vs. 0.912; Brier score 0.093 vs. 0.117)

## Repository structure
mci-ad-prediction/
├── data/               # Place ADNI CSV here (not included — see below)
├── src/
│   ├── preprocessor.py           # Full model preprocessing
│   ├── preprocessor_ablation.py  # Ablation preprocessing
│   ├── evaluate_classifier.py    # Shared CV and evaluation utilities
│   ├── benchmark_models.py       # Stage 1 and 3 model comparison
│   ├── train_ablation.py         # Stage 4 sequential feature ablation
│   ├── train_cdrsb.py            # CDR-SB clinical baseline comparator
│   └── XGBoost_SHAP.py           # Stage 2 SHAP feature importance
├── results/            # Output CSVs written here after running scripts
├── requirements.txt
└── README.md

## Data access

This study uses data from the **Alzheimer's Disease Neuroimaging 
Initiative (ADNI)**. ADNI data are available to qualified researchers 
at no cost via:

> https://adni.loni.usc.edu

ADNI data **cannot be redistributed** and are therefore not included 
in this repository. To reproduce the analyses, apply for access through 
the ADNI data portal, download ADNIMERGE.csv into data/ and run the files in data/ place the prepared dataset at: data/adni_24m_progression_dataset_filled.csv

The dataset should contain the following columns at minimum:
RID, AGE, PTGENDER, PTEDUCAT, APOE4, MMSE, CDRSB, 
EcogSPTotal, EcogPtMem, RAVLT_forgetting, 
RAVLT_immediate, EcogSPMem,
EcogMem_discrepancy, EcogSPMem_missing

Where `Target_24m` is a binary variable (1 = progressed to AD within 
24 months, 0 = stable MCI).

## Setup

Requires Python 3.9 or later.

```bash
git clone https://github.com/Clinovia/mci-ad-prediction.git
cd mci-ad-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the analyses

Run the scripts in order from the `src/` directory:

```bash
cd src

# Stage 1 — Full model comparison (RF, XGBoost, LR on 9 features)
python benchmark_models.py

# Stage 2 — SHAP feature importance on XGBoost full model
python XGBoost_SHAP.py

# Stage 3 — Minimal model comparison (RF, XGBoost, LR on 6 features)
# (re-run benchmark_models.py with minimal feature flag if applicable)

# Stage 4 — Sequential feature ablation with and without APOE4
python train_ablation.py

# Clinical baseline — CDR-SB threshold comparator
python train_cdrsb.py
```

Results are saved as CSV files in `results/`.

