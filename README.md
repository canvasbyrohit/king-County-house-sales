[README.md](https://github.com/user-attachments/files/29147132/README.md)
# 🏠 King County House Sales — Data Mining Project
### B141 Data Mining | University Submission

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-green?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📌 Project Overview

A complete end-to-end Data Mining project applying the **CRISP-DM methodology** to
21,613 residential property transactions in King County, Washington State (USA).

The project delivers:
- A **Gradient Boosting regression model** (R² = 0.912, MAPE = 11.51%) for automated property valuation
- A **K-Means clustering** system identifying two distinct market segments
- **24 visualisations** covering spatial, temporal, univariate, bivariate, and multivariate analyses
- **7 stakeholder recommendations** for estate agents, developers, lenders, and city planners

---

## 🎯 Business Problem

> Estate agents, developers, and mortgage lenders in King County lack a transparent,
> data-driven pricing mechanism. Subjective valuations lead to mispricing, extended
> listing periods, and suboptimal investment decisions.

**Solution:** An Automated Valuation Model (AVM) powered by machine learning that
predicts residential property prices with industry-benchmark accuracy and segments
the market into actionable investment tiers.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Name** | House Sales in King County, USA |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalespredictions) |
| **Records** | 21,613 transactions |
| **Features** | 21 original + 9 engineered = 30 total |
| **Period** | May 2014 – May 2015 |
| **Target** | `price` (continuous — regression) |
| **Licence** | CC0 Public Domain |

> ⚠️ **Note:** The raw dataset (`kc_house_data.csv`) is not committed to this repository
> due to size. Download from Kaggle and place in `data/raw/`.

---

## 🏗️ Repository Structure

```
king-county-house-sales/
│
├── 📁 data/
│   ├── raw/                        # Original dataset (not committed — see note above)
│   │   └── kc_house_data.csv
│   └── processed/                  # Cleaned & engineered dataset
│       └── kc_house_cleaned.csv
│
├── 📁 notebooks/                   # Jupyter notebooks (one per phase)
│   ├── 01_data_collection_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering_modelling.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_insights_recommendations.ipynb
│
├── 📁 src/                         # Reusable Python modules
│   ├── preprocessing.py            # Cleaning & feature engineering functions
│   ├── modelling.py                # Model training & evaluation helpers
│   ├── visualisation.py            # Plotting utilities
│   └── utils.py                    # General helper functions
│
├── 📁 figures/                     # All 24 generated visualisations (PNG)
│   ├── fig01_price_distribution.png
│   ├── fig02_grade_vs_price.png
│   ├── ...
│   └── fig24_executive_dashboard.png
│
├── 📁 models/                      # Serialised trained models
│   ├── gradient_boosting_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── 📁 reports/                     # Project documentation
│   ├── project_report.md           # Full written report
│   └── evaluation_summary.csv      # Model metrics table
│
├── 📁 .github/                     # GitHub configuration
│   └── ISSUE_TEMPLATE.md
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Files excluded from version control
├── LICENSE                         # MIT Licence
└── README.md                       # This file
```

---

## ⚡ Quick Start

### Option 1 — Google Colab (Recommended)

1. Click the **Open in Colab** badge above
2. Upload `kc_house_data.csv` when prompted
3. Run all cells in sequence

### Option 2 — Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/king-county-house-sales.git
cd king-county-house-sales

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Place kc_house_data.csv in data/raw/

# 5. Launch Jupyter
jupyter notebook notebooks/
```

---

## 🔬 CRISP-DM Methodology

```
Phase 1 ✅  Business Understanding   →  Objectives, scope, success metrics
Phase 2 ✅  Data Understanding       →  EDA, 24 visualisations, hypothesis testing
Phase 3 ✅  Data Preparation         →  Cleaning, encoding, feature engineering
Phase 4 ✅  Modelling                →  5 regression models + K-Means clustering
Phase 5 ✅  Evaluation               →  R², RMSE, MAE, MAPE, CV, residual analysis
Phase 6 ✅  Deployment               →  Insights, recommendations, AVM blueprint
```

---

## 📈 Key Results

### Regression — Best Model: Gradient Boosting

| Metric | Value | Target | Status |
|---|---|---|---|
| R² Score | **0.912** | ≥ 0.85 | ✅ Exceeded |
| MAPE | **11.51%** | ≤ 15% | ✅ Exceeded |
| RMSE | **$111,337** | ≤ $130K | ✅ Met |
| ±10% Accuracy | **57.6%** | Industry ≥ 55% | ✅ Met |

### Clustering — K-Means (k=2)

| Segment | Size | Median Price | Profile |
|---|---|---|---|
| Premium | 32.2% (6,958) | $680,000 | Grade 9+, newer, eastern corridor |
| Standard | 67.8% (14,655) | $375,000 | Grade 7, older, western suburbs |

---

## 🔑 Top 5 Price Drivers

1. `grade_sqft` — Interaction of grade × living area (composite quality-size signal)
2. `log_sqft` — Log-transformed living area
3. `zip_encoded` — Location (target-mean encoded zipcode)
4. `lat` — Geographic latitude (north-south price gradient)
5. `grade` — King County structural quality rating (1–13)

---

## 📦 Models Compared

| Model | R² | RMSE | MAPE |
|---|---|---|---|
| Linear Regression (Baseline) | 0.813 | $168,011 | 14.90% |
| Ridge Regression (α=10) | 0.813 | $167,963 | 14.90% |
| Lasso Regression (α=0.0001) | 0.814 | $167,699 | 14.90% |
| Random Forest (n=300) | 0.857 | $147,297 | 12.80% |
| **Gradient Boosting (n=400)** | **0.912** | **$111,337** | **11.51%** |

---

## 💡 Key Insights

- **Grade is the #1 value lever** — Grade 13 median ($2.98M) is 10.6× Grade 6 ($0.28M)
- **Waterfront commands 211% premium** but represents only 0.75% of stock
- **Renovation delivers +33.9% median uplift** and neutralises age depreciation
- **Spring–summer drives volume (+39%), not price** — sellers hold firm year-round
- **Location gap**: Zipcode 98039 ($1.89M) vs 98002 ($0.23M) — 8.2× spread

---

## 📋 Visualisations (24 Figures)

| Range | Category |
|---|---|
| Fig 01–04 | Univariate & Bivariate Analysis |
| Fig 05–08 | Geospatial, Waterfront, Trend, Age Analysis |
| Fig 09–12 | Correlation, Zipcode, Room Counts, Multivariate Heatmap |
| Fig 13–17 | Feature Importance, Actual vs Predicted, Model Comparison, Clustering |
| Fig 18–20 | Residual Deep-Dive, CV vs Test, Accuracy by Tier |
| Fig 21–24 | Insights Dashboard, Investment Map, Renovation ROI, Executive Summary |

---

## 🛠️ Technology Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical computing |
| `matplotlib` | ≥ 3.7 | Visualisation |
| `seaborn` | ≥ 0.12 | Statistical plots |
| `scikit-learn` | ≥ 1.3 | ML models, preprocessing, evaluation |
| `scipy` | ≥ 1.10 | Statistical analysis |

---

## 👥 Stakeholder Recommendations Summary

| Stakeholder | Recommendation | KPI |
|---|---|---|
| Estate Agents | Deploy model as listing price tool | ↓ Days-on-market 20% |
| Estate Agents | Grade-led renovation advising | +12% above base prediction |
| Developers | Acquire undervalued-zipcode properties | ≥15% below predicted value |
| Developers | Build grade 9–10 stock in Cluster 1 areas | 25% gross margin |
| Lenders | Dynamic LTV from model confidence intervals | ↓ Collateral shortfall 30% |
| Lenders | Mandatory appraisal for >$1M properties | Zero luxury over-lending |
| City Planners | Affordable housing in bottom-15 zipcodes | 500 new units p.a. |

---

## 📄 Licence

This project is licensed under the **MIT Licence** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dataset: [harlfoxem on Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalespredictions)
- Original data source: King County Department of Assessments
- Module: B141 Data Mining

---

*Produced as part of a university Data Mining module assessment.*
