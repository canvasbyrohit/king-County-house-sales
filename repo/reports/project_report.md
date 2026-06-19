# B141 Data Mining — Project Report
## King County House Sales: Predictive Pricing & Market Segmentation

---

## Abstract

This report presents a complete Data Mining project applied to 21,613 residential
property transactions in King County, Washington State, USA (May 2014 – May 2015).
Following the CRISP-DM lifecycle, the project delivers a Gradient Boosting regression
model achieving R² = 0.912 and MAPE = 11.51%, meeting industry Automated Valuation
Model (AVM) benchmarks. K-Means clustering identifies two distinct market segments.
Seven actionable stakeholder recommendations are derived from model outputs and
exploratory analysis.

---

## 1. Business Understanding

**Problem:** The King County residential market lacks a transparent, data-driven
pricing mechanism. Estate agents, developers, buyers, and lenders rely on subjective
valuations leading to mispricing and inefficiency.

**Objectives:**
- O1: Build a regression model to predict residential property sale price
- O2: Identify the most influential features driving property value
- O3: Segment the market into distinct property tiers via clustering
- O4: Analyse geospatial price patterns across King County zipcodes
- O5: Flag overpriced/underpriced properties relative to their features
- O6: Deliver actionable pricing recommendations to stakeholders

**Success Metrics:** R² ≥ 0.85, MAPE ≤ 15%, Silhouette ≥ 0.40, ±10% accuracy ≥ 55%

---

## 2. Data Understanding

**Dataset:** House Sales in King County, USA
**Source:** https://www.kaggle.com/datasets/harlfoxem/housesalespredictions
**Records:** 21,613 | **Features:** 21 | **Period:** May 2014 – May 2015

Key descriptive statistics:
- Median price: $450,000 | Mean price: $540,088 (right-skewed, skewness = 4.02)
- Most common bedroom count: 3 | Most common grade: 7
- Waterfront properties: 163 (0.75% of dataset)
- Zipcode range: 70 unique zipcodes

---

## 3. Data Preparation

**Steps performed:**
1. Date parsing: `date` → datetime
2. Type corrections: `id` → string, ordinal columns confirmed as int
3. Missing values: None found across all 21 columns
4. Duplicate removal: 0 full-row duplicates; re-listings retained (legitimate transactions)
5. Data error correction: `bedrooms = 33` → corrected to `3`
6. Feature engineering: 9 new features (house_age, renovated, yrs_since_reno,
   sale_month, sale_year, basement_flag, price_per_sqft, living_ratio, lot_ratio)
7. Modelling features: 6 additional (grade_sqft, bath_bed_ratio, total_rooms,
   high_grade, log_sqft, log_lot)
8. Target transformation: log1p(price) — reduces skewness from 4.02 to 0.92
9. Encoding: Zipcode target-mean encoding (train-set fit only)
10. Scaling: StandardScaler (linear models only; tree models unscaled)

---

## 4. Exploratory Data Analysis

**12 visualisations produced** covering:
- Price distribution (raw and log-transformed)
- Grade vs price (bar + boxplot)
- Correlation heatmap (15 features)
- Living area vs price (scatter + log-log regression)
- Geospatial price map (absolute and per-sqft)
- Waterfront premium analysis (211% premium confirmed)
- Monthly sales trend (volume and median price, May 2014 – May 2015)
- House age × renovation interaction
- Feature correlation ranking (bar chart)
- Zipcode price analysis (top/bottom 15)
- Bedroom and bathroom distributions
- Bedrooms × grade interaction heatmap

**Key EDA findings:**
- sqft_living: r = 0.70 (strongest quantitative predictor)
- grade: r = 0.67 (strongest categorical predictor)
- condition: r = 0.04 (weakest — condition matters far less than grade)
- Waterfront: +211% median price premium over non-waterfront
- May–July: 39% of annual transaction volume

---

## 5. Modelling

**Regression Models:**

| Model | Design Choice | Rationale |
|---|---|---|
| Linear Regression | No regularisation | Interpretable baseline |
| Ridge (α=10) | L2, CV-selected | Handles multicollinearity |
| Lasso (α=0.0001) | L1, CV-selected | Automatic feature selection |
| Random Forest (n=300) | Max depth 20, sqrt features | Non-linear, robust |
| Gradient Boosting (n=400) | lr=0.08, depth=5, subsample=0.85 | Best accuracy |

**Clustering:**
- Algorithm: K-Means
- Features: lat, long, price, sqft_living, grade, condition, house_age
- Optimal k=2 (silhouette = 0.246, elbow analysis)

---

## 6. Evaluation

| Model | R² | RMSE | MAE | MAPE | CV R² |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.912** | **$111,337** | **$62,772** | **11.51%** | 0.895 |
| Random Forest | 0.857 | $147,297 | $74,185 | 12.80% | 0.888 |
| Lasso | 0.814 | $167,699 | $85,276 | 14.90% | 0.864 |
| Ridge | 0.813 | $167,963 | $85,271 | 14.90% | 0.864 |
| Linear Regression | 0.813 | $168,011 | $85,288 | 14.90% | 0.864 |

**Business accuracy (Gradient Boosting):**
- ±10%: 57.6% | ±15%: 73.9% | ±20%: 83.6%

**Residual analysis:** Near-normal distribution; mean ≈ $0 (no systematic bias);
slight heteroscedasticity at luxury tier (>$1M) — expected and documented.

---

## 7. Insights & Recommendations

### Key Findings
1. Grade is the most powerful value lever (10.6× spread from grade 6 to 13)
2. Location creates irreversible price stratification (8.2× zipcode spread)
3. Waterfront is a scarcity premium (+211%) concentrated in 0.75% of stock
4. Living area is the strongest quantitative predictor (r = 0.70)
5. Renovation delivers +33.9% median price uplift
6. Spring–summer drives volume (+39%) but not price (<3% variation)
7. Two market segments: Premium ($680K) and Standard ($375K)
8. Model matches industry AVM benchmarks (MAPE 11.51%)

### Recommendations
- Estate Agents: Deploy model as listing price tool; advise grade-led renovations
- Developers: Target undervalued zipcodes; build grade 9–10 stock in Cluster 1 areas
- Lenders: Dynamic LTV from model confidence intervals; mandatory appraisal >$1M
- City Planners: Affordable housing in bottom-15 zipcodes; waterfront access equity

---

## 8. Limitations

1. Temporal validity: Single 12-month window (2014–2015)
2. No external features (school ratings, crime, walkability)
3. No unstructured data (photos, listing descriptions)
4. King County-specific — not transferable without retraining
5. Clustering resolution: Silhouette 0.246 reflects gradual geographic gradients

---

## 9. Future Work

1. External data enrichment (Walk Score API, GreatSchools, crime statistics)
2. XGBoost/LightGBM for RMSE reduction
3. Spatial lag regression (Moran's I, spatial multiplier)
4. Time-series price forecasting (ARIMA/Prophet)
5. Neural network MLP comparison
6. SHAP explainability layer for regulatory compliance

---

*B141 Data Mining | King County House Sales Project*
*Dataset: https://www.kaggle.com/datasets/harlfoxem/housesalespredictions*
