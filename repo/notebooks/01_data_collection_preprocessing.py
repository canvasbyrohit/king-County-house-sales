# =============================================================================
# B141 DATA MINING — King County House Sales
# Phase 3: Data Collection, Cleaning & Preprocessing
# =============================================================================
# Dataset : House Sales in King County, USA
# Source  : https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
# Author  : HARLFOXEM 
# =============================================================================

# ── CELL 1 : Install & Import ─────────────────────────────────────────────────
# Run this cell first in Google Colab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Plot styling
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'font.family': 'sans-serif'
})

print("✅ Libraries loaded successfully")
print(f"   pandas  : {pd.__version__}")
print(f"   numpy   : {np.__version__}")


# ── CELL 2 : Data Collection ──────────────────────────────────────────────────
"""
DATA COLLECTION
===============
Source  : Kaggle — House Sales in King County, USA
URL     : https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
Period  : May 2014 – May 2015
Method  : Direct CSV download (public dataset, no authentication required)
Licence : CC0 Public Domain

The dataset records 21,613 residential property transactions across King County,
Washington State, USA, including Seattle. Each row represents one unique sale
with 21 attributes covering size, location, condition, and sale metadata.
"""

# Load dataset (update path if running locally)
URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/kc_house_data.csv"

# ── For Google Colab: upload kc_house_data.csv then use the line below ────────
# from google.colab import files
# uploaded = files.upload()
# df_raw = pd.read_csv('kc_house_data.csv')

# ── Local / uploaded path ─────────────────────────────────────────────────────
df_raw = pd.read_csv('kc_house_data.csv')   # change path if needed

print("=" * 55)
print("  DATASET LOADED")
print("=" * 55)
print(f"  Rows      : {df_raw.shape[0]:,}")
print(f"  Columns   : {df_raw.shape[1]}")
print(f"  Memory    : {df_raw.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print("=" * 55)


# ── CELL 3 : Data Quality Assessment ─────────────────────────────────────────
"""
DATA QUALITY ASSESSMENT
=======================
Before any cleaning we audit the dataset across six dimensions:
  1. Completeness  — missing values
  2. Uniqueness    — duplicate rows / IDs
  3. Validity      — values within expected domain
  4. Consistency   — logical relationships between columns
  5. Accuracy      — statistical outliers
  6. Timeliness    — date range coverage
"""

print("\n── 3.1  COLUMN OVERVIEW ─────────────────────────────────────────")
info_df = pd.DataFrame({
    'dtype'    : df_raw.dtypes,
    'non_null' : df_raw.notnull().sum(),
    'missing'  : df_raw.isnull().sum(),
    'miss_%'   : (df_raw.isnull().mean() * 100).round(2),
    'unique'   : df_raw.nunique(),
    'sample'   : df_raw.iloc[0]
})
print(info_df.to_string())

print("\n── 3.2  DESCRIPTIVE STATISTICS ──────────────────────────────────")
print(df_raw.describe(percentiles=[.05, .25, .5, .75, .95]).T.to_string())

print("\n── 3.3  DUPLICATE CHECK ─────────────────────────────────────────")
print(f"  Full-row duplicates : {df_raw.duplicated().sum()}")
print(f"  Duplicate IDs       : {df_raw['id'].duplicated().sum()}")
print(f"  (Same house, multiple sales — legitimate re-listings)")


# ── CELL 4 : Data Type Correction ────────────────────────────────────────────
"""
DATA TYPE CORRECTION
====================
Issues identified:
  • 'date'  stored as object string  → parse to datetime
  • 'id'    is a numeric identifier  → keep as string (no arithmetic)
  • 'zipcode', 'waterfront', 'view', 'condition', 'grade', 'floors'
            are ordinal/categorical  → convert to appropriate types
  • 'yr_renovated' uses 0 as sentinel for "never renovated"
"""

df = df_raw.copy()

# Parse date
df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
print(f"✅ 'date' converted to datetime | range: {df['date'].min().date()} → {df['date'].max().date()}")

# Convert ID to string (prevent accidental arithmetic)
df['id'] = df['id'].astype(str)

# Ordinal columns — convert to int (already int but confirm)
ordinal_cols = ['waterfront', 'view', 'condition', 'grade', 'floors', 'zipcode']
for col in ordinal_cols:
    df[col] = df[col].astype(int)

print(f"✅ Ordinal columns confirmed as int: {ordinal_cols}")
print(f"\nUpdated dtypes:\n{df.dtypes}")


# ── CELL 5 : Missing Value Handling ──────────────────────────────────────────
"""
MISSING VALUE HANDLING
======================
Audit result: NO missing values detected in any column.

This dataset is unusually complete; however we apply domain-aware
interpretation to the 'yr_renovated' column where 0 means
"property has never been renovated" (not a missing value).
We preserve 0 and will engineer a binary flag in Phase 5.
"""

print("── Missing Values After Type Correction ──────────────────────────")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  ✅ No missing values found across all 21 columns.")

# Verify yr_renovated sentinel
print(f"\n  yr_renovated == 0  : {(df['yr_renovated'] == 0).sum():,} properties (never renovated)")
print(f"  yr_renovated  > 0  : {(df['yr_renovated'] > 0).sum():,} properties (have been renovated)")


# ── CELL 6 : Duplicate Removal ────────────────────────────────────────────────
"""
DUPLICATE REMOVAL
=================
Two types of duplicates assessed:

1. Full-row duplicates: 0 found — no action required.

2. Same-house, multiple-sale entries (same 'id', different 'date'):
   These represent legitimate re-listings (a property sold twice
   within the 12-month window). We RETAIN all records because:
   - Each row is a distinct financial transaction.
   - Price changes between re-listings are analytically valuable.
   - Removing them would discard real market behaviour.

Decision: Keep all 21,613 records.
"""

dupe_ids = df[df['id'].duplicated(keep=False)].sort_values(['id', 'date'])
print(f"  Unique property IDs : {df['id'].nunique():,}")
print(f"  Total transactions  : {len(df):,}")
print(f"  Re-listed properties: {df['id'].duplicated().sum():,}")
print(f"\n  Sample re-listing (same house, two sale dates):")
print(dupe_ids[['id', 'date', 'price', 'sqft_living', 'condition']].head(4).to_string(index=False))
print("\n  ✅ Decision: All 21,613 rows retained — each is a unique transaction.")


# ── CELL 7 : Outlier Detection & Treatment ────────────────────────────────────
"""
OUTLIER DETECTION & TREATMENT
==============================
Method 1: IQR fences — flag observations beyond 3 × IQR
Method 2: Z-score  — flag |z| > 3.5
Method 3: Domain knowledge — values that violate logical constraints

Columns examined: price, sqft_living, sqft_lot, bedrooms, bathrooms

Treatment strategy (per column):
  • 'bedrooms' == 33 : data-entry error → cap at 10 (next plausible max)
  • Extreme 'price'  : retained but flagged; real luxury sales are valid
  • Extreme 'sqft'   : retained; large estates are valid
"""

def iqr_outlier_report(df, cols):
    rows = []
    for col in cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({'feature': col, 'Q1': Q1, 'Q3': Q3,
                     'lower_fence': lower, 'upper_fence': upper,
                     'outliers': n_out, 'pct': round(n_out / len(df) * 100, 2)})
    return pd.DataFrame(rows).set_index('feature')

target_cols = ['price', 'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms']
print("── IQR Outlier Report (3 × IQR fences) ──────────────────────────")
print(iqr_outlier_report(df, target_cols).to_string())

# Fix data-entry error: bedroom value of 33
print(f"\n  Bedroom distribution (≥ 8): {df[df['bedrooms'] >= 8]['bedrooms'].value_counts().sort_index().to_dict()}")
print(f"  ⚠️  33 bedrooms is a clear data-entry error (likely '3' mistyped as '33')")
df.loc[df['bedrooms'] == 33, 'bedrooms'] = 3
print(f"  ✅ 33 → 3 bedrooms corrected")

# Log-transformation flag (for modelling — applied in Phase 5)
print(f"\n  Price skewness  (raw)    : {df['price'].skew():.3f}")
print(f"  Price skewness  (log1p)  : {np.log1p(df['price']).skew():.3f}")
print(f"  → Log-transformation of 'price' recommended for regression modelling")

# Visual: price distribution before/after log
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['price'] / 1e6, bins=80, color='steelblue', edgecolor='white')
axes[0].set_title('Price Distribution (Raw)', fontweight='bold')
axes[0].set_xlabel('Price (USD millions)')
axes[0].set_ylabel('Frequency')

axes[1].hist(np.log1p(df['price']), bins=80, color='coral', edgecolor='white')
axes[1].set_title('Price Distribution (Log-Transformed)', fontweight='bold')
axes[1].set_xlabel('log(Price + 1)')
axes[1].set_ylabel('Frequency')

plt.suptitle('Outlier & Skewness Assessment — Target Variable', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Figure saved: fig_price_distribution.png")


# ── CELL 8 : Feature Engineering (Preprocessing Stage) ───────────────────────
"""
FEATURE ENGINEERING — PREPROCESSING STAGE
==========================================
Domain-derived features created before scaling/encoding:

  1. house_age        : years since built (as of sale date)
  2. renovated        : binary flag — 1 if ever renovated, else 0
  3. yrs_since_reno   : years since last renovation (0 if never)
  4. sale_month       : month of sale (seasonality signal)
  5. sale_year        : year of sale
  6. basement_flag    : binary — 1 if property has a basement
  7. price_per_sqft   : derived price efficiency metric (EDA only — NOT used as model feature)
  8. living_ratio     : sqft_living / sqft_living15 (vs neighbourhood avg)
  9. lot_ratio        : sqft_lot / sqft_lot15 (vs neighbourhood avg)
"""

sale_year_ref = df['date'].dt.year

df['house_age']      = sale_year_ref - df['yr_built']
df['renovated']      = (df['yr_renovated'] > 0).astype(int)
df['yrs_since_reno'] = np.where(df['yr_renovated'] > 0,
                                sale_year_ref - df['yr_renovated'], 0)
df['sale_month']     = df['date'].dt.month
df['sale_year']      = df['date'].dt.year
df['basement_flag']  = (df['sqft_basement'] > 0).astype(int)
df['price_per_sqft'] = df['price'] / df['sqft_living']   # EDA use only
df['living_ratio']   = df['sqft_living'] / df['sqft_living15']
df['lot_ratio']      = df['sqft_lot']    / df['sqft_lot15']

print("✅ Engineered features added:")
new_cols = ['house_age','renovated','yrs_since_reno','sale_month',
            'sale_year','basement_flag','price_per_sqft','living_ratio','lot_ratio']
print(df[new_cols].describe().T[['mean','min','max']].round(2).to_string())


# ── CELL 9 : Categorical Encoding ────────────────────────────────────────────
"""
CATEGORICAL ENCODING
====================
Ordinal features (grade, condition, view) retain their numeric scale —
they have a meaningful rank order and are already integer-coded.

'zipcode' has 70 unique values — too many for one-hot encoding.
Strategy: use as-is for tree models; for linear models we apply
          target-mean encoding (computed on training set only — done
          in Phase 5 to prevent leakage).

'waterfront' is already binary (0/1) — no encoding needed.
'floors' is treated as a numeric ordinal.
"""

print("── Encoding Summary ──────────────────────────────────────────────")
print(f"  waterfront  : binary 0/1    → no encoding needed")
print(f"  view        : ordinal 0–4   → numeric retained")
print(f"  condition   : ordinal 1–5   → numeric retained")
print(f"  grade       : ordinal 1–13  → numeric retained")
print(f"  floors      : ordinal       → numeric retained")
print(f"  zipcode     : 70 categories → target-mean encode in Phase 5 (prevents leakage)")
print(f"  renovated   : binary 0/1    → already encoded in Cell 8")


# ── CELL 10 : Feature Scaling ─────────────────────────────────────────────────
"""
FEATURE SCALING
===============
Scaling is applied at modelling time (Phase 5) to prevent data leakage.
Here we document the strategy per feature group:

  • Linear / Ridge / Lasso     → StandardScaler on all numeric features
  • Tree models (RF, XGBoost)  → No scaling required
  • Clustering (K-Means)       → StandardScaler (distance-sensitive)

We define the feature lists here for consistency across phases.
"""

# Features excluded from modelling
EXCLUDE = ['id', 'date', 'price', 'price_per_sqft',  # leakage / identifiers
           'yr_built', 'yr_renovated',                # replaced by engineered features
           'sqft_above', 'sqft_basement']              # components of sqft_living

# Final feature set
FEATURES = [c for c in df.columns if c not in EXCLUDE]
TARGET   = 'price'

print("── Model Feature Set ─────────────────────────────────────────────")
print(f"  Total columns      : {df.shape[1]}")
print(f"  Excluded columns   : {len(EXCLUDE)}")
print(f"  Modelling features : {len(FEATURES)}")
print(f"\n  Features: {FEATURES}")
print(f"\n  Target  : {TARGET}")


# ── CELL 11 : Final Data Quality Report ──────────────────────────────────────

print("\n" + "=" * 55)
print("  FINAL DATA QUALITY REPORT")
print("=" * 55)
print(f"  Original records   : 21,613")
print(f"  Records removed    : 0  (no full-row duplicates)")
print(f"  Records retained   : {len(df):,}")
print(f"  Original features  : 21")
print(f"  Engineered features: 9  (new)")
print(f"  Final columns      : {df.shape[1]}")
print(f"  Missing values     : 0")
print(f"  Data errors fixed  : 1  (bedroom=33 → 3)")
print(f"  Encoding applied   : Binary flags + ordinal retention")
print(f"  Scaling strategy   : Deferred to Phase 5 (prevent leakage)")
print(f"  Modelling features : {len(FEATURES)}")
print("=" * 55)

# Save cleaned dataset
df.to_csv('kc_house_cleaned.csv', index=False)
print("\n  ✅ Cleaned dataset saved → kc_house_cleaned.csv")
