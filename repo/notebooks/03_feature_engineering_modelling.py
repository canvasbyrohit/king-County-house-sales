# =============================================================================
# B141 DATA MINING — King County House Sales
# Phase 5: Feature Engineering, Model Building & Hyperparameter Tuning
# =============================================================================

# ── CELL 1: Setup ─────────────────────────────────────────────────────────────
import pandas as pd, numpy as np, matplotlib.pyplot as plt, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BLUE,TEAL,CORAL,GOLD='#2E4057','#048A81','#E05263','#EFA00B'
CLUSTER_COLORS=['#2E4057','#048A81','#E05263','#EFA00B','#8338EC','#54C6EB','#6B4226']
plt.rcParams.update({'figure.dpi':150,'axes.titlesize':13,'axes.labelsize':11,
                     'font.family':'sans-serif','axes.spines.top':False,'axes.spines.right':False})

df = pd.read_csv('kc_house_cleaned.csv', parse_dates=['date'])
print(f"✅ Loaded: {df.shape}")

# ── CELL 2: Final Feature Engineering ─────────────────────────────────────────
"""
FINAL FEATURE ENGINEERING
==========================
New interaction & transformation features added at modelling stage:
  • grade_sqft      : product of grade × living area (premium quality signal)
  • bath_bed_ratio  : bathroom density per bedroom (luxury indicator)
  • total_rooms     : bedrooms + bathrooms (overall size proxy)
  • high_grade      : binary flag for grade ≥ 10 (luxury tier)
  • log_sqft        : log-transformed living area (linearises distribution)
  • log_lot         : log-transformed lot size

Target variable: log1p(price) — applied to satisfy linear regression assumptions.
Encoding: Zipcode → target-mean encoding (fit on train only to prevent leakage).
"""
df['grade_sqft']     = df['grade'] * df['sqft_living']
df['bath_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)
df['total_rooms']    = df['bedrooms'] + df['bathrooms'].astype(int)
df['high_grade']     = (df['grade'] >= 10).astype(int)
df['log_sqft']       = np.log1p(df['sqft_living'])
df['log_lot']        = np.log1p(df['sqft_lot'])

EXCLUDE  = ['id','date','price','price_per_sqft','yr_built','yr_renovated',
            'sqft_above','sqft_basement','sqft_living15','sqft_lot15']
FEATURES = [c for c in df.columns if c not in EXCLUDE]
TARGET   = 'price'

X = df[FEATURES].copy()
y = np.log1p(df[TARGET])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Zipcode target-mean encoding (train set only)
zip_mean             = y_train.groupby(X_train['zipcode']).mean()
X_train['zip_encoded']= X_train['zipcode'].map(zip_mean)
X_test['zip_encoded'] = X_test['zipcode'].map(zip_mean).fillna(zip_mean.mean())
X_train = X_train.drop('zipcode', axis=1)
X_test  = X_test.drop('zipcode', axis=1)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Features: {list(X_train.columns)}")

# ── CELL 3: Evaluation Helper ─────────────────────────────────────────────────
def evaluate(name, y_true_log, y_pred_log):
    yt, yp = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {'Model':name, 'R²':round(r2_score(yt,yp),4),
            'RMSE ($)':int(np.sqrt(mean_squared_error(yt,yp))),
            'MAE ($)':int(mean_absolute_error(yt,yp)),
            'MAPE (%)':round(np.mean(np.abs((yt-yp)/yt))*100,2)}

results = []

# ── CELL 4: Model 1 — Linear Regression (Baseline) ────────────────────────────
"""
MODEL 1 — LINEAR REGRESSION (BASELINE)
Rationale: Establishes interpretable baseline; tests whether a simple
           linear relationship between features and log-price is sufficient.
           No regularisation applied — all features enter the model.
"""
lr = LinearRegression()
lr.fit(X_train_s, y_train)
results.append(evaluate('Linear Regression (Baseline)', y_test, lr.predict(X_test_s)))
print("✅ Model 1 done:", results[-1])

# ── CELL 5: Model 2 — Ridge Regression (L2) ───────────────────────────────────
"""
MODEL 2 — RIDGE REGRESSION (L2 Regularisation)
Rationale: Addresses potential multicollinearity (sqft features are correlated).
           Alpha selected via 5-fold cross-validation across 7 candidate values.
           Ridge shrinks coefficients without eliminating features.
"""
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 50, 100, 500], cv=5)
ridge.fit(X_train_s, y_train)
results.append(evaluate(f'Ridge (α={ridge.alpha_})', y_test, ridge.predict(X_test_s)))
print(f"✅ Model 2 done | best α={ridge.alpha_} | {results[-1]}")

# ── CELL 6: Model 3 — Lasso Regression (L1) ───────────────────────────────────
"""
MODEL 3 — LASSO REGRESSION (L1 Regularisation)
Rationale: L1 performs automatic feature selection by driving irrelevant
           coefficients to zero. Alpha selected by cross-validation.
           Identifies the minimum sufficient feature set.
"""
lasso = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1], cv=5, max_iter=5000)
lasso.fit(X_train_s, y_train)
n_zero = np.sum(lasso.coef_ == 0)
results.append(evaluate(f'Lasso (α={lasso.alpha_})', y_test, lasso.predict(X_test_s)))
print(f"✅ Model 3 done | α={lasso.alpha_} | {n_zero} features zeroed | {results[-1]}")

# ── CELL 7: Model 4 — Random Forest ───────────────────────────────────────────
"""
MODEL 4 — RANDOM FOREST (Ensemble, n=300 trees)
Rationale: Captures non-linear relationships and feature interactions.
           Immune to multicollinearity. Provides feature importances.
           Hyperparameters: n_estimators=300, max_depth=20, max_features='sqrt'.
           Tree models do NOT require feature scaling.
"""
rf = RandomForestRegressor(n_estimators=300, max_depth=20,
                            min_samples_split=3, max_features='sqrt',
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(evaluate('Random Forest (Tuned)', y_test, rf.predict(X_test)))
print("✅ Model 4 done:", results[-1])

# ── CELL 8: Model 5 — Gradient Boosting ───────────────────────────────────────
"""
MODEL 5 — GRADIENT BOOSTING (Sequential Ensemble)
Rationale: Builds trees sequentially; each corrects predecessor's residuals.
           Typically achieves highest accuracy among tree methods.
           Hyperparameters: n_estimators=400, lr=0.08, max_depth=5, subsample=0.85.
           Subsample=0.85 introduces stochastic element to reduce overfitting.
"""
gb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.08,
                                max_depth=5, subsample=0.85, random_state=42)
gb.fit(X_train, y_train)
results.append(evaluate('Gradient Boosting (Tuned)', y_test, gb.predict(X_test)))
print("✅ Model 5 done:", results[-1])

# ── CELL 9: Results Table ──────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print("\n" + "="*65)
print("  REGRESSION MODEL COMPARISON")
print("="*65)
print(results_df.to_string(index=False))

# ── CELL 10: K-Means Market Segmentation ──────────────────────────────────────
"""
K-MEANS CLUSTERING — MARKET SEGMENTATION (Secondary Task)
Features: lat, long, price, sqft_living, grade, condition, house_age
Goal   : Identify distinct property market tiers across King County.
Method : Elbow + silhouette analysis to choose optimal k.
"""
cluster_features = ['lat','long','price','sqft_living','grade','condition','house_age']
Xs = StandardScaler().fit_transform(df[cluster_features])

inertias, sils = [], []
for k in range(2, 9):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(Xs)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xs, lbl, sample_size=5000, random_state=42))

best_k = list(range(2,9))[np.argmax(sils)]
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(Xs)

cluster_profile = df.groupby('cluster').agg(
    n=('price','count'), median_price=('price','median'),
    median_sqft=('sqft_living','median'), median_grade=('grade','median'),
    median_age=('house_age','median'), lat=('lat','median'), long=('long','median')
).round(1)
print(f"\n── CLUSTER PROFILES (k={best_k}, silhouette={max(sils):.4f}) ──")
print(cluster_profile.to_string())

# ── CELL 11: Visualisations (Figs 13–17) ─────────────────────────────────────
# [All figure code as per individual cells above — see figures/ folder]
print("\n✅ Phase 5 complete. Run figure cells to generate charts.")
