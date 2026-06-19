# =============================================================================
# src/modelling.py
# B141 Data Mining — King County House Sales
# Reusable Modelling, Hyperparameter Tuning & Evaluation Functions
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, silhouette_score)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler


# ── Regression Models ─────────────────────────────────────────────────────────

def build_linear_regression(X_train_s, y_train):
    """Fit baseline OLS Linear Regression."""
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    print("✅ Linear Regression fitted")
    return model


def build_ridge(X_train_s, y_train,
                alphas=(0.01, 0.1, 1, 10, 50, 100, 500), cv=5):
    """Fit Ridge Regression with CV-selected alpha."""
    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(X_train_s, y_train)
    print(f"✅ Ridge fitted | best α = {model.alpha_}")
    return model


def build_lasso(X_train_s, y_train,
                alphas=(0.0001, 0.001, 0.01, 0.1, 1), cv=5):
    """Fit Lasso Regression with CV-selected alpha."""
    model = LassoCV(alphas=alphas, cv=cv, max_iter=5000)
    model.fit(X_train_s, y_train)
    n_zero = np.sum(model.coef_ == 0)
    print(f"✅ Lasso fitted | best α = {model.alpha_} | {n_zero} features zeroed")
    return model


def build_random_forest(X_train, y_train,
                        n_estimators=300, max_depth=20,
                        min_samples_split=3, max_features='sqrt',
                        random_state=42):
    """Fit tuned Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, max_features=max_features,
        random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"✅ Random Forest fitted | {n_estimators} trees | depth={max_depth}")
    return model


def build_gradient_boosting(X_train, y_train,
                             n_estimators=400, learning_rate=0.08,
                             max_depth=5, subsample=0.85,
                             random_state=42):
    """Fit tuned Gradient Boosting Regressor (best model)."""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, subsample=subsample,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print(f"✅ Gradient Boosting fitted | {n_estimators} trees | lr={learning_rate}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_regression(name: str, y_true_log, y_pred_log) -> dict:
    """
    Compute all regression metrics on original (non-log) scale.
    Assumes both inputs are log1p-transformed.
    """
    y_true = np.expm1(np.array(y_true_log))
    y_pred = np.expm1(np.array(y_pred_log))
    mse    = mean_squared_error(y_true, y_pred)
    return {
        'Model':    name,
        'R²':       round(r2_score(y_true, y_pred), 4),
        'RMSE ($)': int(np.sqrt(mse)),
        'MSE':      int(mse),
        'MAE ($)':  int(mean_absolute_error(y_true, y_pred)),
        'MAPE (%)': round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
    }


def business_accuracy(y_true_log, y_pred_log) -> dict:
    """
    Compute % of predictions within ±10%, ±15%, ±20% of actual price.
    Industry AVM benchmark: ±10% on ≥55% of valuations.
    """
    y_true    = np.expm1(np.array(y_true_log))
    y_pred    = np.expm1(np.array(y_pred_log))
    pct_error = np.abs((y_true - y_pred) / y_true) * 100
    return {
        '±10%': f"{(pct_error <= 10).mean() * 100:.1f}%",
        '±15%': f"{(pct_error <= 15).mean() * 100:.1f}%",
        '±20%': f"{(pct_error <= 20).mean() * 100:.1f}%",
    }


def cross_validate_model(model, X, y, n_splits=5, random_state=42) -> dict:
    """Run k-fold cross-validation and return mean ± std R²."""
    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
    return {
        'CV Mean R²': round(scores.mean(), 4),
        'CV Std':     round(scores.std(), 4),
        'CV Min':     round(scores.min(), 4),
        'CV Max':     round(scores.max(), 4),
    }


def compare_models(results: list) -> pd.DataFrame:
    """Convert list of evaluation dicts to a ranked DataFrame."""
    df = pd.DataFrame(results).sort_values('R²', ascending=False)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    return df


# ── Clustering ────────────────────────────────────────────────────────────────

def find_optimal_k(X_scaled, k_range=range(2, 9),
                   sample_size=5000, random_state=42) -> pd.DataFrame:
    """
    Compute inertia and silhouette score for k = 2..8.
    Returns DataFrame for elbow/silhouette plotting.
    """
    rows = []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbl = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, lbl,
                               sample_size=min(sample_size, len(X_scaled)),
                               random_state=random_state)
        rows.append({'k': k, 'inertia': km.inertia_, 'silhouette': round(sil, 4)})
        print(f"  k={k}: silhouette={sil:.4f}, inertia={km.inertia_:.0f}")
    result = pd.DataFrame(rows)
    best_k = result.loc[result['silhouette'].idxmax(), 'k']
    print(f"\n✅ Best k = {best_k} (silhouette = {result['silhouette'].max():.4f})")
    return result


def fit_kmeans(X_scaled, k: int, random_state=42) -> KMeans:
    """Fit final K-Means model with chosen k."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X_scaled)
    print(f"✅ K-Means fitted with k={k}")
    return km


def cluster_profile(df: pd.DataFrame, cluster_col: str = 'cluster') -> pd.DataFrame:
    """Generate summary statistics per cluster."""
    return df.groupby(cluster_col).agg(
        n             = ('price', 'count'),
        median_price  = ('price', 'median'),
        median_sqft   = ('sqft_living', 'median'),
        median_grade  = ('grade', 'median'),
        median_age    = ('house_age', 'median'),
        lat           = ('lat', 'median'),
        long          = ('long', 'median')
    ).round(1)
