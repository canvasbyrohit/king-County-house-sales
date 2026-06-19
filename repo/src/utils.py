# =============================================================================
# src/utils.py
# B141 Data Mining — King County House Sales
# General Utility Functions
# =============================================================================

import os
import pickle
import pandas as pd
import numpy as np


# ── Project Palette ───────────────────────────────────────────────────────────
COLOURS = {
    'blue':    '#2E4057',
    'teal':    '#048A81',
    'coral':   '#E05263',
    'gold':    '#EFA00B',
    'purple':  '#8338EC',
    'sky':     '#54C6EB',
    'brown':   '#6B4226',
}

CLUSTER_COLOURS = ['#2E4057', '#048A81', '#E05263',
                   '#EFA00B', '#8338EC', '#54C6EB', '#6B4226']

PLOT_DEFAULTS = {
    'figure.dpi':        150,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'font.family':       'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
}


def save_model(model, path: str):
    """Serialise a fitted sklearn model to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved → {path}")


def load_model(path: str):
    """Load a serialised sklearn model from disk."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded ← {path}")
    return model


def predict_price(model, features: dict, scaler=None,
                  zip_encoding: dict = None) -> float:
    """
    Generate a single property price prediction.

    Args:
        model      : Fitted sklearn model
        features   : Dict of feature_name → value
        scaler     : Optional fitted StandardScaler (for linear models)
        zip_encoding: Optional zipcode → encoded mean mapping

    Returns:
        Predicted price in USD (float)
    """
    df = pd.DataFrame([features])
    if zip_encoding and 'zipcode' in df.columns:
        df['zip_encoded'] = df['zipcode'].map(zip_encoding)
        df = df.drop('zipcode', axis=1)
    if scaler:
        X = scaler.transform(df)
    else:
        X = df.values
    log_price = model.predict(X)[0]
    return float(np.expm1(log_price))


def format_price(value: float) -> str:
    """Format a USD price value for display."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value / 1_000:.0f}K"


def price_tier(price: float) -> str:
    """Classify a price into a market tier."""
    if price < 300_000:
        return 'Entry (<$300K)'
    elif price < 500_000:
        return 'Mid ($300K–$500K)'
    elif price < 750_000:
        return 'Upper-Mid ($500K–$750K)'
    elif price < 1_000_000:
        return 'Luxury ($750K–$1M)'
    else:
        return 'Ultra-Luxury (>$1M)'


def summarise_results(eval_df: pd.DataFrame) -> str:
    """Print a formatted evaluation summary."""
    best = eval_df.loc[eval_df['R²'].idxmax()]
    lines = [
        "=" * 60,
        "  MODEL EVALUATION SUMMARY",
        "=" * 60,
        f"  Best Model : {best['Model']}",
        f"  R²         : {best['R²']:.4f}",
        f"  RMSE       : ${best['RMSE ($)']:,}",
        f"  MAE        : ${best['MAE ($)']:,}",
        f"  MAPE       : {best['MAPE (%)']:.2f}%",
        "=" * 60,
    ]
    return '\n'.join(lines)
