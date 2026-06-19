# =============================================================================
# src/preprocessing.py
# B141 Data Mining — King County House Sales
# Reusable Data Preprocessing & Feature Engineering Functions
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw King County dataset and apply initial type corrections."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
    df['id']   = df['id'].astype(str)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def fix_data_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Correct known data-entry errors."""
    df = df.copy()
    # Fix bedroom value of 33 — clear data-entry error (likely '3')
    n_fixed = (df['bedrooms'] == 33).sum()
    df.loc[df['bedrooms'] == 33, 'bedrooms'] = 3
    if n_fixed > 0:
        print(f"✅ Fixed {n_fixed} record(s): bedrooms 33 → 3")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-driven engineered features.

    Phase 3 Features (from raw columns):
        house_age       : Years since built as of sale date
        renovated       : Binary — 1 if ever renovated
        yrs_since_reno  : Years since last renovation (0 = never)
        sale_month      : Month of sale (1–12)
        sale_year       : Year of sale
        basement_flag   : Binary — 1 if property has a basement
        price_per_sqft  : Price / sqft_living (EDA only, excluded from model)
        living_ratio    : sqft_living / sqft_living15
        lot_ratio       : sqft_lot / sqft_lot15

    Phase 5 Features (interaction & transformations):
        grade_sqft      : grade × sqft_living (quality-size interaction)
        bath_bed_ratio  : bathrooms / (bedrooms + 1)
        total_rooms     : bedrooms + int(bathrooms)
        high_grade      : Binary — 1 if grade >= 10 (luxury tier)
        log_sqft        : log1p(sqft_living)
        log_lot         : log1p(sqft_lot)
    """
    df = df.copy()
    sale_year_ref = df['date'].dt.year

    # Phase 3 features
    df['house_age']      = sale_year_ref - df['yr_built']
    df['renovated']      = (df['yr_renovated'] > 0).astype(int)
    df['yrs_since_reno'] = np.where(df['yr_renovated'] > 0,
                                    sale_year_ref - df['yr_renovated'], 0)
    df['sale_month']     = df['date'].dt.month
    df['sale_year']      = df['date'].dt.year
    df['basement_flag']  = (df['sqft_basement'] > 0).astype(int)
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    df['living_ratio']   = df['sqft_living'] / df['sqft_living15']
    df['lot_ratio']      = df['sqft_lot']    / df['sqft_lot15']

    # Phase 5 features
    df['grade_sqft']     = df['grade'] * df['sqft_living']
    df['bath_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)
    df['total_rooms']    = df['bedrooms'] + df['bathrooms'].astype(int)
    df['high_grade']     = (df['grade'] >= 10).astype(int)
    df['log_sqft']       = np.log1p(df['sqft_living'])
    df['log_lot']        = np.log1p(df['sqft_lot'])

    print(f"✅ Engineered features added — dataset now {df.shape[1]} columns")
    return df


def get_feature_lists():
    """Return standardised feature exclusion and inclusion lists."""
    EXCLUDE = [
        'id', 'date', 'price', 'price_per_sqft',
        'yr_built', 'yr_renovated',
        'sqft_above', 'sqft_basement',
        'sqft_living15', 'sqft_lot15'
    ]
    return EXCLUDE


def encode_zipcode(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series) -> tuple:
    """
    Apply target-mean encoding to 'zipcode'.
    Fit on training set only to prevent data leakage.

    Returns:
        X_train, X_test with 'zipcode' replaced by 'zip_encoded'
    """
    zip_mean = y_train.groupby(X_train['zipcode']).mean()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train['zip_encoded'] = X_train['zipcode'].map(zip_mean)
    X_test['zip_encoded']  = X_test['zipcode'].map(zip_mean).fillna(zip_mean.mean())
    X_train = X_train.drop('zipcode', axis=1)
    X_test  = X_test.drop('zipcode', axis=1)
    print("✅ Zipcode target-mean encoded (train-fit only)")
    return X_train, X_test


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Apply StandardScaler fitted on training data.
    Use for linear models only; tree models do not require scaling.

    Returns:
        X_train_scaled, X_test_scaled, fitted scaler
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    print("✅ StandardScaler fitted on train, applied to test")
    return X_train_s, X_test_s, scaler


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a data quality summary table."""
    report = pd.DataFrame({
        'dtype':    df.dtypes,
        'non_null': df.notnull().sum(),
        'missing':  df.isnull().sum(),
        'miss_%':   (df.isnull().mean() * 100).round(2),
        'unique':   df.nunique(),
        'min':      df.select_dtypes(include='number').min(),
        'max':      df.select_dtypes(include='number').max(),
    })
    return report


def full_preprocessing_pipeline(raw_path: str) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline.
    Run this function to go from raw CSV to model-ready DataFrame.
    """
    df = load_raw_data(raw_path)
    df = fix_data_errors(df)
    df = engineer_features(df)
    print(f"\n✅ Pipeline complete: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    return df
