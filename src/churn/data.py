"""Data loading, validation and cleaning."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config

logger = logging.getLogger(__name__)


def download_dataset(path: Path = config.RAW_DATA_PATH) -> Path:
    """Download the IBM Telco dataset if it is not already on disk."""
    if path.exists():
        return path
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s", config.DATASET_URL)
    urllib.request.urlretrieve(config.DATASET_URL, path)  # noqa: S310 - fixed URL
    return path


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV exactly as published."""
    if not path.exists():
        download_dataset(path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Fix the known data-quality issues in the Telco dataset.

    * ``TotalCharges`` is stored as text and is blank for 11 brand-new
      customers (tenure == 0). They have not been billed yet, so 0 is the
      correct value rather than an imputed mean.
    * ``SeniorCitizen`` is 0/1 while every other flag is Yes/No; normalise it.
    * Strip stray whitespace and drop exact duplicate rows.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace("", np.nan), errors="coerce")
    missing = df["TotalCharges"].isna()
    if missing.any():
        logger.info("Filling %d blank TotalCharges (tenure==0) with 0", int(missing.sum()))
        df.loc[missing, "TotalCharges"] = 0.0

    df["SeniorCitizen"] = df["SeniorCitizen"].map(
        {0: "No", 1: "Yes", "0": "No", "1": "Yes", "No": "No", "Yes": "Yes"}
    )
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").astype(float)

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        if len(df) != before:
            logger.info("Dropped %d duplicate rows", before - len(df))

    return df.reset_index(drop=True)


def encode_target(series: pd.Series) -> pd.Series:
    """Map Yes/No churn labels to 1/0."""
    return series.map({"Yes": 1, "No": 0}).astype(int)


def validate(df: pd.DataFrame) -> None:
    """Fail fast if required columns are missing or the data is malformed."""
    missing = set(config.RAW_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[config.RAW_NUMERIC].isna().any().any():
        raise ValueError("Numeric columns contain missing values after cleaning")
    if (df[config.RAW_NUMERIC] < 0).any().any():
        raise ValueError("Numeric columns contain negative values")


def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (features, target, customer_ids) for the full cleaned dataset."""
    df = clean(load_raw())
    validate(df)
    X = df[config.RAW_FEATURES]
    y = encode_target(df[config.TARGET])
    ids = df[config.ID_COL]
    return X, y, ids


def split(
    X: pd.DataFrame, y: pd.Series, ids: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified train/test split so both sets keep the ~26.5% churn rate."""
    return train_test_split(
        X,
        y,
        ids,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
