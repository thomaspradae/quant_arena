# utils.py
"""
Small utilities for data alignment and basic sanity checks.
"""

from typing import Union, List
import pandas as pd
import numpy as np


def align_series_to_index(
    series: Union[pd.Series, pd.DataFrame],
    index: pd.Index,
    method: str = "ffill"
) -> Union[pd.Series, pd.DataFrame]:
    """
    Reindex `series` to `index`. Default: forward-fill then back-fill any leading NaNs.
    - series: pd.Series or pd.DataFrame (time-indexed)
    - index: desired pd.DatetimeIndex
    - method: 'ffill', 'bfill', or 'none'
    Returns a new object with the requested index.
    """
    if series is None:
        raise ValueError("series is None")

    # If already same index, return a copy
    if series.index.equals(index):
        return series.copy()

    # Reindex (introduces NaNs where dates are missing)
    re = series.reindex(index)

    if method == 'none':
        return re
    elif method == 'ffill':
        # forward-fill then back-fill to cover leading NaNs
        re = re.ffill().bfill()
        # after ffill/bfill, coerce dtype where appropriate
        if isinstance(re, pd.DataFrame):
            re = re.infer_objects()
        else:
            re = pd.Series(re).infer_objects()
        return re
    elif method == 'bfill':
        re = re.bfill().ffill()
        if isinstance(re, pd.DataFrame):
            re = re.infer_objects()
        else:
            re = pd.Series(re).infer_objects()
        return re
    else:
        raise ValueError("method must be 'ffill', 'bfill', or 'none'")


def sanity_check_market_data(
    df: pd.DataFrame,
    require_cols: List[str] = None,
    nan_threshold: float = 0.05,
    fail_fast: bool = True
) -> dict:
    """
    Basic sanity checks for market data DataFrame with datetime index.
    Returns a report dict. Raises ValueError on critical failures if fail_fast=True.

    Checks:
      - index is DatetimeIndex and monotonic
      - no duplicate timestamps
      - required columns present
      - no negative/zero close prices
      - high fraction of NaNs (warning)
    """
    report = {'ok': True, 'errors': [], 'warnings': []}

    if df is None:
        report['errors'].append("DataFrame is None")
        report['ok'] = False
        if fail_fast:
            raise ValueError("Market data is None")
        return report

    # index checks
    if not isinstance(df.index, pd.DatetimeIndex):
        report['errors'].append("Index is not a DatetimeIndex")
    else:
        if not df.index.is_monotonic_increasing:
            report['errors'].append("Index is not monotonic increasing")
        if df.index.has_duplicates:
            report['errors'].append("Index has duplicate timestamps")

    # column checks
    if require_cols:
        missing = [c for c in require_cols if c not in df.columns]
        if missing:
            report['errors'].append(f"Missing required columns: {missing}")

    # price sanity
    if 'close' in df.columns:
        nonpos = (df['close'] <= 0).sum()
        if nonpos > 0:
            report['errors'].append(f"{int(nonpos)} non-positive 'close' prices found")

    # NaN fraction
    total_vals = df.shape[0] * df.shape[1]
    nan_vals = int(df.isna().sum().sum())
    nan_frac = nan_vals / max(1, total_vals)
    if nan_frac > nan_threshold:
        report['warnings'].append(f"High fraction of NaNs: {nan_frac:.3f}")

    if report['errors']:
        report['ok'] = False
        if fail_fast:
            raise ValueError("Market data sanity check failed: " + "; ".join(report['errors']))

    return report
