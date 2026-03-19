#!/usr/bin/env python3
"""
rigorous_regime_detection_fixed2.py

Fixed & improved version — includes missing helper compute_regime_stability.
Run: python rigorous_regime_detection_fixed2.py
"""
import os
import warnings
warnings.filterwarnings('ignore')

import math
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import yfinance as yf

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    GaussianHMM = None
    HMM_AVAILABLE = False
    print("WARNING: hmmlearn not available — HMMs will be skipped.")

try:
    from wkmeans import WKMeans
    WKMEANS_AVAILABLE = True
except Exception:
    WKMEANS_AVAILABLE = False

sns.set_style("darkgrid")
OUTPUT_DIR = "rigorous_regime_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
TICKER = "SPY"
FULL_START = "1997-01-01"
FULL_END = "2024-12-31"

# Walk-forward periods (train_start, train_end, test_start, test_end)
WALK_FORWARD_PERIODS = [
    ("1997-01-01", "2010-12-31", "2011-01-01", "2015-12-31"),
    ("1997-01-01", "2015-12-31", "2016-01-01", "2020-12-31"),
    ("1997-01-01", "2020-12-31", "2021-01-01", "2024-12-31"),
]

# Known crash windows (for evaluation)
CRASH_WINDOWS = [
    (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-03-31")),   # 2008 crisis
    (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30")),   # COVID
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31")),   # 2022 selloff
]


# ---------------------------
# Data & features
# ---------------------------
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} in {start}:{end}")
    # normalize column names
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    df.index = pd.to_datetime(df.index)
    return df


def build_comprehensive_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    # returns
    df['ret'] = df['adj_close'].pct_change()
    df['log_ret'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    df['abs_ret'] = df['log_ret'].abs()
    df['signed_abs_ret'] = np.sign(df['log_ret']) * df['abs_ret']

    # multi-scale MA of returns
    for window in [5, 10, 21, 42, 63]:
        df[f'ma_ret_{window}'] = df['log_ret'].rolling(window, min_periods=max(1, window//3)).mean()

    # volatility (realized)
    for window in [5, 10, 21, 42, 63]:
        df[f'vol_{window}'] = df['log_ret'].rolling(window, min_periods=max(1, window//3)).std() * np.sqrt(252)

    # vol ratio
    vol_short = df['log_ret'].rolling(10, min_periods=5).std() * np.sqrt(252)
    vol_long = df['log_ret'].rolling(63, min_periods=30).std() * np.sqrt(252)
    df['vol_ratio'] = vol_short / (vol_long.replace(0, np.nan))

    # momentum
    for window in [21, 42, 63]:
        df[f'mom_{window}'] = df['log_ret'].rolling(window, min_periods=max(1, window//3)).sum()

    # skewness & kurtosis
    for window in [21, 42, 63]:
        df[f'skew_{window}'] = df['log_ret'].rolling(window, min_periods=max(5, window//3)).skew()
        df[f'kurt_{window}'] = df['log_ret'].rolling(window, min_periods=max(5, window//3)).kurt()

    # ranges & ATR
    prev_close = df['adj_close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for window in [14, 21]:
        df[f'atr_{window}'] = true_range.rolling(window, min_periods=max(3, window//2)).mean() / df['adj_close']

    # intraday gap
    df['gap'] = (df['open'] - prev_close) / prev_close

    # volume
    df['vol_ma_ratio_21'] = df['volume'] / (df['volume'].rolling(21, min_periods=1).mean().replace(0, np.nan))
    df['vol_ma_ratio_63'] = df['volume'] / (df['volume'].rolling(63, min_periods=1).mean().replace(0, np.nan))
    df['log_volume'] = np.log(df['volume'] + 1)

    # VW returns
    df['vwret_5'] = (df['log_ret'] * df['volume']).rolling(5, min_periods=3).sum() / (df['volume'].rolling(5, min_periods=3).sum().replace(0, np.nan))

    # trend indicators
    for window in [21, 42, 63]:
        df[f'trend_{window}'] = (df['adj_close'] / df['adj_close'].shift(window) - 1)

    exclude = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude]

    df_clean = df.dropna(subset=feature_cols)
    return df_clean, feature_cols


def select_stable_features(df: pd.DataFrame, feature_cols: List[str], train_end_date: str, n_features: int = 15):
    df_train = df[df.index <= train_end_date]
    rows = []
    for feat in feature_cols:
        s = df_train[feat].dropna()
        if len(s) < 100:
            continue
        acf1 = s.autocorr(lag=1)
        acf_score = 1.0 - abs(acf1)
        mid = len(s) // 2
        v1 = s.iloc[:mid].std()
        v2 = s.iloc[mid:].std()
        var_stability = min(v1, v2) / max(v1, v2) if max(v1, v2) > 0 else 0
        outlier_frac = (np.abs(s - s.mean()) > 3 * s.std()).mean()
        outlier_score = 1.0 - outlier_frac
        score = acf_score * 0.3 + var_stability * 0.4 + outlier_score * 0.3
        rows.append({'feature': feat, 'score': score, 'acf1': acf1, 'var_stability': var_stability, 'outlier_frac': outlier_frac})
    df_scores = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    selected = df_scores.head(n_features)['feature'].tolist()
    return selected, df_scores


# ---------------------------
# Utilities: scaling / PCA / smoothing
# ---------------------------
def make_feature_matrix(df: pd.DataFrame, features: List[str], scaler=None, pca=None, pca_var=0.95):
    X = df[features].values
    if scaler is None:
        scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    if pca is None:
        pca = PCA(n_components=pca_var, svd_solver='full')
    Xp = pca.fit_transform(Xs)
    return Xp, scaler, pca


def smooth_states_minrun(states: pd.Series, min_run_days: int = 3) -> pd.Series:
    """Replace runs shorter than min_run_days by previous regime (forward-fill)."""
    s = states.copy().astype(object)
    runs = (s != s.shift()).cumsum()
    run_sizes = s.groupby(runs).transform('size')
    mask_short = run_sizes < min_run_days
    s.loc[mask_short] = np.nan
    s = s.fillna(method='ffill').fillna(method='bfill')  # fill edge cases
    return s.astype(int)


# ---------------------------
# Model fitters
# ---------------------------
def fit_hmm_stable(X: np.ndarray, n_states: int = 3, cov_type: str = 'diag', n_restarts: int = 8, n_iter: int = 500, init_with_kmeans: bool = True):
    if not HMM_AVAILABLE:
        return None
    best_ll = -np.inf
    best_model = None
    for seed in range(n_restarts):
        try:
            model = GaussianHMM(n_components=n_states, covariance_type=cov_type, n_iter=n_iter, random_state=seed, tol=1e-4)
            if init_with_kmeans and X.shape[0] >= n_states:
                try:
                    km = KMeans(n_clusters=n_states, n_init=10, random_state=seed).fit(X)
                    # set rough initial means (they may be overwritten by fit depending on init_params)
                    model.means_ = km.cluster_centers_
                except Exception:
                    pass
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception:
            continue
    return best_model


def fit_gmm_wrapper(X: np.ndarray, n_states: int = 3):
    gmm = GaussianMixture(n_components=n_states, random_state=42, n_init=10)
    gmm.fit(X)
    return gmm


def fit_kmeans(X: np.ndarray, n_states: int = 3):
    km = KMeans(n_clusters=n_states, random_state=42, n_init=20)
    km.fit(X)
    return km


# ---------------------------
# Crash mapping & metrics
# ---------------------------
def choose_crash_regime_by_overlap(labels: pd.Series, crash_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]):
    crash_days = set()
    for a, b in crash_windows:
        crash_days.update(pd.date_range(a, b, freq='D'))
    counts = {}
    totals = {}
    for lab in np.unique(labels.dropna()):
        idxs = labels[labels == lab].index
        totals[lab] = len(idxs)
        counts[lab] = sum(1 for d in idxs if d in crash_days)
    fractions = {lab: (counts.get(lab, 0) / totals.get(lab, 1)) for lab in totals.keys()}
    if not fractions:
        return None
    best = max(fractions.items(), key=lambda kv: (kv[1], counts.get(kv[0], 0)))[0]
    return int(best)


def precision_recall_basic(predicted_labels: pd.Series, crash_regime: int, crash_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]):
    s = predicted_labels.dropna()
    crash_days = set()
    for a, b in crash_windows:
        crash_days.update(pd.date_range(a, b, freq='D'))
    preds = s[s == crash_regime].index
    true_positives = sum(1 for d in preds if d in crash_days)
    total_pred = len(preds)
    total_crash_days = sum(1 for d in s.index if d in crash_days)
    precision = true_positives / max(1, total_pred)
    recall = true_positives / max(1, total_crash_days)
    return precision, recall


def event_level_detection(predicted_labels: pd.Series, crash_windows: List[Tuple[pd.Timestamp, pd.Timestamp]], crash_regime: int, slack_days: int = 3):
    s = predicted_labels.dropna()
    detected = 0
    lags = []
    for a, b in crash_windows:
        window_range = pd.date_range(a - pd.Timedelta(days=slack_days), b + pd.Timedelta(days=slack_days), freq='D')
        hits = [d for d in s[s == crash_regime].index if d in window_range]
        if hits:
            detected += 1
            first_hit = min(hits)
            lags.append((first_hit - a).days)
    recall_event = detected / max(1, len(crash_windows))
    avg_lag = np.mean(lags) if lags else None
    return recall_event, avg_lag


# ---------------------------
# Small but critical helper (was missing)
# ---------------------------
def compute_regime_stability(labels: pd.Series):
    """
    Compute regime stability metrics:
     - mean_duration: average contiguous run length (in days)
     - transition_rate: transitions / length
     - entropy: normalized entropy of label distribution
     - n_regimes_observed: distinct regimes observed
    """
    s = labels.reset_index(drop=True)
    if s.empty:
        return {'mean_duration': 0.0, 'transition_rate': 0.0, 'entropy': 0.0, 'n_regimes_observed': 0}
    runs = []
    current = s.iloc[0]
    current_run = 1
    for val in s.iloc[1:]:
        if val == current:
            current_run += 1
        else:
            runs.append(current_run)
            current = val
            current_run = 1
    runs.append(current_run)
    mean_duration = float(np.mean(runs)) if runs else 0.0
    transitions = max(len(runs) - 1, 0)
    transition_rate = transitions / max(1, len(s))
    counts = pd.Series(labels).value_counts()
    probs = counts / counts.sum() if counts.sum() > 0 else counts
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
    max_entropy = math.log2(len(counts)) if len(counts) > 0 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return {'mean_duration': mean_duration, 'transition_rate': transition_rate, 'entropy': normalized_entropy, 'n_regimes_observed': len(counts)}


# ---------------------------
# Walk-forward validation
# ---------------------------
def walk_forward_validate(df_full: pd.DataFrame, feature_cols: List[str], n_states: int = 3, methods: List[str] = ['hmm', 'gmm', 'kmeans']):
    results = {}
    for idx, (train_start, train_end, test_start, test_end) in enumerate(WALK_FORWARD_PERIODS):
        print("="*80)
        print(f"PERIOD {idx+1}: Train {train_start} -> {train_end} ; Test {test_start} -> {test_end}")
        print("="*80)
        df_train = df_full[(df_full.index >= train_start) & (df_full.index <= train_end)]
        df_test = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)]
        if len(df_train) < 100 or len(df_test) < 20:
            print("  insufficient data, skipping")
            continue

        # Build features + scaler + PCA on training only, then transform train & test
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(df_train[feature_cols].values)
        pca = PCA(n_components=0.95, svd_solver='full')
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_scaled = scaler.transform(df_test[feature_cols].values)
        X_test_pca = pca.transform(X_test_scaled)

        period_results = {}

        # HMM
        if 'hmm' in methods and HMM_AVAILABLE:
            print("  Training HMM (diag cov, multiple restarts)...")
            hmm_model = fit_hmm_stable(X_train_pca, n_states=n_states, cov_type='diag', n_restarts=8, n_iter=800, init_with_kmeans=True)
            if hmm_model is not None:
                train_labels = pd.Series(hmm_model.predict(X_train_pca), index=df_train.index)
                test_labels = pd.Series(hmm_model.predict(X_test_pca), index=df_test.index)
                # smooth tiny runs
                test_labels_sm = smooth_states_minrun(test_labels, min_run_days=3)
                period_results['hmm'] = {'train_labels': train_labels, 'test_labels': test_labels_sm, 'model': hmm_model,
                                         'scaler': scaler, 'pca': pca}
            else:
                print("  HMM fitting failed for this period.")

        # GMM
        if 'gmm' in methods:
            print("  Training GMM...")
            gmm = fit_gmm_wrapper(X_train_pca, n_states=n_states)
            train_labels = pd.Series(gmm.predict(X_train_pca), index=df_train.index)
            test_labels = pd.Series(gmm.predict(X_test_pca), index=df_test.index)
            test_labels_sm = smooth_states_minrun(test_labels, min_run_days=2)
            period_results['gmm'] = {'train_labels': train_labels, 'test_labels': test_labels_sm, 'model': gmm, 'scaler': scaler, 'pca': pca}

        # KMeans
        if 'kmeans' in methods:
            print("  Training KMeans...")
            km = fit_kmeans(X_train_pca, n_states=n_states)
            train_labels = pd.Series(km.predict(X_train_pca), index=df_train.index)
            test_labels = pd.Series(km.predict(X_test_pca), index=df_test.index)
            test_labels_sm = smooth_states_minrun(test_labels, min_run_days=2)
            period_results['kmeans'] = {'train_labels': train_labels, 'test_labels': test_labels_sm, 'model': km, 'scaler': scaler, 'pca': pca}

        results[f'period_{idx+1}'] = period_results
    return results


# ---------------------------
# Agreement metrics & summarization
# ---------------------------
def compute_inter_method_agreement(period_results):
    methods = list(period_results.keys())
    if len(methods) < 2:
        return None
    indices = [period_results[m]['test_labels'].index for m in methods]
    common_idx = indices[0]
    for idx in indices[1:]:
        common_idx = common_idx.intersection(idx)
    if len(common_idx) < 10:
        return None
    labels = {m: period_results[m]['test_labels'].loc[common_idx].values for m in methods}
    rows = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            ari = adjusted_rand_score(labels[m1], labels[m2])
            nmi = normalized_mutual_info_score(labels[m1], labels[m2])
            rows.append({'method1': m1, 'method2': m2, 'ari': ari, 'nmi': nmi})
    return pd.DataFrame(rows)


# ---------------------------
# Main
# ---------------------------
def main():
    print("\nRIGOROUS REGIME DETECTION (fixed version)\n")
    print("Fetching data...")
    df = fetch_data(TICKER, FULL_START, FULL_END)
    print(f"Fetched {len(df)} rows for {TICKER}")

    print("Building features...")
    df_feat, all_features = build_comprehensive_features(df)
    print(f"Built {len(all_features)} features; {len(df_feat)} rows after dropna")

    print("Selecting stable features (train_end_date = 2010-12-31)...")
    selected, scores = select_stable_features(df_feat, all_features, train_end_date='2010-12-31', n_features=15)
    print("Selected features:", selected)
    scores.to_csv(os.path.join(OUTPUT_DIR, 'feature_stability_scores.csv'), index=False)

    print("\nWalking forward...")
    wf = walk_forward_validate(df_feat, selected, n_states=3, methods=['hmm', 'gmm', 'kmeans'])

    summary_rows = []
    for period_name, period_data in wf.items():
        print("\n" + "-"*60)
        print(period_name.upper())
        print("-"*60)
        # per-method stats
        for method, md in period_data.items():
            test_labels = md['test_labels']
            stability = compute_regime_stability(test_labels)
            # choose crash regime by overlap
            crash_regime = choose_crash_regime_by_overlap(test_labels, CRASH_WINDOWS)
            precision, recall = precision_recall_basic(test_labels, crash_regime, CRASH_WINDOWS) if crash_regime is not None else (0.0, 0.0)
            event_rec, avg_lag = event_level_detection(test_labels, CRASH_WINDOWS, crash_regime, slack_days=3) if crash_regime is not None else (0.0, None)
            print(f"\n{method.upper()}:")
            print(f"  n_regimes_observed: {stability['n_regimes_observed']}")
            print(f"  mean_duration: {stability['mean_duration']:.1f} days")
            print(f"  transition_rate: {stability['transition_rate']:.2%}")
            print(f"  entropy: {stability['entropy']:.3f}")
            print(f"  crash_regime (by overlap): {crash_regime}")
            print(f"  precision (day-level): {precision:.3f}, recall (day-level): {recall:.3f}")
            print(f"  event_recall (window-detection): {event_rec:.3f}, avg_lag (days): {avg_lag}")
            # store
            summary_rows.append({
                'period': period_name, 'method': method,
                'mean_duration': stability['mean_duration'],
                'transition_rate': stability['transition_rate'],
                'entropy': stability['entropy'],
                'crash_precision': precision, 'crash_recall': recall,
                'event_recall': event_rec, 'event_avg_lag': avg_lag,
                'n_regimes': stability['n_regimes_observed']
            })

        # inter-method agreement
        agreement = compute_inter_method_agreement(period_data)
        if agreement is not None:
            print("\nInter-method agreement (ARI / NMI):")
            print(agreement.to_string(index=False))
            agreement.to_csv(os.path.join(OUTPUT_DIR, f"{period_name}_agreement.csv"), index=False)

        # Save labels per method for period
        for method, md in period_data.items():
            md['test_labels'].to_csv(os.path.join(OUTPUT_DIR, f"{period_name}_{method}_test_labels.csv"))

    # aggregate summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "walk_forward_summary_fixed2.csv"), index=False)

    print("\nAggregated metrics across periods:")
    if not summary_df.empty:
        agg = summary_df.groupby('method').agg({
            'mean_duration': ['mean', 'std'],
            'transition_rate': ['mean', 'std'],
            'entropy': ['mean', 'std'],
            'crash_precision': ['mean', 'std'],
            'crash_recall': ['mean', 'std'],
            'event_recall': ['mean', 'std']
        })
        print(agg.to_string())
        agg.to_csv(os.path.join(OUTPUT_DIR, "walk_forward_agg_fixed2.csv"))
    else:
        print("No results to aggregate.")

    print("\nOptional: WKMeans benchmark (requires wkmeans package).")
    if WKMEANS_AVAILABLE:
        try:
            print("Running quick WKMeans on distributional windows (60d) — may be slow.")
            window = 60
            ret = df_feat['log_ret'].dropna()
            dist_list, idxs = [], []
            for i in range(window, len(ret)):
                v = ret.iloc[i-window:i].values
                if np.any(np.isnan(v)):
                    continue
                dist_list.append(v)
                idxs.append(ret.index[i])
            if dist_list:
                arr = np.vstack(dist_list)
                wk = WKMeans(k=3)
                wk.fit(arr)
                wk_labels = wk.predict(arr)
                wk_series = pd.Series(wk_labels, index=idxs, name='wk_state')
                wk_series.to_csv(os.path.join(OUTPUT_DIR, "wk_state_series.csv"))
                print("WKMeans done.")
            else:
                print("Not enough distributional windows for WKMeans.")
        except Exception as e:
            print("WKMeans run failed:", e)
    else:
        print("WKMeans not installed — skip.")

    print("\nAll outputs saved in", OUTPUT_DIR)
    print("Completed.")


if __name__ == "__main__":
    main()
