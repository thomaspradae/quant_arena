#!/usr/bin/env python3
"""
rigorous_regime_detection_fixed4.py

FINAL CORRECTED VERSION: 
- Fixes crash in `event_level_detection` argument order.
- Correctly saves the test label files for each period.
- Now enables the downstream analysis script to run correctly.
"""
import os
import warnings
warnings.filterwarnings('ignore')

import math
from datetime import datetime
from typing import List, Tuple, Dict

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
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
OUTPUT_DIR = "rigorous_regime_results_auto_n"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Deterministic seed
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Config
TICKER = "SPY"
FULL_START = "1997-01-01"
FULL_END = "2024-12-31"

WALK_FORWARD_PERIODS = [
    ("1997-01-01", "2010-12-31", "2011-01-01", "2015-12-31"),
    ("1997-01-01", "2015-12-31", "2016-01-01", "2020-12-31"),
    ("1997-01-01", "2020-12-31", "2021-01-01", "2024-12-31"),
]

CRASH_WINDOWS = [
    (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-03-31")),
    (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30")),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31")),
]

# ---------------------------
# Data & features
# ---------------------------
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} in {start}:{end}")
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    df.index = pd.to_datetime(df.index)
    return df

def build_comprehensive_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df['ret'] = df['adj_close'].pct_change()
    df['log_ret'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    df['abs_ret'] = df['log_ret'].abs()
    df['signed_abs_ret'] = np.sign(df['log_ret']) * df['abs_ret']
    for window in [5, 10, 21, 42, 63]:
        df[f'ma_ret_{window}'] = df['log_ret'].rolling(window, min_periods=max(1, window//3)).mean()
    for window in [5, 10, 21, 42, 63]:
        df[f'vol_{window}'] = df['log_ret'].rolling(window, min_periods=max(1, window//3)).std() * np.sqrt(252)
    vol_short = df['log_ret'].rolling(10, min_periods=5).std() * np.sqrt(252)
    vol_long = df['log_ret'].rolling(63, min_periods=30).std() * np.sqrt(252)
    df['vol_ratio'] = vol_short / (vol_long.replace(0, np.nan))
    for window in [21, 42, 63]:
        df[f'mom_{window}'] = df['log_ret'].rolling(window, min_periods=max(1, window//3)).sum()
    for window in [21, 42, 63]:
        df[f'skew_{window}'] = df['log_ret'].rolling(window, min_periods=max(5, window//3)).skew()
        df[f'kurt_{window}'] = df['log_ret'].rolling(window, min_periods=max(5, window//3)).kurt()
    prev_close = df['adj_close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for window in [14, 21]:
        df[f'atr_{window}'] = true_range.rolling(window, min_periods=max(3, window//2)).mean() / df['adj_close']
    df['gap'] = (df['open'] - prev_close) / prev_close
    df['vol_ma_ratio_21'] = df['volume'] / (df['volume'].rolling(21, min_periods=1).mean().replace(0, np.nan))
    df['vol_ma_ratio_63'] = df['volume'] / (df['volume'].rolling(63, min_periods=1).mean().replace(0, np.nan))
    df['log_volume'] = np.log(df['volume'] + 1)
    df['vwret_5'] = (df['log_ret'] * df['volume']).rolling(5, min_periods=3).sum() / (df['volume'].rolling(5, min_periods=3).sum().replace(0, np.nan))
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
        if len(s) < 100: continue
        acf1 = s.autocorr(lag=1)
        acf_score = 1.0 - abs(acf1)
        mid = len(s) // 2
        v1, v2 = s.iloc[:mid].std(), s.iloc[mid:].std()
        var_stability = min(v1, v2) / max(v1, v2) if max(v1, v2) > 0 else 0
        outlier_frac = (np.abs(s - s.mean()) > 3 * s.std()).mean()
        outlier_score = 1.0 - outlier_frac
        score = acf_score * 0.3 + var_stability * 0.4 + outlier_score * 0.3
        rows.append({'feature': feat, 'score': score})
    df_scores = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    return df_scores.head(n_features)['feature'].tolist(), df_scores

# ---------------------------
# Optimal State Selection
# ---------------------------
def find_optimal_n_states(X_train: np.ndarray, methods: List[str], state_range: List[int] = [2, 3, 4, 5]) -> Dict[str, int]:
    optimal_states = {}
    
    if 'hmm' in methods and HMM_AVAILABLE:
        bics = []
        for n in state_range:
            try:
                model = GaussianHMM(n_components=n, covariance_type='diag', n_iter=300, random_state=SEED).fit(X_train)
                bics.append(model.bic(X_train))
            except Exception:
                bics.append(np.inf)
        optimal_states['hmm'] = state_range[np.argmin(bics)] if bics and not all(np.isinf(bics)) else 3
    
    if 'gmm' in methods:
        bics = []
        for n in state_range:
            try:
                model = GaussianMixture(n_components=n, random_state=SEED, n_init=5).fit(X_train)
                bics.append(model.bic(X_train))
            except Exception:
                bics.append(np.inf)
        optimal_states['gmm'] = state_range[np.argmin(bics)] if bics and not all(np.isinf(bics)) else 3
        
    if 'kmeans' in methods:
        inertias = []
        for n in state_range:
            try:
                model = KMeans(n_clusters=n, random_state=SEED, n_init=10).fit(X_train)
                inertias.append(model.inertia_)
            except Exception:
                inertias.append(np.inf)
        
        if len(inertias) > 2:
            deltas = np.diff(inertias, 2)
            optimal_states['kmeans'] = state_range[np.argmax(deltas) + 1] if any(deltas > 0) else state_range[0]
        else:
            optimal_states['kmeans'] = 3

    return optimal_states

# ---------------------------
# Helper Functions
# ---------------------------
def smooth_states_minrun(states: pd.Series, min_run_days: int = 3) -> pd.Series:
    s = states.copy().astype(object)
    runs = (s != s.shift()).cumsum()
    run_sizes = s.groupby(runs).transform('size')
    mask_short = run_sizes < min_run_days
    s.loc[mask_short] = np.nan
    s = s.fillna(method='ffill').fillna(method='bfill')
    s = s.fillna(method='bfill')
    if s.isnull().any(): return pd.Series(dtype=int)
    return s.astype(int)

def fit_hmm_stable(X: np.ndarray, n_states: int, cov_type: str = 'diag', n_restarts: int = 8, n_iter: int = 500, init_with_kmeans: bool = True):
    if not HMM_AVAILABLE: return None
    best_ll, best_model = -np.inf, None
    for i in range(n_restarts):
        seed = SEED + i
        try:
            model = GaussianHMM(n_components=n_states, covariance_type=cov_type, n_iter=n_iter, random_state=seed, tol=1e-4)
            if init_with_kmeans and X.shape[0] >= n_states:
                km = KMeans(n_clusters=n_states, n_init=10, random_state=seed).fit(X)
                model.means_ = km.cluster_centers_
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll: best_ll, best_model = ll, model
        except Exception: continue
    return best_model

def fit_gmm_wrapper(X: np.ndarray, n_states: int):
    return GaussianMixture(n_components=n_states, random_state=SEED, n_init=10).fit(X)

def fit_kmeans(X: np.ndarray, n_states: int):
    return KMeans(n_clusters=n_states, random_state=SEED, n_init=20).fit(X)

def choose_crash_regime_by_overlap(labels: pd.Series, crash_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]):
    if labels.empty: return None
    crash_days = set()
    for a, b in crash_windows: crash_days.update(pd.date_range(a, b, freq='D'))
    counts, totals = {}, {}
    for lab in np.unique(labels.dropna()):
        idxs = labels[labels == lab].index
        totals[lab], counts[lab] = len(idxs), sum(1 for d in idxs if d in crash_days)
    fractions = {lab: (counts.get(lab, 0) / totals.get(lab, 1)) for lab in totals.keys()}
    if not fractions: return None
    return int(max(fractions.items(), key=lambda kv: (kv[1], counts.get(kv[0], 0)))[0])

def precision_recall_basic(predicted_labels: pd.Series, crash_regime: int, crash_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]):
    s = predicted_labels.dropna()
    if s.empty: return 0.0, 0.0
    crash_days = set()
    for a, b in crash_windows: crash_days.update(pd.date_range(a, b, freq='D'))
    preds = s[s == crash_regime].index
    true_positives = sum(1 for d in preds if d in crash_days)
    precision = true_positives / max(1, len(preds))
    recall = true_positives / max(1, sum(1 for d in s.index if d in crash_days))
    return precision, recall

def event_level_detection(predicted_labels: pd.Series, crash_windows: List[Tuple[pd.Timestamp, pd.Timestamp]], crash_regime: int, slack_days: int = 3):
    s, detected, lags = predicted_labels.dropna(), 0, []
    if s.empty: return 0.0, None
    for a, b in crash_windows:
        window_range = pd.date_range(a - pd.Timedelta(days=slack_days), b + pd.Timedelta(days=slack_days), freq='D')
        hits = [d for d in s[s == crash_regime].index if d in window_range]
        if hits:
            detected += 1
            lags.append((min(hits) - a).days)
    return detected / max(1, len(crash_windows)), np.mean(lags) if lags else None

def compute_regime_stability(labels: pd.Series):
    s = labels.dropna().reset_index(drop=True)
    if s.empty:
        return {'mean_duration': 0.0, 'transition_rate': 0.0, 'entropy': 0.0, 'n_regimes_observed': 0}
    runs = []
    if len(s) > 0:
        current_run = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
    mean_duration = float(np.mean(runs)) if runs else 0.0
    transitions = max(len(runs) - 1, 0)
    transition_rate = transitions / max(1, len(s))
    counts = labels.value_counts()
    probs = counts / counts.sum() if counts.sum() > 0 else counts
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
    max_entropy = math.log2(len(counts)) if len(counts) > 0 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return {'mean_duration': mean_duration, 'transition_rate': transition_rate, 'entropy': normalized_entropy, 'n_regimes_observed': len(counts)}

def compute_inter_method_agreement(period_results):
    methods = list(period_results.keys())
    if len(methods) < 2: return None

    indices = [period_results[m]['test_labels'].index for m in methods]
    common_idx = indices[0]
    for other_idx in indices[1:]:
        common_idx = common_idx.intersection(other_idx)

    if len(common_idx) < 10: return None
        
    labels = {m: period_results[m]['test_labels'].loc[common_idx].values for m in methods}
    rows = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            rows.append({
                'method1': m1, 'method2': m2, 
                'ari': adjusted_rand_score(labels[m1], labels[m2]), 
                'nmi': normalized_mutual_info_score(labels[m1], labels[m2])
            })
    return pd.DataFrame(rows)

# ---------------------------
# Walk-forward validation
# ---------------------------
def walk_forward_validate(df_full: pd.DataFrame, feature_cols: List[str], methods: List[str] = ['hmm', 'gmm', 'kmeans']):
    results = {}
    for idx, (train_start, train_end, test_start, test_end) in enumerate(WALK_FORWARD_PERIODS):
        print("="*80)
        print(f"PERIOD {idx+1}: Train {train_start} -> {train_end} ; Test {test_start} -> {test_end}")
        print("="*80)
        df_train = df_full[(df_full.index >= train_start) & (df_full.index <= train_end)]
        df_test = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)]
        if len(df_train) < 100 or len(df_test) < 20: continue

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(df_train[feature_cols].values)
        pca = PCA(n_components=0.95, svd_solver='full')
        X_train_pca = pca.fit_transform(X_train_scaled)
        
        print("  Searching for optimal n_states for each model...")
        optimal_n = find_optimal_n_states(X_train_pca, methods, state_range=[2, 3, 4, 5])
        print(f"  Optimal states found: {optimal_n}")

        X_test_scaled = scaler.transform(df_test[feature_cols].values)
        X_test_pca = pca.transform(X_test_scaled)

        period_results = {}
        
        if 'hmm' in methods and HMM_AVAILABLE:
            n_states = optimal_n.get('hmm', 3)
            print(f"  Training HMM with {n_states} states...")
            hmm_model = fit_hmm_stable(X_train_pca, n_states=n_states, cov_type='diag', n_restarts=8, n_iter=800)
            if hmm_model:
                test_labels = pd.Series(hmm_model.predict(X_test_pca), index=df_test.index)
                period_results['hmm'] = {'test_labels': smooth_states_minrun(test_labels, 3), 'n_states': n_states}
        
        if 'gmm' in methods:
            n_states = optimal_n.get('gmm', 3)
            print(f"  Training GMM with {n_states} states...")
            gmm = fit_gmm_wrapper(X_train_pca, n_states=n_states)
            test_labels = pd.Series(gmm.predict(X_test_pca), index=df_test.index)
            period_results['gmm'] = {'test_labels': smooth_states_minrun(test_labels, 2), 'n_states': n_states}
            
        if 'kmeans' in methods:
            n_states = optimal_n.get('kmeans', 3)
            print(f"  Training KMeans with {n_states} states...")
            km = fit_kmeans(X_train_pca, n_states=n_states)
            test_labels = pd.Series(km.predict(X_test_pca), index=df_test.index)
            period_results['kmeans'] = {'test_labels': smooth_states_minrun(test_labels, 2), 'n_states': n_states}

        results[f'period_{idx+1}'] = period_results
    return results

# ---------------------------
# Main
# ---------------------------
def main():
    print("\nRIGOROUS REGIME DETECTION (auto-n version)\n")
    print("Fetching data...")
    df = fetch_data(TICKER, FULL_START, FULL_END)
    print(f"Fetched {len(df)} rows for {TICKER}")

    print("Building features...")
    df_feat, all_features = build_comprehensive_features(df)
    print(f"Built {len(all_features)} features; {len(df_feat)} rows after dropna")
    df_feat.to_csv(os.path.join(OUTPUT_DIR, 'feature_data_for_analysis.csv'))

    print("Selecting stable features (train_end_date = 2010-12-31)...")
    selected, scores = select_stable_features(df_feat, all_features, train_end_date='2010-12-31', n_features=15)
    print("Selected features:", selected)
    scores.to_csv(os.path.join(OUTPUT_DIR, 'feature_stability_scores.csv'), index=False)

    print("\nWalking forward...")
    wf = walk_forward_validate(df_feat, selected, methods=['hmm', 'gmm', 'kmeans'])

    summary_rows = []
    for period_name, period_data in wf.items():
        print("\n" + "-"*60)
        print(period_name.upper())
        print("-"*60)
        for method, md in period_data.items():
            test_labels = md['test_labels']
            stability = compute_regime_stability(test_labels)
            crash_regime = choose_crash_regime_by_overlap(test_labels, CRASH_WINDOWS)
            
            # Corrected argument order in the following two lines
            precision, recall = precision_recall_basic(test_labels, crash_regime, CRASH_WINDOWS) if crash_regime is not None else (0.0, 0.0)
            event_rec, avg_lag = event_level_detection(test_labels, CRASH_WINDOWS, crash_regime, 3) if crash_regime is not None else (0.0, None)
            
            print(f"\n{method.upper()} (found {md['n_states']} states):")
            print(f"  n_regimes_observed: {stability['n_regimes_observed']}")
            print(f"  mean_duration: {stability['mean_duration']:.1f} days")
            print(f"  transition_rate: {stability['transition_rate']:.2%}")
            print(f"  crash_regime: {crash_regime}")
            print(f"  precision: {precision:.3f}, recall: {recall:.3f}")
            print(f"  event_recall: {event_rec:.3f}, avg_lag: {avg_lag}")

            summary_rows.append({'period': period_name, 'method': method, 'optimal_n_states': md['n_states'], **stability, 'crash_precision': precision, 'crash_recall': recall, 'event_recall': event_rec})
            
            # Added: Save the labels file for each method and period
            test_labels.to_csv(os.path.join(OUTPUT_DIR, f"{period_name}_{method}_test_labels.csv"))

        agreement = compute_inter_method_agreement(period_data)
        if agreement is not None:
            print("\nInter-method agreement (ARI / NMI):")
            print(agreement.to_string(index=False))

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "walk_forward_summary_auto_n.csv"), index=False)
        print("\nAggregated metrics across periods:")
        agg = summary_df.groupby('method').agg(
            optimal_n_states=('optimal_n_states', 'mean'),
            mean_duration=('mean_duration', 'mean'),
            transition_rate=('transition_rate', 'mean'),
            crash_precision=('crash_precision', 'mean'),
            crash_recall=('crash_recall', 'mean'),
            event_recall=('event_recall', 'mean')
        )
        print(agg.to_string())
        agg.to_csv(os.path.join(OUTPUT_DIR, "walk_forward_agg_auto_n.csv"))

    print("\nAll outputs saved in", OUTPUT_DIR)
    print("Completed.")

if __name__ == "__main__":
    main()