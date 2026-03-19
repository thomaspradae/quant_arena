#!/usr/bin/env python3
"""
regime_v2_full_with_outputs.py

Full regime detection pipeline with explicit model summaries and outputs you can bring back for inspection.

Usage:
    python3 regime_v2_full_with_outputs.py

Outputs (folder rigorous_regime_results/):
 - feature_data.csv
 - feature_stability_scores.csv
 - period_{i}_{method}_test_labels.csv
 - period_{i}_pca_info.json
 - period_{i}_{method}_model_summary.json
 - walkforward_summary.csv
 - pca_reports.json
 - run_report_summary.json
"""
import os
import sys
import json
import math
import logging
import inspect
from typing import Optional, Dict, Any, List, Tuple

import random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# deterministic
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regime_v2_full_with_outputs")

# Required libraries
try:
    import yfinance as yf
except Exception as e:
    raise ImportError("yfinance is required. pip install yfinance") from e

try:
    from arch import arch_model
except Exception as e:
    raise ImportError("arch is required. pip install arch") from e

try:
    import pomegranate as pg
    # check StudentTDistribution presence
    if not (hasattr(pg, "distributions") and hasattr(pg.distributions, "StudentTDistribution")):
        raise ImportError("pomegranate.distributions.StudentTDistribution is required for the Student-t HMM path.")
except Exception as e:
    raise ImportError("pomegranate with StudentTDistribution support is required. pip install pomegranate and ensure it's compatible.") from e

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except Exception:
    HMMLEARN_AVAILABLE = False
    GaussianHMM = None

# output dir
OUTPUT_DIR = "rigorous_regime_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# config
TICKER = "SPY"
FULL_START = "1997-01-01"
FULL_END = "2024-01-01"

WALK_FORWARD_PERIODS = [
    ("1997-01-01", "2010-12-31", "2011-01-01", "2015-12-31"),
    ("1997-01-01", "2015-12-31", "2016-01-01", "2020-12-31"),
    ("1997-01-01", "2020-12-31", "2021-01-01", "2024-01-01"),
]

CRASH_WINDOWS = [
    (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-03-31")),
    (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30")),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31")),
]

# ---------------------------
# Helpers
# ---------------------------
def save_json(obj, fp):
    with open(fp, "w") as f:
        json.dump(obj, f, default=lambda x: str(x), indent=2)
    logger.info(f"Wrote {fp}")

def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("No data fetched.")
    df = df[["Open","High","Low","Close","Adj Close","Volume"]].rename(
        columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    df.index = pd.to_datetime(df.index)
    return df

def build_comprehensive_features(df):
    df = df.copy()
    df['ret'] = df['adj_close'].pct_change()
    df['log_ret'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    df['abs_ret'] = df['log_ret'].abs()
    df['signed_abs_ret'] = np.sign(df['log_ret']) * df['abs_ret']
    for window in [5,10,21,42,63]:
        df[f'ma_ret_{window}'] = df['log_ret'].rolling(window, min_periods=max(1, window//3)).mean()
        df[f'vol_{window}'] = df['log_ret'].rolling(window, min_periods=max(1, window//3)).std() * np.sqrt(252)
    vol_short = df['log_ret'].rolling(10, min_periods=5).std() * np.sqrt(252)
    vol_long = df['log_ret'].rolling(63, min_periods=30).std() * np.sqrt(252)
    df['vol_ratio'] = vol_short / (vol_long.replace(0, np.nan))
    for window in [21,42,63]:
        df[f'mom_{window}'] = df['log_ret'].rolling(window, min_periods=max(1, window//3)).sum()
        df[f'skew_{window}'] = df['log_ret'].rolling(window, min_periods=max(5, window//3)).skew()
        df[f'kurt_{window}'] = df['log_ret'].rolling(window, min_periods=max(5, window//3)).kurt()
    prev_close = df['adj_close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    true_range = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    for window in [14,21]:
        df[f'atr_{window}'] = true_range.rolling(window, min_periods=max(3, window//2)).mean() / df['adj_close']
    df['gap'] = (df['open'] - prev_close) / prev_close
    df['vol_ma_ratio_21'] = df['volume'] / (df['volume'].rolling(21, min_periods=1).mean().replace(0,np.nan))
    df['vol_ma_ratio_63'] = df['volume'] / (df['volume'].rolling(63, min_periods=1).mean().replace(0,np.nan))
    df['log_volume'] = np.log(df['volume'] + 1)
    df['vwret_5'] = (df['log_ret'] * df['volume']).rolling(5, min_periods=3).sum() / (df['volume'].rolling(5, min_periods=3).sum().replace(0, np.nan))
    for window in [21,42,63]:
        df[f'trend_{window}'] = (df['adj_close'] / df['adj_close'].shift(window) - 1)
    exclude = ['open','high','low','close','adj_close','volume']
    feature_cols = [c for c in df.columns if c not in exclude]
    df_clean = df.dropna(subset=feature_cols)
    return df_clean, feature_cols

def robust_transform_df(df, feature_cols, method='winsorize', q=0.01):
    df2 = df.copy()
    if method == 'winsorize':
        for c in feature_cols:
            if c not in df2.columns: continue
            s = df2[c].dropna()
            if s.empty: continue
            lo, hi = s.quantile(q), s.quantile(1-q)
            df2[c] = df2[c].clip(lo, hi)
        return df2, None
    elif method in ('rank_gauss','quantile_gauss'):
        transformer = QuantileTransformer(output_distribution='normal', random_state=SEED, copy=True)
        arr = transformer.fit_transform(df2[feature_cols].values)
        df2.loc[:, feature_cols] = arr
        return df2, transformer
    else:
        raise ValueError("Unknown transform method")

def fit_garch_select(df, returns_col='log_ret', out_col='garch_sigma'):
    r = df[returns_col].dropna() * 100.0
    if len(r) < 200:
        raise RuntimeError("Not enough data for GARCH selection (need >=200)")
    candidates = [
        {'vol':'Garch','p':1,'q':1,'dist':'normal'},
        {'vol':'Garch','p':1,'q':1,'dist':'StudentsT'},
        {'vol':'Garch','p':1,'q':2,'dist':'StudentsT'},
    ]
    best_bic = np.inf
    best_res = None
    best_spec = None
    for spec in candidates:
        try:
            am = arch_model(r, vol=spec['vol'], p=spec['p'], q=spec['q'], dist=spec['dist'])
            res = am.fit(disp='off', show_warning=False)
            bic = res.bic
            logger.info(f"GARCH spec {spec} -> BIC {bic:.2f}")
            if bic < best_bic:
                best_bic = bic
                best_res = res
                best_spec = spec
        except Exception as e:
            logger.warning(f"GARCH spec {spec} failed: {e}")
            continue
    if best_res is None:
        raise RuntimeError("All GARCH fits failed.")
    cond_vol = best_res.conditional_volatility / 100.0
    cond_vol = cond_vol.reindex(df.index).ffill().bfill()
    df = df.copy()
    df[out_col] = cond_vol
    meta = {'chosen_spec': best_spec, 'bic': float(best_bic)}
    logger.info(f"Selected GARCH spec {best_spec}")
    return df, meta

# ---------------------------
# Detectors
# ---------------------------

class VolatilityRegimeDetector:
    def __init__(self, window=21, num_regimes=3, method='quantile'):
        self.window = window
        self.num_regimes = num_regimes
        self.method = method
        self.name = f"Vol_{window}_{num_regimes}"

    def detect(self, df):
        returns = df['close'].pct_change()
        vol = returns.rolling(window=self.window, min_periods=max(3, self.window//2)).std() * np.sqrt(252)
        vol = vol.dropna()
        if self.method == 'quantile':
            ql, qh = vol.quantile(0.01), vol.quantile(0.99)
            volc = vol.clip(ql,qh)
            if self.num_regimes == 2:
                thr = volc.median()
                labels = pd.Series(0, index=volc.index)
                labels[volc >= thr] = 1
            else:
                qs = np.linspace(0,1,self.num_regimes+1)[1:-1]
                ths = volc.quantile(qs)
                labels = pd.Series(0, index=volc.index)
                for i, th in enumerate(ths):
                    labels[volc >= th] = i + 1
            labels = labels.astype(int)
        else:
            X = vol.values.reshape(-1,1)
            km = KMeans(n_clusters=self.num_regimes, random_state=SEED, n_init=20).fit(X)
            centers = km.cluster_centers_.flatten()
            order = np.argsort(centers)
            mapping = {old:new for new,old in enumerate(order)}
            labels = pd.Series([mapping[l] for l in km.labels_], index=vol.index).astype(int)
        return labels

class TailAwareGMMDetector:
    def __init__(self, n_components=2, random_state=SEED):
        self.n = n_components
        self.random_state = random_state
        self.name = f"TailGMM_{n_components}_plus_tail"

    def detect(self, df_or_df_with_components, feature_cols=None):
        # Accepts DataFrame with features (could be PCA components)
        if feature_cols is None:
            returns = df_or_df_with_components['close'].pct_change().dropna()
            X = returns.values.reshape(-1,1)
            idx = returns.index
        else:
            sub = df_or_df_with_components[feature_cols].dropna()
            X = sub.values
            idx = sub.index
        if X.shape[0] < 10:
            raise RuntimeError("Not enough data for TailAwareGMM")
        gmm = GaussianMixture(n_components=self.n+1, covariance_type='full', random_state=self.random_state, n_init=10)
        gmm.fit(X)
        covs = gmm.covariances_
        if covs.ndim == 3:
            variances = np.array([np.linalg.det(cov) for cov in covs])
        else:
            variances = np.array([float(cov) for cov in covs])
        tail_idx = int(np.argmax(variances))
        labels = gmm.predict(X)
        means = np.array([gmm.means_[i].mean() for i in range(gmm.means_.shape[0])])
        non_tail = [i for i in range(len(means)) if i != tail_idx]
        sorted_non_tail = sorted(non_tail, key=lambda i: means[i])
        mapping = {orig:new for new, orig in enumerate(sorted_non_tail)}
        mapping[tail_idx] = self.n
        mapped = np.array([mapping[l] for l in labels])
        return pd.Series(mapped, index=idx).astype(int), {'gmm_means': gmm.means_.tolist(), 'gmm_covariances_shape': gmm.covariances_.shape, 'tail_idx': int(tail_idx)}

class StudentTHMMDetector:
    def __init__(self, num_regimes=3, n_init=5):
        self.num_regimes = num_regimes
        self.n_init = max(1, int(n_init))
        self.name = f"tHMM_{num_regimes}"
        self.model = None

    def detect(self, df, features=None, max_iters=200):
        returns = df['close'].pct_change().dropna()
        if features is None:
            feats = pd.DataFrame({'returns': returns, 'abs_returns': returns.abs()}).dropna()
        else:
            feats = df[features].dropna()
        if len(feats) < max(10, 3*self.num_regimes):
            raise RuntimeError("Not enough data for HMM")
        X = feats.values
        idx = feats.index

        # Initialize with KMeans
        km = KMeans(n_clusters=self.num_regimes, random_state=SEED, n_init=20).fit(X)
        assign = km.labels_

        # Build StudentT product distributions per state
        dists = []
        StudentT = pg.distributions.StudentTDistribution
        Product = pg.distributions.ProductDistribution
        Normal = pg.distributions.NormalDistribution
        for s in range(self.num_regimes):
            mask = assign == s
            if mask.sum() < 4:
                # fallback to multivariate gaussian/product-of-normals
                mu = X[mask].mean(axis=0) if mask.sum() > 0 else X.mean(axis=0)
                cov = np.cov(X.T) + 1e-6*np.eye(X.shape[1])
                try:
                    mg = pg.distributions.MultivariateGaussianDistribution(mu, cov)
                    dists.append(mg)
                    continue
                except Exception:
                    pass
                dims = []
                for dim in range(X.shape[1]):
                    col = X[mask][:,dim] if mask.sum()>0 else X[:,dim]
                    dims.append(Normal(float(col.mean()), float(np.std(col)+1e-8)))
                dists.append(Product(dims))
            else:
                block = X[mask]
                dims = []
                for dim in range(X.shape[1]):
                    col = block[:,dim]
                    kurt = stats.kurtosis(col, fisher=False, bias=False) if len(col)>3 else 3.0
                    df_est = max(3.0, min(30.0, 6.0 * (1.0 / (kurt - 3.0 + 1e-6) if kurt > 3.0 else 6.0)))
                    loc = float(np.mean(col))
                    scale = float(np.std(col) + 1e-8)
                    dims.append(StudentT(df=df_est, loc=loc, scale=scale))
                dists.append(Product(dims))

        # Build HMM
        model = pg.HiddenMarkovModel()
        states = [pg.HiddenMarkovModel.State(dist, name=f"S{s}") for s, dist in enumerate(dists)]
        for st in states:
            model.add_state(st)
        n_states = len(states)
        for st in states:
            model.add_transition(model.start, st, 1.0/n_states)
        for i, si in enumerate(states):
            model.add_transition(si, si, 0.7)
            for j, sj in enumerate(states):
                if i != j:
                    model.add_transition(si, sj, 0.3/(n_states-1))
        model.bake()

        X_seq = [X.astype(float)]
        # from_samples expects distribution class; pass class of first dist for API
        try:
            model = pg.HiddenMarkovModel.from_samples(dists[0].__class__, n_components=self.num_regimes, X=X_seq, algorithm='baum-welch', n_jobs=1, max_iterations=max_iters)
        except Exception as ex:
            # fallback attempt using ProductDistribution class directly
            try:
                model = pg.HiddenMarkovModel.from_samples(pg.distributions.ProductDistribution, n_components=self.num_regimes, X=X_seq, algorithm='baum-welch', n_jobs=1, max_iterations=max_iters)
            except Exception as ex2:
                raise RuntimeError(f"pomegranate HMM from_samples failed: {ex} ; fallback also failed: {ex2}")

        logp, path = model.viterbi(X.astype(float))
        decoded = [s.name for (p,s) in path[1:]]
        numeric = []
        for nm in decoded:
            try:
                if isinstance(nm, str) and nm.startswith("S"):
                    numeric.append(int(nm[1:]))
                elif hasattr(nm, 'name') and isinstance(nm.name, str) and nm.name.startswith("S"):
                    numeric.append(int(nm.name[1:]))
                else:
                    numeric.append(0)
            except Exception:
                numeric.append(0)
        regimes = pd.Series(numeric, index=idx).astype(int)
        # store model
        self.model = model
        # return regimes and a summary dict
        meta = {
            'n_states': self.num_regimes,
            'logp_first': float(logp) if hasattr(logp, 'item') else float(logp),
            'using_pomegranate': True
        }
        return regimes, meta

# ---------------------------
# Metrics & utilities
# ---------------------------

def compute_regime_stability(labels: pd.Series):
    s = labels.reset_index(drop=True)
    if s.empty:
        return {'mean_duration':0.0,'transition_rate':0.0,'entropy':0.0,'n_regimes_observed':0}
    runs = []
    cur = s.iloc[0]; run = 1
    for val in s.iloc[1:]:
        if val == cur:
            run += 1
        else:
            runs.append(run)
            cur = val; run = 1
    runs.append(run)
    mean_duration = float(np.mean(runs)) if runs else 0.0
    transitions = max(len(runs)-1, 0)
    transition_rate = transitions / max(1, len(s))
    counts = pd.Series(labels).value_counts()
    probs = counts / counts.sum() if counts.sum()>0 else counts
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
    max_entropy = math.log2(len(counts)) if len(counts)>0 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return {'mean_duration': mean_duration, 'transition_rate': transition_rate, 'entropy': normalized_entropy, 'n_regimes_observed': int(len(counts))}

def choose_crash_regime_by_overlap(labels: pd.Series, crash_windows):
    crash_days = set()
    for a,b in crash_windows:
        crash_days.update(pd.date_range(a,b,freq='D'))
    counts, totals = {}, {}
    for lab in np.unique(labels.dropna()):
        idxs = labels[labels == lab].index
        totals[lab] = len(idxs)
        counts[lab] = sum(1 for d in idxs if d in crash_days)
    fractions = {lab: counts.get(lab,0)/totals.get(lab,1) for lab in totals.keys()}
    if not fractions:
        return None
    best = max(fractions.items(), key=lambda kv:(kv[1], counts.get(kv[0],0)))[0]
    return int(best)

def precision_recall_basic(predicted_labels, crash_regime, crash_windows):
    s = predicted_labels.dropna()
    crash_days = set()
    for a,b in crash_windows:
        crash_days.update(pd.date_range(a,b,freq='D'))
    preds = s[s == crash_regime].index
    tp = sum(1 for d in preds if d in crash_days)
    total_pred = len(preds)
    total_crash_days = sum(1 for d in s.index if d in crash_days)
    precision = tp / max(1, total_pred)
    recall = tp / max(1, total_crash_days)
    return precision, recall

def event_level_detection(predicted_labels, crash_windows, crash_regime, slack_days=3):
    s = predicted_labels.dropna()
    detected = 0; lags = []
    for a,b in crash_windows:
        window_range = pd.date_range(a - pd.Timedelta(days=slack_days), b + pd.Timedelta(days=slack_days), freq='D')
        hits = [d for d in s[s==crash_regime].index if d in window_range]
        if hits:
            detected += 1
            first_hit = min(hits)
            lags.append((first_hit - a).days)
    recall_event = detected / max(1, len(crash_windows))
    avg_lag = float(np.mean(lags)) if lags else None
    return recall_event, avg_lag

# ---------------------------
# Walk-forward routine
# ---------------------------

def walk_forward_validate(df_full, feature_cols, detectors):
    results = {}
    for idx, (train_start, train_end, test_start, test_end) in enumerate(WALK_FORWARD_PERIODS):
        period_name = f"period_{idx+1}"
        logger.info("="*80)
        logger.info(f"{period_name} : Train {train_start}->{train_end} ; Test {test_start}->{test_end}")
        logger.info("="*80)
        df_train = df_full[(df_full.index >= train_start) & (df_full.index <= train_end)]
        df_test = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)]
        if len(df_train) < 200 or len(df_test) < 20:
            logger.warning("Skipping period due to insufficient data")
            continue

        # Fit GARCH on train
        df_train_local = df_train.copy()
        df_train_local = df_train_local.assign(log_ret = df_train_local['log_ret'])
        garch_df_train, garch_meta = fit_garch_select(df_train_local, returns_col='log_ret', out_col='garch_sigma')
        save_json(garch_meta, os.path.join(OUTPUT_DIR, f"{period_name}_garch_meta.json"))

        garch_sigma_full = garch_df_train['garch_sigma'].reindex(df_full.index).ffill().bfill()
        df_full_loc = df_full.copy()
        df_full_loc['garch_sigma'] = garch_sigma_full

        # PCA on training features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(df_train[feature_cols].values)
        pca = PCA(n_components=0.95, svd_solver='full', random_state=SEED)
        X_train_pca = pca.fit_transform(X_train_scaled)

        X_test_scaled = scaler.transform(df_test[feature_cols].values)
        X_test_pca = pca.transform(X_test_scaled)

        # mapping PCA components into DF for detectors that accept feature columns
        df_tr_pca = pd.DataFrame(X_train_pca, index=df_train.index, columns=[f"pc{i}" for i in range(X_train_pca.shape[1])])
        df_te_pca = pd.DataFrame(X_test_pca, index=df_test.index, columns=df_tr_pca.columns)

        period_results = {}

        for det in detectors:
            name = det.name if hasattr(det, 'name') else det.__class__.__name__
            logger.info(f"Fitting detector: {name}")
            try:
                if isinstance(det, VolatilityRegimeDetector):
                    combined = pd.concat([df_train, df_test]).sort_index()
                    labels_full = det.detect(combined)
                    train_labels = labels_full.reindex(df_train.index).ffill().bfill()
                    test_labels = labels_full.reindex(df_test.index).ffill().bfill()
                    meta = {'method': 'volatility_quantile', 'window': det.window, 'num_regimes': det.num_regimes}
                    period_results[name] = {'train_labels': train_labels, 'test_labels': test_labels, 'meta': meta}
                    # save model summary
                    save_json(meta, os.path.join(OUTPUT_DIR, f"{period_name}_{name}_model_summary.json"))
                elif isinstance(det, TailAwareGMMDetector):
                    # run on PCA space (training then training+test)
                    labels_train, meta_train = det.detect(df_tr_pca, feature_cols=list(df_tr_pca.columns))
                    labels_train = labels_train.reindex(df_train.index).ffill().bfill()
                    combined_pca = pd.concat([df_tr_pca, df_te_pca])
                    labels_combined, meta_combined = det.detect(combined_pca, feature_cols=list(combined_pca.columns))
                    labels_test = labels_combined.reindex(df_test.index).ffill().bfill()
                    meta = {'method': 'tail_aware_gmm', 'gmm_meta': meta_combined, 'gmm_train_meta': meta_train}
                    period_results[name] = {'train_labels': labels_train, 'test_labels': labels_test, 'meta': meta}
                    save_json(meta, os.path.join(OUTPUT_DIR, f"{period_name}_{name}_model_summary.json"))
                elif isinstance(det, StudentTHMMDetector):
                    # fit on combined train+test so we can decode across whole window (keeps transitions continuous)
                    combined = pd.concat([df_train, df_test]).sort_index()
                    regimes_full, meta = det.detect(combined, features=None, max_iters=200)
                    train_labels = regimes_full.reindex(df_train.index).ffill().bfill()
                    test_labels = regimes_full.reindex(df_test.index).ffill().bfill()
                    meta.update({'method': 'pomegranate_student_t_hmm', 'n_states': det.num_regimes})
                    period_results[name] = {'train_labels': train_labels, 'test_labels': test_labels, 'meta': meta, 'model_obj': det.model}
                    # Save model summary: include some introspection of model if possible (transition matrix sizes etc.)
                    try:
                        # pomegranate HiddenMarkovModel has attribute 'states' and 'edge' info; we extract simple info
                        model = det.model
                        states_info = [s.name for s in model.states if hasattr(s, 'name')]
                        trans_edges = []
                        # iterate transitions via model.distributions or model._transitions if available
                        # We'll collect number of states and model string repr
                        model_repr = repr(model) if hasattr(model, '__repr__') else str(type(model))
                        model_summary = {'model_repr': model_repr, 'n_states': len(states_info), 'states': states_info}
                    except Exception as e:
                        model_summary = {'error_introspecting_model': str(e)}
                    meta['model_summary'] = model_summary
                    save_json(meta, os.path.join(OUTPUT_DIR, f"{period_name}_{name}_model_summary.json"))
                else:
                    raise RuntimeError("Unknown detector type.")
            except Exception as e:
                logger.error(f"Detector {name} failed: {e}")
                raise
        # save labels CSVs and period-level PCA info
        for method_name, info in period_results.items():
            test_series = info['test_labels']
            out_fp = os.path.join(OUTPUT_DIR, f"{period_name}_{method_name}_test_labels.csv")
            pd.DataFrame({'date': test_series.index, 'regime': test_series.values}).to_csv(out_fp, index=False)
            logger.info(f"Wrote {out_fp}")

        pca_info = {
            'period': period_name,
            'train_window': [train_start, train_end],
            'n_pca_components': int(pca.n_components_),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'pca_components_shape': pca.components_.shape,
            'selected_features': feature_cols
        }
        save_json(pca_info, os.path.join(OUTPUT_DIR, f"{period_name}_pca_info.json"))

        results[period_name] = period_results

    return results

# ---------------------------
# Agreement & summary
# ---------------------------

def compute_inter_method_agreement(period_dict):
    methods = list(period_dict.keys())
    if len(methods) < 2:
        return None
    idxs = [period_dict[m]['test_labels'].index for m in methods]
    common = idxs[0]
    for idx in idxs[1:]:
        common = common.intersection(idx)
    if len(common) < 2:
        return None
    rows = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            a = period_dict[methods[i]]['test_labels'].loc[common].values
            b = period_dict[methods[j]]['test_labels'].loc[common].values
            ari = adjusted_rand_score(a,b)
            nmi = normalized_mutual_info_score(a,b)
            rows.append({'method1': methods[i], 'method2': methods[j], 'ari': ari, 'nmi': nmi})
    return pd.DataFrame(rows)

# ---------------------------
# Stable feature selection
# ---------------------------

def select_stable_features(df, feature_cols, train_end_date='2010-12-31', n_features=20):
    df_train = df[df.index <= train_end_date]
    rows = []
    for feat in feature_cols:
        s = df_train[feat].dropna()
        if len(s) < 100:
            continue
        acf1 = s.autocorr(lag=1)
        acf_score = 1.0 - abs(acf1 if not pd.isna(acf1) else 0.0)
        mid = len(s)//2
        v1 = s.iloc[:mid].std()
        v2 = s.iloc[mid:].std()
        var_stability = min(v1, v2) / max(v1, v2) if max(v1,v2) > 0 else 0.0
        outlier_frac = (np.abs(s - s.mean()) > 3*s.std()).mean()
        outlier_score = 1.0 - outlier_frac
        score = acf_score*0.3 + var_stability*0.4 + outlier_score*0.3
        rows.append({'feature': feat, 'score': float(score), 'acf1': float(acf1) if not pd.isna(acf1) else None, 'var_stability': float(var_stability), 'outlier_frac': float(outlier_frac)})
    df_scores = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    selected = df_scores.head(n_features)['feature'].tolist()
    return selected, df_scores

# ---------------------------
# Main
# ---------------------------

def main():
    logger.info("Running full regime detection with pomegranate StudentT HMM (prints + saves model summaries)")
    logger.info(f"Fetching {TICKER} {FULL_START}->{FULL_END}")
    df = fetch_data(TICKER, FULL_START, FULL_END)
    logger.info(f"Fetched {len(df)} rows")

    logger.info("Building comprehensive features...")
    df_feat, feature_cols = build_comprehensive_features(df)
    logger.info(f"Built {len(feature_cols)} features; {len(df_feat)} rows after dropna")
    df_feat.to_csv(os.path.join(OUTPUT_DIR, "feature_data.csv"))

    logger.info("Selecting stable features using train_end_date=2010-12-31...")
    selected, scores = select_stable_features(df_feat, feature_cols, train_end_date='2010-12-31', n_features=20)
    logger.info("Selected features (top 20):")
    logger.info(selected)
    scores.to_csv(os.path.join(OUTPUT_DIR, "feature_stability_scores.csv"), index=False)

    # detectors
    detectors = [
        VolatilityRegimeDetector(window=21, num_regimes=3, method='quantile'),
        TailAwareGMMDetector(n_components=2, random_state=SEED),
        StudentTHMMDetector(num_regimes=3, n_init=6)
    ]

    # run walk-forward
    wf = walk_forward_validate(df_feat, selected, detectors)

    # summarise and save
    summary_rows = []
    for period_name, period_res in wf.items():
        logger.info(f"Summarizing {period_name}")
        for method_name, info in period_res.items():
            test_labels = info['test_labels']
            stab = compute_regime_stability(test_labels)
            crash_regime = choose_crash_regime_by_overlap(test_labels, CRASH_WINDOWS)
            if crash_regime is not None:
                precision, recall = precision_recall_basic(test_labels, crash_regime, CRASH_WINDOWS)
                event_rec, avg_lag = event_level_detection(test_labels, CRASH_WINDOWS, crash_regime, slack_days=3)
            else:
                precision, recall = (None, None)
                event_rec, avg_lag = (None, None)
            summary_rows.append({
                'period': period_name,
                'method': method_name,
                'n_regimes_observed': stab['n_regimes_observed'],
                'mean_duration': stab['mean_duration'],
                'transition_rate': stab['transition_rate'],
                'entropy': stab['entropy'],
                'crash_regime': crash_regime,
                'crash_precision': precision,
                'crash_recall': recall,
                'event_recall': event_rec,
                'event_avg_lag': avg_lag
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "walkforward_summary.csv"), index=False)
    logger.info("Wrote walkforward_summary.csv")

    # agreement
    for period_name, period_res in wf.items():
        agree_df = compute_inter_method_agreement(period_res)
        if agree_df is not None:
            agree_fp = os.path.join(OUTPUT_DIR, f"{period_name}_agreement.csv")
            agree_df.to_csv(agree_fp, index=False)
            logger.info(f"Wrote {agree_fp}")

    # PCA reports across periods
    pca_reports = []
    for idx, (train_start, train_end, _, _) in enumerate(WALK_FORWARD_PERIODS):
        df_train = df_feat[(df_feat.index >= train_start) & (df_feat.index <= train_end)]
        if df_train.empty: continue
        scaler = RobustScaler()
        Xs = scaler.fit_transform(df_train[selected].values)
        pca = PCA(n_components=min(10, len(selected)), random_state=SEED)
        Xp = pca.fit_transform(Xs)
        comps = pca.components_
        top_loadings = {}
        for ci in range(min(3, comps.shape[0])):
            comp = comps[ci]
            idxs = np.argsort(np.abs(comp))[::-1][:8]
            top_loadings[f"pc{ci+1}"] = [(selected[i], float(comp[i])) for i in idxs]
        pca_reports.append({
            'period': f'period_{idx+1}',
            'train_window': [train_start, train_end],
            'n_components': int(pca.n_components_),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'top_loadings': top_loadings
        })
    save_json(pca_reports, os.path.join(OUTPUT_DIR, "pca_reports.json"))

    report = {
        'n_periods': len(wf),
        'periods': list(wf.keys()),
        'selected_features': selected,
        'notes': 'StudentT HMM via pomegranate.distributions.StudentTDistribution; GMM + Tail component; PCA reported.'
    }
    save_json(report, os.path.join(OUTPUT_DIR, "run_report_summary.json"))

    # print succinct human-readable summaries to console (so you can copy/paste)
    print("\n=== RUN SUMMARY ===")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("\nSelected features (top):")
    for f in selected:
        print("  ", f)
    print("\nPer-period detector summaries (first few lines):")
    for period_name, period_res in wf.items():
        print(f"\n{period_name}:")
        for method_name, info in period_res.items():
            print(f"  {method_name}:")
            meta = info.get('meta', {})
            # print stability quickly
            stab = compute_regime_stability(info['test_labels'])
            print(f"    n_regimes_observed: {stab['n_regimes_observed']}, mean_duration: {stab['mean_duration']:.1f} days, entropy: {stab['entropy']:.3f}")
            print(f"    meta keys: {list(meta.keys())}")
            # print first 5 label dates -> regimes
            head = info['test_labels'].head(5)
            for d,v in zip(head.index, head.values):
                print(f"      {d.date()} -> {v}")
    print("\nDone. Bring the files in rigorous_regime_results/ (walkforward_summary.csv, pca_reports.json, *_model_summary.json) and paste them back here for detailed interpretation.")

if __name__ == "__main__":
    main()
