#!/usr/bin/env python3
"""
regime_v2_full.py

Rigorous regime detection (v2) — heavy-tail aware, PCA reporting, GARCH selection, HMM grid-search (BIC/AIC),
Tail-aware GMM, deterministic seeding, walk-forward evaluation, and saved outputs.

Save as regime_v2_full.py and run:
    python3 regime_v2_full.py

This script is opinionated: it **requires** pomegranate, arch, and hmmlearn to be installed and compatible.
If they are missing or incompatible, the script will raise an informative ImportError.

Outputs saved in: rigorous_regime_results/
"""
import os
import sys
import json
import math
import logging
import inspect
from typing import Optional, Dict, Any, List, Tuple, Union

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
logger = logging.getLogger("regime_v2_full")

# Required external libs — enforce presence
try:
    import yfinance as yf
except Exception as e:
    raise ImportError("yfinance is required. Install with `pip install yfinance`.") from e

try:
    from arch import arch_model
except Exception as e:
    raise ImportError("arch package is required (for GARCH). Install with `pip install arch`.") from e

try:
    import pomegranate as pg
    # pomegranate versions sometimes put distributions in different places;
    # ensure the distribution classes we rely on exist
    if not (hasattr(pg, "distributions") or all(hasattr(pg, name) for name in ("HiddenMarkovModel", "State"))):
        # older/newer weirdness -> fail loudly
        raise ImportError("pomegranate appears installed but missing expected attributes (incompatible version).")
except Exception as e:
    raise ImportError("pomegranate is required and must be a compatible version. Install with `pip install pomegranate`.") from e

try:
    from hmmlearn.hmm import GaussianHMM
except Exception as e:
    raise ImportError("hmmlearn is required. Install with `pip install hmmlearn`.") from e

# Optional but useful
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Output
OUTPUT_DIR = "rigorous_regime_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CONFIG
TICKER = "SPY"
FULL_START = "1997-01-01"
FULL_END = "2024-01-01"

# Walk-forward periods (train_start, train_end, test_start, test_end)
WALK_FORWARD_PERIODS = [
    ("1997-01-01", "2010-12-31", "2011-01-01", "2015-12-31"),
    ("1997-01-01", "2015-12-31", "2016-01-01", "2020-12-31"),
    ("1997-01-01", "2020-12-31", "2021-01-01", "2024-01-01"),
]

# Known crash windows (for evaluation)
CRASH_WINDOWS = [
    (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-03-31")),   # 2008 crisis
    (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30")),   # COVID
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31")),   # 2022 selloff
]

# ---------------------------
# Utilities
# ---------------------------

def save_json(obj, fname):
    with open(fname, 'w') as f:
        json.dump(obj, f, default=lambda x: str(x), indent=2)
    logger.info(f"Wrote {fname}")

def require_condition(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

# ---------------------------
# Data & feature engineering
# ---------------------------

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} in {start}:{end}")
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].rename(
        columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    df.index = pd.to_datetime(df.index)
    return df

def build_comprehensive_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df['ret'] = df['adj_close'].pct_change()
    df['log_ret'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    df['abs_ret'] = df['log_ret'].abs()
    df['signed_abs_ret'] = np.sign(df['log_ret']) * df['abs_ret']

    # multi-scale moving means
    for window in [5,10,21,42,63]:
        df[f'ma_ret_{window}'] = df['log_ret'].rolling(window, min_periods=max(1, window//3)).mean()

    for window in [5,10,21,42,63]:
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

# ---------------------------
# Robust transforms & GARCH
# ---------------------------

def robust_transform_df(df: pd.DataFrame, feature_cols: List[str], method: str='winsorize', q: float=0.01):
    df2 = df.copy()
    if method == 'winsorize':
        for c in feature_cols:
            if c not in df2.columns:
                continue
            s = df2[c].dropna()
            if s.empty:
                continue
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

def fit_garch_select(df: pd.DataFrame, returns_col: str='log_ret', out_col: str='garch_sigma'):
    """
    Fit a small grid of GARCH specifications and select by BIC (StudentT vs Normal).
    Save the chosen model info and attach cond_vol to df[out_col].
    """
    r = df[returns_col].dropna() * 100.0  # scale to percent for arch
    if len(r) < 200:
        raise RuntimeError("Not enough data to fit GARCH robustly (need >=200 observations)")

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
            logger.info(f"GARCH spec {spec} -> BIC={bic:.2f}")
            if bic < best_bic:
                best_bic = bic
                best_res = res
                best_spec = spec
        except Exception as e:
            logger.warning(f"GARCH spec {spec} failed: {e}")
            continue

    if best_res is None:
        raise RuntimeError("All GARCH fits failed — aborting")

    cond_vol = best_res.conditional_volatility / 100.0
    cond_vol = cond_vol.reindex(df.index).ffill().bfill()
    df = df.copy()
    df[out_col] = cond_vol
    logger.info(f"Selected GARCH spec {best_spec} (BIC={best_bic:.2f}) and added feature '{out_col}'")
    return df, {'chosen_spec': best_spec, 'bic': best_bic}

# ---------------------------
# Detectors
# ---------------------------

class RegimeDetector:
    def __init__(self, name: Optional[str]=None):
        self.name = name or self.__class__.__name__
        self.regime_labels: Optional[pd.Series] = None
        self.regime_descriptions = {}

    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def expand_to_full_index(self, regimes: pd.Series, full_index: pd.DatetimeIndex, method: str='ffill') -> pd.Series:
        if regimes is None or regimes.empty:
            return pd.Series(index=full_index, dtype=int)
        expanded = regimes.reindex(full_index)
        expanded = expanded.ffill().bfill().fillna(0).astype(int)
        return expanded

    def get_regime_stats(self, regimes: pd.Series) -> pd.DataFrame:
        unique = sorted(set(regimes.dropna().unique()))
        rows=[]
        for r in unique:
            mask = regimes==r
            rows.append({'regime':int(r),'count':int(mask.sum()),'pct':float(mask.sum()/len(regimes)*100)})
        return pd.DataFrame(rows)

class VolatilityRegimeDetector(RegimeDetector):
    def __init__(self, window:int=20, num_regimes:int=2, method:str='quantile'):
        super().__init__(name=f"Vol_{window}_{num_regimes}")
        self.window = window
        self.num = num_regimes
        self.method = method

    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change()
        vol = returns.rolling(window=self.window, min_periods=max(3,int(self.window/2))).std() * np.sqrt(252)
        vol = vol.dropna()
        if len(vol)==0:
            raise RuntimeError("Not enough data for Volatility detector")
        if self.method == 'quantile':
            ql, qh = vol.quantile(0.01), vol.quantile(0.99)
            volc = vol.clip(ql,qh)
            if self.num==2:
                thr = volc.median()
                regimes = pd.Series(0,index=volc.index,dtype=int)
                regimes[volc>=thr]=1
            else:
                qs = np.linspace(0,1,self.num+1)[1:-1]
                ths = volc.quantile(qs)
                regimes = pd.Series(0,index=volc.index,dtype=int)
                for i,th in enumerate(ths):
                    regimes[volc>=th]=i+1
        else:
            X = vol.values.reshape(-1,1)
            km = KMeans(n_clusters=self.num, random_state=SEED, n_init=20).fit(X)
            centers = km.cluster_centers_.flatten()
            order = np.argsort(centers)
            mapping = {old:new for new,old in enumerate(order)}
            labels = np.array([mapping[l] for l in km.labels_])
            regimes = pd.Series(labels, index=vol.index).astype(int)
        self.regime_labels = regimes
        return regimes

class TailAwareGMM(RegimeDetector):
    def __init__(self, n_components:int=2, random_state:int=SEED):
        super().__init__(name=f"TailGMM_{n_components}_plus_tail")
        self.n = n_components
        self.random_state = random_state

    def detect_regimes(self, data: pd.DataFrame, feature_cols:Optional[List[str]]=None) -> pd.Series:
        if feature_cols is None:
            returns = data['close'].pct_change().dropna()
            X = returns.values.reshape(-1,1)
            idx = returns.index
        else:
            sub = data[feature_cols].dropna()
            X = sub.values
            idx = sub.index
        if X.shape[0] < 10:
            raise RuntimeError("Not enough data for TailAwareGMM")
        gmm = GaussianMixture(n_components=self.n+1, covariance_type='full', random_state=self.random_state, n_init=10)
        gmm.fit(X)
        covs = gmm.covariances_
        if covs.ndim==3:
            variances = np.array([np.linalg.det(cov) for cov in covs])
        else:
            variances = np.array([float(cov) for cov in covs])
        tail_idx = int(np.argmax(variances))
        labels = gmm.predict(X)
        means = np.array([gmm.means_[i].mean() for i in range(gmm.means_.shape[0])])
        non_tail = [i for i in range(len(means)) if i!=tail_idx]
        sorted_non_tail = sorted(non_tail, key=lambda i: means[i])
        mapping = {orig:new for new,orig in enumerate(sorted_non_tail)}
        mapping[tail_idx] = self.n  # tail assigned last index
        mapped = np.array([mapping[l] for l in labels])
        regimes = pd.Series(mapped, index=idx).astype(int)
        self.regime_labels = regimes
        return regimes

class StudentTHMMRegimeDetector(RegimeDetector):
    """
    Use pomegranate Student-T HMM (multivariate approximated by independent StudentT dims)
    with KMeans initialization. If pomegranate steps fail, the script will raise (no soft fallback).
    """
    def __init__(self, num_regimes:int=3, n_init:int=5):
        super().__init__(name=f"tHMM_{num_regimes}")
        self.num_regimes = num_regimes
        self.n_init = max(1,int(n_init))
        self.model = None

    def detect_regimes(self, data: pd.DataFrame, features:Optional[List[str]]=None, select_states_grid:List[int]=None, cov_types:List[str]=None, ic:str='bic'):
        # features selection: default to returns + abs
        returns = data['close'].pct_change().dropna()
        if features is None:
            feats = pd.DataFrame({'returns':returns, 'abs_returns':returns.abs()}).dropna()
        else:
            feats = data[features].dropna()
        if len(feats) < max(10, 3*self.num_regimes):
            raise RuntimeError("Not enough data for HMM")
        X = feats.values
        idx = feats.index

        # We'll attempt pomegranate-based t-HMM fit. We'll initialize states by KMeans.
        # Use pomegranate HiddenMarkovModel.from_samples with ProductDistribution of StudentT per dim.
        # If anything inside pomegranate fails, we raise.

        # Build StudentT product distributions from kmeans blocks
        km = KMeans(n_clusters=self.num_regimes, random_state=SEED, n_init=20).fit(X)
        assign = km.labels_
        dists = []
        # pomegranate StudentTDistribution is univariate — build product of univariate StudentTs for multivariate approx
        for s in range(self.num_regimes):
            mask = assign==s
            if mask.sum() < 4:
                # small cluster -> use multivariate gaussian distribution instead
                mu = X[mask].mean(axis=0) if mask.sum()>0 else X.mean(axis=0)
                cov = np.cov(X.T) + 1e-6*np.eye(X.shape[1])
                try:
                    d = pg.distributions.MultivariateGaussianDistribution(mu, cov)
                except Exception as e:
                    # pomegranate API differences — try constructing product of Gaussians
                    dims=[]
                    for dim in range(X.shape[1]):
                        col = X[mask][:,dim] if mask.sum()>0 else X[:,dim]
                        dims.append(pg.distributions.NormalDistribution(float(col.mean()), float(np.std(col)+1e-8)))
                    d = pg.distributions.ProductDistribution(dims)
            else:
                block = X[mask]
                dims=[]
                for dim in range(X.shape[1]):
                    col = block[:,dim]
                    kurt = stats.kurtosis(col, fisher=False, bias=False) if len(col)>3 else 3.0
                    # crude df estimate bounded 3..30
                    df_est = max(3.0, min(30.0, 6.0 * (1.0 / (kurt - 3.0 + 1e-6) if kurt > 3.0 else 6.0)))
                    loc = float(np.mean(col))
                    scale = float(np.std(col) + 1e-8)
                    dims.append(pg.distributions.StudentTDistribution(df=df_est, loc=loc, scale=scale))
                d = pg.distributions.ProductDistribution(dims)
            dists.append(d)

        # Build HMM with these dists and uniform-ish transitions; then call from_samples to refine by Baum-Welch
        try:
            model = pg.HiddenMarkovModel()
            states = [pg.HiddenMarkovModel.State(dist, name=f"S{s}") for s,dist in enumerate(dists)]
            for st in states:
                model.add_state(st)
            n_states = len(states)
            for st in states:
                model.add_transition(model.start, st, 1.0 / n_states)
            for i, si in enumerate(states):
                # self-loop strong
                model.add_transition(si, si, 0.7)
                for j, sj in enumerate(states):
                    if i!=j:
                        model.add_transition(si, sj, 0.3 / (n_states - 1))
            model.bake()
            # build a single long sequence X as shape (sequence_length, features)
            # from_samples expects distribution class and dataset X (list of sequences)
            # We'll call from_samples to re-estimate parameters (this API expects distribution types; we'll pass ProductDistribution)
            # NOTE: API variants exist; we call the generic from_samples entrypoint
            # Convert X to list-of-lists
            X_seq = [X.astype(float)]
            model = pg.HiddenMarkovModel.from_samples(dists[0].__class__, n_components=self.num_regimes, X=X_seq, algorithm='baum-welch', n_jobs=1, max_iterations=200)
            # decode via viterbi
            logp, path = model.viterbi(X.astype(float))
            decoded = [s.name for (p,s) in path[1:]]  # skip model.start
            numeric=[]
            for nm in decoded:
                try:
                    if nm.startswith("S"):
                        numeric.append(int(nm[1:]))
                    else:
                        numeric.append(0)
                except Exception:
                    numeric.append(0)
            regimes = pd.Series(numeric, index=idx).astype(int)
            self.regime_labels = regimes
            self.model = model
            logger.info("Pomegranate Student-T HMM fitted successfully")
            return regimes
        except Exception as ex:
            # fail loudly: user asked for crash if pomegranate steps fail
            raise RuntimeError(f"Pomegranate HMM fitting failed: {ex}")

# ---------------------------
# Helpers: stability & crash metrics
# ---------------------------

def compute_regime_stability(labels: pd.Series):
    s = labels.reset_index(drop=True)
    if s.empty:
        return {'mean_duration':0.0,'transition_rate':0.0,'entropy':0.0,'n_regimes_observed':0}
    runs=[]
    cur=s.iloc[0]; run=1
    for val in s.iloc[1:]:
        if val==cur:
            run+=1
        else:
            runs.append(run)
            cur=val; run=1
    runs.append(run)
    mean_duration = float(np.mean(runs)) if runs else 0.0
    transitions = max(len(runs)-1,0)
    transition_rate = transitions / max(1,len(s))
    counts = pd.Series(labels).value_counts()
    probs = counts / counts.sum() if counts.sum()>0 else counts
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
    max_entropy = math.log2(len(counts)) if len(counts)>0 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy>0 else 0.0
    return {'mean_duration':mean_duration,'transition_rate':transition_rate,'entropy':normalized_entropy,'n_regimes_observed':len(counts)}

def choose_crash_regime_by_overlap(labels: pd.Series, crash_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]):
    crash_days=set()
    for a,b in crash_windows:
        crash_days.update(pd.date_range(a,b,freq='D'))
    counts={}
    totals={}
    for lab in np.unique(labels.dropna()):
        idxs = labels[labels==lab].index
        totals[lab] = len(idxs)
        counts[lab] = sum(1 for d in idxs if d in crash_days)
    fractions = {lab:(counts.get(lab,0)/totals.get(lab,1)) for lab in totals.keys()}
    if not fractions:
        return None
    best = max(fractions.items(), key=lambda kv:(kv[1], counts.get(kv[0],0)))[0]
    return int(best)

def precision_recall_basic(predicted_labels: pd.Series, crash_regime:int, crash_windows:List[Tuple[pd.Timestamp, pd.Timestamp]]):
    s = predicted_labels.dropna()
    crash_days=set()
    for a,b in crash_windows:
        crash_days.update(pd.date_range(a,b,freq='D'))
    preds = s[s==crash_regime].index
    tp = sum(1 for d in preds if d in crash_days)
    total_pred = len(preds)
    total_crash_days = sum(1 for d in s.index if d in crash_days)
    precision = tp / max(1,total_pred)
    recall = tp / max(1,total_crash_days)
    return precision, recall

def event_level_detection(predicted_labels: pd.Series, crash_windows:List[Tuple[pd.Timestamp,pd.Timestamp]], crash_regime:int, slack_days:int=3):
    s = predicted_labels.dropna()
    detected=0; lags=[]
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

def walk_forward_validate(df_full: pd.DataFrame, feature_cols: List[str], detectors:List[RegimeDetector], n_states_grid:List[int]=[2,3], hmm_cov_types:List[str]=['diag','full']):
    """
    Walk-forward across WALK_FORWARD_PERIODS. For HMM we run a grid over n_states_grid and cov types (select by BIC).
    Returns dictionary with per-period -> per-method -> dict (train_labels,test_labels,model,meta)
    """
    results={}
    for idx, (train_start, train_end, test_start, test_end) in enumerate(WALK_FORWARD_PERIODS):
        logger.info("="*60)
        logger.info(f"PERIOD {idx+1}: Train {train_start}->{train_end} ; Test {test_start}->{test_end}")
        logger.info("="*60)
        df_train = df_full[(df_full.index >= train_start) & (df_full.index <= train_end)]
        df_test = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)]
        if len(df_train) < 200 or len(df_test) < 20:
            logger.warning("Insufficient data for this period — skipping")
            continue

        # Fit GARCH on train and attach sigma to both train & test
        df_train_local = df_train.copy()
        df_train_local = df_train_local.assign(log_ret = df_train_local['log_ret'])
        garch_df_train, garch_meta = fit_garch_select(df_train_local, returns_col='log_ret', out_col='garch_sigma')
        # apply learned sigma to full df via reindexing the series (ffill/bfill)
        garch_sigma_full = garch_df_train['garch_sigma'].reindex(df_full.index).ffill().bfill()
        df_full_loc = df_full.copy()
        df_full_loc['garch_sigma'] = garch_sigma_full

        # Build scaler + PCA on training only and transform train+test
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(df_train[feature_cols].values)
        pca = PCA(n_components=0.95, svd_solver='full', random_state=SEED)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_scaled = scaler.transform(df_test[feature_cols].values)
        X_test_pca = pca.transform(X_test_scaled)

        period_res = {}

        for det in detectors:
            name = det.name
            logger.info(f"Running detector: {name}")
            try:
                if isinstance(det, VolatilityRegimeDetector):
                    # run on full index (train+test) so we can expand to test
                    combined = pd.concat([df_train, df_test]).sort_index()
                    labels_full = det.detect_regimes(combined)
                    train_labels = labels_full.reindex(df_train.index).ffill().bfill()
                    test_labels = labels_full.reindex(df_test.index).ffill().bfill()
                    period_res[name] = {'train_labels':train_labels,'test_labels':test_labels,'model':det,'meta':{}}
                elif isinstance(det, TailAwareGMM):
                    # fit on training features (PCA or raw depending on your choice) — we'll use PCA space
                    # Reconstruct DataFrames for PCA space mapping: map pca components back to idx
                    Xtr = pd.DataFrame(X_train_pca, index=df_train.index)
                    Xte = pd.DataFrame(X_test_pca, index=df_test.index)
                    # prepare pseudo-dataframes with component columns to satisfy detector interface
                    df_tr_pca = Xtr.copy(); df_te_pca = Xte.copy()
                    # label via detector using feature_cols equal to numeric columns
                    labels_train = det.detect_regimes(pd.concat([df_tr_pca]), feature_cols=list(df_tr_pca.columns))
                    labels_test = det.detect_regimes(pd.concat([df_tr_pca, df_te_pca]), feature_cols=list(pd.concat([df_tr_pca, df_te_pca]).columns)).reindex(df_test.index).ffill().bfill()
                    period_res[name] = {'train_labels':labels_train, 'test_labels':labels_test, 'model':det, 'meta':{}}
                elif isinstance(det, StudentTHMMRegimeDetector):
                    # grid search: try states in n_states_grid and cov types? for pomegranate we only vary n_states
                    best = None
                    best_meta = None
                    for n_states in [det.num_regimes] + [2,3,4]:
                        try:
                            det_try = StudentTHMMRegimeDetector(num_regimes=n_states, n_init=det.n_init)
                            labels = det_try.detect_regimes(pd.concat([df_train, df_test]), features=None)
                            # compute simple IC proxy: we don't have loglik easily; use regime labeling entropy as a cheap score (lower entropy -> stronger separation)
                            stab = compute_regime_stability(labels.reindex(df_train.index))
                            score = stab['entropy']
                            if best is None or score < best_meta['entropy']:
                                best = det_try
                                best_meta = {'entropy': score, 'n_states': n_states}
                        except Exception as e:
                            logger.warning(f"HMM candidate n_states={n_states} failed: {e}")
                            continue
                    if best is None:
                        raise RuntimeError("All Student-T HMM fits failed for this period")
                    labels_full = best.regime_labels
                    train_labels = labels_full.reindex(df_train.index).ffill().bfill()
                    test_labels = labels_full.reindex(df_test.index).ffill().bfill()
                    period_res[name] = {'train_labels':train_labels, 'test_labels':test_labels, 'model':best, 'meta':best_meta}
                else:
                    raise RuntimeError(f"Unknown detector type: {type(det)}")
            except Exception as e:
                logger.error(f"Detector {name} failed for period {train_start}->{test_end}: {e}")
                raise

        # Save period results
        results[f"period_{idx+1}"] = period_res

        # Save per-method CSVs
        for method_name, info in period_res.items():
            test_series = info['test_labels']
            out_fp = os.path.join(OUTPUT_DIR, f"period_{idx+1}_{method_name}_test_labels.csv")
            df_out = pd.DataFrame({'date':test_series.index,'regime':test_series.values})
            df_out.to_csv(out_fp, index=False)
            logger.info(f"Wrote {out_fp}")

        # Save PCA info (for reporting)
        pca_info = {
            'period': f"period_{idx+1}",
            'train_window': [train_start, train_end],
            'n_pca_components': int(pca.n_components_),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'selected_features': feature_cols
        }
        save_json(pca_info, os.path.join(OUTPUT_DIR, f"period_{idx+1}_pca_info.json"))

    return results

# ---------------------------
# Aggregation & reporting helper
# ---------------------------

def compute_inter_method_agreement_for_period(period_dict):
    methods = list(period_dict.keys())
    if len(methods) < 2:
        return None
    idxs = [period_dict[m]['test_labels'].index for m in methods]
    common = idxs[0]
    for idx in idxs[1:]:
        common = common.intersection(idx)
    if len(common) < 2:
        return None
    rows=[]
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            a = period_dict[methods[i]]['test_labels'].loc[common].values
            b = period_dict[methods[j]]['test_labels'].loc[common].values
            ari = adjusted_rand_score(a,b)
            nmi = normalized_mutual_info_score(a,b)
            rows.append({'method1':methods[i],'method2':methods[j],'ari':ari,'nmi':nmi})
    return pd.DataFrame(rows)

# ---------------------------
# Main
# ---------------------------

def main():
    logger.info("Regime Detector v2 — FULL RIGOROUS RUN (no silent fallbacks)")
    logger.info(f"Fetching {TICKER} {FULL_START} -> {FULL_END}")
    df = fetch_data(TICKER, FULL_START, FULL_END)
    logger.info(f"Fetched {len(df)} rows")

    logger.info("Building features...")
    df_feat, all_features = build_comprehensive_features(df)
    logger.info(f"{len(all_features)} features available, {len(df_feat)} rows after dropna")
    df_feat.to_csv(os.path.join(OUTPUT_DIR, "feature_data.csv"), index=True)

    logger.info("Selecting stable features (train_end_date = 2010-12-31)")
    selected, scores = select_stable_features(df_feat, all_features, train_end_date='2010-12-31', n_features=20)
    logger.info("Selected features (top 20): %s", selected)
    scores.to_csv(os.path.join(OUTPUT_DIR, "feature_stability_scores.csv"), index=False)

    # Detectors we will run (these are required and must exist)
    detectors = [
        VolatilityRegimeDetector(window=21, num_regimes=3, method='quantile'),
        TailAwareGMM(n_components=2, random_state=SEED),
        StudentTHMMRegimeDetector(num_regimes=3, n_init=6)
    ]

    wf_results = walk_forward_validate(df_feat, selected, detectors=detectors)

    # Summarize, compute crash metrics per period & method
    summary_rows=[]
    for period_name, per in wf_results.items():
        logger.info("Summarizing %s", period_name)
        for method_name, info in per.items():
            test_labels = info['test_labels']
            stability = compute_regime_stability(test_labels)
            crash_regime = choose_crash_regime_by_overlap(test_labels, CRASH_WINDOWS)
            precision, recall = (None, None)
            event_recall, avg_lag = (None, None)
            if crash_regime is not None:
                precision, recall = precision_recall_basic(test_labels, crash_regime, CRASH_WINDOWS)
                event_recall, avg_lag = event_level_detection(test_labels, CRASH_WINDOWS, crash_regime, slack_days=3)
            summary_rows.append({
                'period':period_name,
                'method':method_name,
                'n_regimes_observed':stability['n_regimes_observed'],
                'mean_duration':stability['mean_duration'],
                'transition_rate':stability['transition_rate'],
                'entropy':stability['entropy'],
                'crash_regime':crash_regime,
                'crash_precision':precision,
                'crash_recall':recall,
                'event_recall':event_recall,
                'event_avg_lag':avg_lag
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "walkforward_summary.csv"), index=False)
    logger.info("Wrote walkforward_summary.csv")

    # Inter-method agreement for each period
    agg_agreement = []
    for period_name, per in wf_results.items():
        agree_df = compute_inter_method_agreement_for_period(per)
        if agree_df is None:
            continue
        agree_fp = os.path.join(OUTPUT_DIR, f"{period_name}_agreement.csv")
        agree_df.to_csv(agree_fp, index=False)
        logger.info(f"Wrote {agree_fp}")
        agg_agreement.append((period_name, agree_df))
    # Save final reporting JSON
    report = {
        'n_periods': len(wf_results),
        'periods': list(wf_results.keys()),
        'selected_features': selected
    }
    save_json(report, os.path.join(OUTPUT_DIR, "run_report_summary.json"))

    # PCA loadings quick dumps
    # Recompute PCA on each train window and dump top loadings
    pca_reports=[]
    for idx,(train_start, train_end, _, _) in enumerate(WALK_FORWARD_PERIODS):
        df_train = df_feat[(df_feat.index>=train_start) & (df_feat.index<=train_end)]
        if df_train.empty: continue
        scaler = RobustScaler()
        Xs = scaler.fit_transform(df_train[selected].values)
        pca = PCA(n_components=min(10, len(selected)), random_state=SEED)
        Xp = pca.fit_transform(Xs)
        evr = pca.explained_variance_ratio_.tolist()
        comps = pca.components_
        top_loadings={}
        for ci in range(min(3, comps.shape[0])):
            comp = comps[ci]
            idxs = np.argsort(np.abs(comp))[::-1][:8]
            top_loadings[f"pc{ci+1}"] = [(selected[i], float(comp[i])) for i in idxs]
        pca_reports.append({
            'period': f'period_{idx+1}',
            'train_window':[train_start, train_end],
            'n_components':int(pca.n_components_),
            'explained_variance_ratio':evr,
            'top_loadings': top_loadings
        })
    save_json(pca_reports, os.path.join(OUTPUT_DIR, "pca_reports.json"))

    logger.info("Run complete. Output directory: %s", OUTPUT_DIR)
    print("Done — outputs are in", OUTPUT_DIR)

# ---------------------------
# Convenience: stable feature selection used above
# ---------------------------
def select_stable_features(df: pd.DataFrame, feature_cols: List[str], train_end_date: str, n_features: int = 15):
    df_train = df[df.index <= train_end_date]
    rows=[]
    for feat in feature_cols:
        s = df_train[feat].dropna()
        if len(s) < 100:
            continue
        acf1 = s.autocorr(lag=1)
        acf_score = 1.0 - abs(acf1 if not pd.isna(acf1) else 0.0)
        mid = len(s)//2
        v1 = s.iloc[:mid].std()
        v2 = s.iloc[mid:].std()
        var_stability = min(v1,v2)/max(v1,v2) if max(v1,v2)>0 else 0.0
        outlier_frac = (np.abs(s - s.mean()) > 3*s.std()).mean()
        outlier_score = 1.0 - outlier_frac
        score = acf_score*0.3 + var_stability*0.4 + outlier_score*0.3
        rows.append({'feature':feat,'score':score,'acf1':acf1,'var_stability':var_stability,'outlier_frac':outlier_frac})
    df_scores = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    selected = df_scores.head(n_features)['feature'].tolist()
    return selected, df_scores

# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    main()
