# regime_detection_full_fixed.py
# Robust full regime detection pipeline.
# Fixes ATR / Series vs DataFrame broadcasting issue and flattens multi-index columns.

import os
import math
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import datetime as dt

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# data
try:
    import yfinance as yf
except Exception:
    yf = None

# models
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None

# wkmeans optional
try:
    from wkmeans import WKMeans
    WKMEANS_AVAILABLE = True
except Exception:
    WKMEANS_AVAILABLE = False

# --- user settings ---
TICKER = "SPY"
START_DATE = "1997-01-01"
END_DATE = "2024-12-31"
OUT_DIR = "regime_out"
os.makedirs(OUT_DIR, exist_ok=True)

# Known crash windows for evaluation (adjust as you like)
CRASH_WINDOWS = [
    (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-03-31")),  # global financial crisis
    (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30")),  # COVID crash
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),  # 2022 volatile period
]

# ----------------------
# Utilities & robustness
# ----------------------
def _flatten_columns_if_multiindex(df):
    """If df has MultiIndex columns (yfinance sometimes does), flatten to single level."""
    # CORRECTED: Instead of joining, just take the first level of the column index.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

# utility: download data (OHLCV)
def fetch_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    if yf is None:
        raise RuntimeError("yfinance not available. Install with `pip install yfinance`.")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("No data returned from Yahoo. Check ticker/date/connectivity.")
    
    # As requested: print the raw columns right after download
    print("Initial raw columns from yfinance:", df.columns.tolist())

    # Flatten MultiIndex columns if present (e.g., ('Open', 'SPY') -> 'Open')
    df = _flatten_columns_if_multiindex(df)

    # Standardize column names to TitleCase to handle yfinance inconsistencies (e.g., 'close' vs 'Close')
    df.columns = [col.title() for col in df.columns]
    
    # Ensure 'Adj Close' present; best-effort fallback to 'Close'
    if 'Adj Close' not in df.columns and 'Adjclose' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            raise RuntimeError(f"Neither 'Adj Close' nor 'Close' present in data. Available columns: {df.columns.tolist()}")

    # Normalize column names for the rest of the script
    rename_map = {}
    if 'Adj Close' in df.columns:
        rename_map['Adj Close'] = 'AdjClose'
    if 'Close' in df.columns:
        rename_map['Close'] = 'Close'
    df = df.rename(columns=rename_map)

    required = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column {col} in downloaded data. Available columns: {df.columns.tolist()}")

    df = df[required]
    df.index = pd.to_datetime(df.index)
    return df

# features: build an extensive multiscale feature set (robust)
def build_features(df, debug=False):
    df = df.copy()
    # Basic sanity
    if 'AdjClose' not in df.columns:
        if 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'AdjClose'})
        else:
            raise RuntimeError("AdjClose not found in dataframe columns.")
    # log returns
    df['log_ret'] = np.log(df['AdjClose'] / df['AdjClose'].shift(1))
    df['ret'] = df['AdjClose'].pct_change()
    df['abs_ret'] = df['log_ret'].abs()
    # moving averages of returns (multi-scale)
    for ma in [3,5,7,10,21,63]:
        df[f'ma_r_{ma}'] = df['log_ret'].rolling(window=ma, min_periods=1).mean()
    # rolling vol (annualized)
    for w in [5,10,21,63,126]:
        df[f'vol_{w}'] = df['log_ret'].rolling(window=w, min_periods=1).std() * np.sqrt(252)
    # momentum and skewness/kurtosis in rolling windows
    for w in [21,63]:
        df[f'mom_{w}'] = df['log_ret'].rolling(window=w, min_periods=1).sum()
        df[f'skew_{w}'] = df['log_ret'].rolling(window=w, min_periods=1).skew()
        df[f'kurt_{w}'] = df['log_ret'].rolling(window=w, min_periods=1).kurt()
    # ATR (average true range) normalized by price - robust method
    prev_close = df['AdjClose'].shift(1)
    # Build three candidate tr series explicitly and align index
    tr1 = (df['High'] - df['Low']).abs()
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    # Concatenate into a DataFrame with same index, take row-wise max -> Series
    tr_df = pd.concat([tr1, tr2, tr3], axis=1)
    # Ensure we take the max across columns to get a Series
    tr = tr_df.max(axis=1)
    # Force into Series indexed like df (prevents weird shapes)
    tr = pd.Series(tr.values, index=df.index)
    atr = tr.rolling(window=14, min_periods=1).mean()
    # Now explicitly divide series-by-series (index-aligned), producing a Series
    df['atr'] = (atr / df['AdjClose']).astype(float)
    # volume features
    df['vol_ratio_21'] = df['Volume'] / df['Volume'].rolling(21, min_periods=1).mean()
    df['log_vol'] = np.log(df['Volume'] + 1)
    # keep core columns (feature list)
    feats = [col for col in df.columns if col not in ['Open','High','Low','Close','AdjClose','Volume']]
    # drop rows with NaN after operations (this keeps index alignment)
    df = df.dropna(subset=feats)
    if debug:
        print("DEBUG build_features:")
        print(" df.shape:", df.shape)
        print(" feats sample:", feats[:12])
        print(" atr.dtype:", df['atr'].dtype, "atr.shape:", df['atr'].shape)
    return df, feats

# helper: assemble feature matrix for selected feature names and scaler
def assemble_X(df, feature_list, scaler=None):
    if len(feature_list) == 0:
        raise ValueError("feature_list empty")
    X = df.loc[:, feature_list].values
    if scaler is None:
        scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

# HMM model selection (BIC/AIC), multiple restarts
def select_hmm(X, n_states_range=[2,3,4], cov_types=['full','diag','tied'], n_restarts=5, n_iter=300):
    if GaussianHMM is None:
        print("hmmlearn not available. Skipping HMM selection.")
        return None
    n_obs, n_feat = X.shape
    results = []
    best_model = None
    best_bic = np.inf
    for n_states in n_states_range:
        for cov in cov_types:
            best_ll = -np.inf
            best_local_model = None
            for seed in range(n_restarts):
                try:
                    m = GaussianHMM(n_components=n_states, covariance_type=cov, n_iter=n_iter, random_state=seed)
                    m.fit(X)
                    ll = m.score(X)
                    if ll > best_ll:
                        best_ll = ll
                        best_local_model = m
                except Exception:
                    continue
            if best_local_model is not None:
                # estimate number params for AIC/BIC (approx)
                if cov == 'full':
                    covs = n_states * n_feat * (n_feat + 1) / 2.0
                elif cov == 'diag':
                    covs = n_states * n_feat
                elif cov == 'tied':
                    covs = n_feat * (n_feat + 1) / 2.0
                elif cov == 'spherical':
                    covs = n_states
                else:
                    covs = n_states * n_feat
                means = n_states * n_feat
                trans = n_states * (n_states - 1)
                init = n_states - 1
                k = int(round(means + covs + trans + init))
                aic = 2*k - 2*best_ll
                bic = k * math.log(max(1, n_obs)) - 2*best_ll
                results.append({"n_states":n_states, "cov":cov, "ll":best_ll, "k":k, "aic":aic, "bic":bic, "model":best_local_model})
                if bic < best_bic:
                    best_bic = bic
                    best_model = best_local_model
    if not results:
        return None
    df_res = pd.DataFrame(results).sort_values('bic').reset_index(drop=True)
    return df_res, best_model

# feed-forward training and online prediction with periodic retrain
def feed_forward_predict(model_constructor, model_params, X_df, train_cut_idx, retrain_step=20, scaler=None, verbose=False):
    n = len(X_df)
    if train_cut_idx < 1 or train_cut_idx >= n:
        raise ValueError("train_cut_idx must be 1..n-1")
    preds = []
    indices = X_df.index[train_cut_idx:]
    X_all = X_df.values
    model = model_constructor(X_all[:train_cut_idx], model_params)
    if verbose:
        print("Initial model trained on", train_cut_idx, "observations")
    for i in range(train_cut_idx, n):
        try:
            s = model.predict(X_all[:i+1])
            preds.append(int(s[-1]))
        except Exception:
            try:
                s = model.predict(X_all[i:i+1])
                preds.append(int(s[0]))
            except Exception:
                preds.append(np.nan)
        if ((i - train_cut_idx + 1) % retrain_step == 0) and (i+1 < n):
            if verbose:
                print("Retrain at absolute index", i+1)
            try:
                model = model_constructor(X_all[:i+1], model_params)
            except Exception as e:
                if verbose:
                    print("Retrain failed:", e)
    return pd.Series(preds, index=indices, name="states")

# model constructors
def fit_hmm(X_train, params):
    if GaussianHMM is None:
        raise RuntimeError("hmmlearn missing")
    model = GaussianHMM(**params)
    model.fit(X_train)
    return model

def fit_gmm_wrapper(X_train, params):
    gmm = GaussianMixture(**params)
    gmm.fit(X_train)
    class Wrapper:
        def __init__(self, gmm):
            self.gmm = gmm
        def predict(self, X):
            return self.gmm.predict(X)
    return Wrapper(gmm)

# diagnostics
def regime_transition_matrix(states_series):
    arr = np.array(states_series)
    mask = ~pd.isna(arr)
    arr = arr[mask]
    if arr.size == 0:
        return np.array([]), np.zeros((0,0), dtype=int), np.zeros((0,0), dtype=float)
    unique = np.unique(arr)
    n = len(unique)
    idx = {u:i for i,u in enumerate(unique)}
    mat = np.zeros((n,n), dtype=int)
    for a,b in zip(arr[:-1], arr[1:]):
        mat[idx[a], idx[b]] += 1
    prob = mat.astype(float)
    row_sums = prob.sum(axis=1, keepdims=True)
    prob = np.divide(prob, row_sums, out=np.zeros_like(prob), where=row_sums!=0)
    return unique, mat, prob

def regime_durations(states_series):
    s = states_series.dropna()
    if s.empty:
        return pd.DataFrame(columns=['regime','duration'])
    durations = []
    labels = []
    current = s.iloc[0]
    start = s.index[0]
    for idx, val in s.iloc[1:].items():
        if val == current:
            continue
        else:
            dur = (idx - start).days
            durations.append(dur)
            labels.append(current)
            current = val
            start = idx
    end = s.index[-1]
    durations.append((end - start).days)
    labels.append(current)
    return pd.DataFrame({'regime':labels,'duration':durations})

def precision_recall_on_crash(states_series, crash_windows, crash_regime):
    s = states_series.dropna()
    if s.empty:
        return 0.0, 0.0
    is_crash_label = s == crash_regime
    positive_preds = s.index[is_crash_label]
    crash_days_idx = set()
    for a,b in crash_windows:
        rng = pd.date_range(a,b, freq='D')
        crash_days_idx.update(rng)
    total_crash_days = sum(1 for d in s.index if d in crash_days_idx)
    in_crash_and_pred = [d for d in positive_preds if d in crash_days_idx]
    recall = len(in_crash_and_pred) / max(1, total_crash_days)
    precision = len(in_crash_and_pred) / max(1, len(positive_preds))
    return precision, recall

def strategy_pnl(states_series, price_series, crash_regime):
    df = pd.DataFrame({'state':states_series, 'price':price_series})
    df = df.dropna()
    if df.empty:
        return pd.DataFrame()
    df['log_ret'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    if df.empty:
        return pd.DataFrame()
    df['position'] = np.where(df['state'] == crash_regime, -1, 1)
    df['pnl'] = df['position'] * df['log_ret']
    df['cum_pnl'] = df['pnl'].cumsum()
    return df

# -----------------------
# MAIN pipeline
# -----------------------
def main():
    print("FETCHING DATA...")
    df = fetch_data(TICKER, START_DATE, END_DATE)
    print(f"Fetched {len(df)} rows for {TICKER}")
    print("Building features (robust)...")
    df_feat, feat_names = build_features(df, debug=False)
    print("Features built:", feat_names[:8], "... total:", len(feat_names))

    # Candidate feature subsets
    feature_sets = {
        "raw_small": ['log_ret','ret','abs_ret'],
        "multi_scale": ['log_ret','ma_r_7','ma_r_21','vol_21','vol_63','abs_ret','atr','vol_ratio_21'],
        "full": feat_names
    }

    scaler_map = {}
    models_selected = {}

    # iterate feature sets and run HMM selection
    for name, fset in feature_sets.items():
        print("\n--- FEATURE SET:", name, "size=", len(fset))
        missing = [f for f in fset if f not in df_feat.columns]
        if missing:
            print("Skipping feature set due to missing features:", missing)
            continue
        Xs, scaler = assemble_X(df_feat, fset, scaler=StandardScaler())
        scaler_map[name] = scaler
        print("Running HMM selection (this can take a bit)...")
        hh = select_hmm(Xs, n_states_range=[2,3,4], cov_types=['full','diag','tied'], n_restarts=4, n_iter=300)
        if hh is None:
            print("No HMM available for set", name)
            continue
        df_res, best_model = hh
        print("Top HMM models by BIC for feature set", name)
        print(df_res[['n_states','cov','ll','k','bic']].head(5).to_string(index=False))
        models_selected[name] = {"df_res":df_res, "best_model":best_model, "scaler":scaler, "fset":fset}

    # pick best overall by BIC across feature sets
    best_overall = None
    for name, entry in models_selected.items():
        df_res = entry['df_res']
        top = df_res.iloc[0]
        if best_overall is None or top['bic'] < best_overall['bic']:
            best_overall = {"feature_set_name": name, **top.to_dict(), "model": entry['best_model'], "scaler": entry['scaler'], "fset": entry['fset']}
    if best_overall is None:
        print("No HMM fitted successfully. Exiting.")
        return
    print("\nBEST OVERALL HMM:")
    print({k: best_overall[k] for k in ['feature_set_name','n_states','cov','bic'] if k in best_overall})

    # In-sample hidden states (best model, full X for that feature set)
    fset = best_overall['fset']
    scaler = best_overall['scaler']
    Xs_all = scaler.transform(df_feat.loc[:, fset].values)
    best_model = best_overall['model']
    try:
        hidden_states_insample = pd.Series(best_model.predict(Xs_all), index=df_feat.index, name="hmm_state_in_sample")
    except Exception as e:
        print("In-sample HMM predict failed:", e)
        hidden_states_insample = pd.Series(index=df_feat.index, data=np.nan, name="hmm_state_in_sample")
    hidden_states_insample.to_csv(os.path.join(OUT_DIR, "hmm_states_in_sample.csv"))

    # Feed-forward out-of-sample simulation:
    total_obs = len(df_feat)
    train_cut = int(total_obs * 0.30)
    if train_cut < 10:
        train_cut = max(10, int(total_obs * 0.1))
    retrain_step = 20
    print(f"\nRunning feed-forward OOS with initial train {train_cut} obs, retrain_step={retrain_step}")
    X_df = pd.DataFrame(scaler.transform(df_feat[fset].values), index=df_feat.index, columns=fset)

    hmm_params = {"n_components": int(best_overall['n_states']), "covariance_type": best_overall['cov'], "n_iter": 200}
    try:
        states_oos = feed_forward_predict(lambda X,p: fit_hmm(X, hmm_params), hmm_params, X_df, train_cut, retrain_step, scaler=scaler, verbose=True)
    except Exception as e:
        print("Feed-forward HMM failed:", e)
        states_oos = pd.Series(index=df_feat.index[train_cut:], data=np.nan, name="hmm_state_oos")

    states_oos.name = "hmm_state_oos"
    states_oos.to_csv(os.path.join(OUT_DIR, "hmm_states_oos.csv"))

    # Diagnostics: transition matrix and durations (OOS)
    print("\nDiagnostics (OOS):")
    uniq, mat, prob = regime_transition_matrix(states_oos)
    print("Regimes:", uniq)
    print("Transition counts:\n", mat)
    print("Transition probabilities:\n", prob)
    if prob.size:
        pd.DataFrame(prob, index=uniq, columns=uniq).to_csv(os.path.join(OUT_DIR,"hmm_transition_probs_oos.csv"))

    durations_df = regime_durations(states_oos)
    durations_df.to_csv(os.path.join(OUT_DIR,"hmm_durations_oos.csv"), index=False)
    if not durations_df.empty:
        print("Regime duration sample stats:\n", durations_df.groupby('regime').duration.describe())

    # Precision/recall on known crash windows
    insample_returns = df_feat['log_ret']
    avg_by_state = {}
    for s in np.unique(hidden_states_insample.dropna()):
        mask = hidden_states_insample == s
        avg_by_state[int(s)] = insample_returns[mask].mean() if mask.sum()>0 else np.nan
    if avg_by_state:
        crash_regime = min(avg_by_state, key=lambda k: avg_by_state[k] if not pd.isna(avg_by_state[k]) else 1e9)
    else:
        crash_regime = 0
    print("Selected crash_regime:", crash_regime, "avg return:", avg_by_state.get(crash_regime, np.nan))

    precision, recall = precision_recall_on_crash(states_oos, CRASH_WINDOWS, crash_regime)
    print(f"Crash detection (OOS) precision={precision:.3f}, recall={recall:.3f}")

    # Toy strategy P&L OOS
    price_for_oos = df_feat['AdjClose'].loc[states_oos.index]
    pnl_df = strategy_pnl(states_oos, price_for_oos, crash_regime)
    if not pnl_df.empty:
        pnl_df.to_csv(os.path.join(OUT_DIR,"hmm_strategy_oos.csv"))
        try:
            plt.figure(figsize=(10,6))
            plt.plot(pnl_df.index, pnl_df['cum_pnl'], label='HMM Regime PnL (long/short)')
            bah = np.log(price_for_oos / price_for_oos.shift(1)).cumsum()
            plt.plot(bah.index, bah.values, label='Buy and Hold (log cum)')
            plt.legend()
            plt.title("OOS Strategy vs Buy & Hold")
            plt.savefig(os.path.join(OUT_DIR,"hmm_strategy_vs_bah.png"), dpi=200)
            plt.close()
        except Exception as e:
            print("Plotting PnL failed:", e)

    # Benchmarks and visualizations omitted here for brevity (same logic as before)
    print("\nAll outputs saved in", OUT_DIR)
    print("Completed.")

if __name__ == "__main__":
    main()