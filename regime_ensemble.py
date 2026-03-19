# ====== regime_ensemble.py ======
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.utils import resample
import math
import warnings
# import your detectors
from regime_detector import HMMRegimeDetector, VolatilityRegimeDetector, ChangePointRegimeDetector, MultifractalRegimeDetector

def _entropy_from_probs(probs: np.ndarray, axis: int = 1, base: float = math.e) -> np.ndarray:
    """Shannon entropy for rows of probability array; handle zeros safely."""
    # probs shape (n_samples, n_states)
    p = np.clip(probs, 1e-12, 1.0)
    ent = -np.sum(p * np.log(p) / np.log(base), axis=axis)
    return ent

def run_all_detectors(
    market_data: pd.DataFrame,
    run_hmm: bool = True,
    run_vol: bool = True,
    run_cp: bool = True,
    run_mf: bool = True,
    hmm_bootstrap_iters: int = 100,
    hmm_block_size: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run HMM, Volatility, ChangePoint, Multifractal detectors and return a
    daily DataFrame aligned to market_data.index with columns:
      - regime_hmm (int), hmm_entropy (float), hmm_boot_agree (0-1 float)
      - regime_vol (int)
      - regime_cp (int)
      - regime_mf (int)
      - consensus_regime (object/text), consensus_strength (0-1)
    """
    idx = market_data.index
    results = pd.DataFrame(index=idx)

    # -------------------------
    # 1) VOLATILITY
    # -------------------------
    if run_vol:
        vol_detector = VolatilityRegimeDetector(window=20, num_regimes=3, method='quantile')
        try:
            vol_labels = vol_detector.detect_regimes(market_data)  # index subset (returns.dropna())
            vol_labels = vol_detector.expand_to_full_index(vol_labels, idx, method='ffill').astype(int)
            results['regime_vol'] = vol_labels
            if verbose: print(f"[vol] done, regimes: {sorted(vol_labels.unique())}")
        except Exception as e:
            warnings.warn(f"Volatility detector failed: {e}")
            results['regime_vol'] = np.nan

    # -------------------------
    # 2) HMM + entropy + bootstrap stability
    # -------------------------
    if run_hmm:
        hmm_detector = HMMRegimeDetector(num_regimes=3, covariance_type='full', n_iter=200, n_init=3)
        try:
            hmm_labels = hmm_detector.detect_regimes(market_data)  # indexed by features (returns.dropna())
            # align
            hmm_labels_full = hmm_detector.expand_to_full_index(hmm_labels, idx, method='ffill').astype(int)
            results['regime_hmm'] = hmm_labels_full

            # compute posterior probs and entropy if model supports it
            try:
                # try to get responsibilities for original features
                returns = market_data['close'].pct_change().dropna()
                features = pd.DataFrame({'returns': returns, 'abs_returns': returns.abs()})
                X = features.values
                # if scaler used internally, we don't have it here; best-effort: scale with same logic as detector
                # try to call detector.model.predict_proba
                if hasattr(hmm_detector.model, 'predict_proba'):
                    probs = hmm_detector.model.predict_proba(hmm_detector.model._compute_log_likelihood(X))
                    # hmmlearn older versions don't expose direct API; fallback below
                else:
                    # fallback: use model._compute_log_likelihood to get log-lik matrix then convert to probs row-wise
                    loglikes = hmm_detector.model._compute_log_likelihood(X)  # shape (n_samples, n_components)
                    # convert loglikes to normalized probs (softmax)
                    a = loglikes - loglikes.max(axis=1, keepdims=True)
                    expa = np.exp(a)
                    probs = expa / expa.sum(axis=1, keepdims=True)
                ent = _entropy_from_probs(probs, axis=1)
                ent_series = pd.Series(ent, index=features.index)
                ent_full = ent_series.reindex(idx).fillna(method='ffill').fillna(method='bfill')
                results['hmm_entropy'] = ent_full
            except Exception:
                # safe fallback: set NaN
                results['hmm_entropy'] = np.nan

            # Bootstrap stability: block bootstrap on returns, fit HMM on each resample,
            # count fraction of resamples where day's label == baseline label
            try:
                returns = market_data['close'].pct_change().dropna()
                n = len(returns)
                base_labels = hmm_labels.reindex(returns.index)  # baseline labels on returns.index
                # prepare storage
                agree_counts = pd.Series(0, index=returns.index, dtype=float)

                for it in range(hmm_bootstrap_iters):
                    # block bootstrap: sample blocks and reconstruct index-aligned resample
                    # simple approach: resample by selecting random start indices for blocks
                    block_size = max(1, int(hmm_block_size))
                    nblocks = int(np.ceil(n / block_size))
                    indices = []
                    for b in range(nblocks):
                        start = np.random.randint(0, n - block_size + 1)
                        indices.extend(list(range(start, start + block_size)))
                    indices = indices[:n]  # trim
                    resampled_returns = returns.values[indices]
                    # create artificial DataFrame with same index (we'll fit to values)
                    df_resample = pd.DataFrame({'close': np.nan}, index=returns.index)
                    # reconstruct cumulative price series from resampled returns (start at 100)
                    price = 100 * np.exp(np.cumsum(resampled_returns))
                    df_resample['close'] = price
                    try:
                        labels_boot = HMMRegimeDetector(num_regimes=hmm_detector.num_regimes,
                                                       covariance_type=hmm_detector.covariance_type,
                                                       n_iter=100, n_init=1).detect_regimes(df_resample)
                    except Exception:
                        continue
                    # align and compare
                    # labels_boot indexed by returns.index
                    agree = (labels_boot.values == base_labels.values).astype(int)
                    agree_counts += agree

                agree_frac = agree_counts / float(max(1, hmm_bootstrap_iters))
                agree_frac_full = agree_frac.reindex(idx).fillna(method='ffill').fillna(method='bfill')
                results['hmm_boot_agree'] = agree_frac_full
            except Exception:
                results['hmm_boot_agree'] = np.nan

            if verbose: print("[hmm] done, entropy and bootstrap stability added.")
        except Exception as e:
            warnings.warn(f"HMM detector failed: {e}")
            results['regime_hmm'] = np.nan
            results['hmm_entropy'] = np.nan
            results['hmm_boot_agree'] = np.nan

    # -------------------------
    # 3) CHANGE POINT
    # -------------------------
    if run_cp:
        try:
            cp_detector = ChangePointRegimeDetector(method='pelt', penalty=10, min_size=20)
            cp_labels = cp_detector.detect_regimes(market_data)  # indexed by returns.index
            cp_full = cp_detector.expand_to_full_index(cp_labels, idx, method='ffill').astype(int)
            results['regime_cp'] = cp_full
            # also export break dates
            bp_dates = cp_detector.get_breakpoint_dates()
            results['cp_break'] = False
            for d in bp_dates:
                if d in results.index:
                    results.at[d, 'cp_break'] = True
            if verbose: print(f"[cp] done, {len(bp_dates)} breakpoints found.")
        except Exception as e:
            warnings.warn(f"ChangePoint detector failed: {e}")
            results['regime_cp'] = np.nan
            results['cp_break'] = False

    # -------------------------
    # 4) MULTIFRACTAL
    # -------------------------
    if run_mf:
        try:
            mf_detector = MultifractalRegimeDetector(window=60, num_regimes=3)
            mf_labels = mf_detector.detect_regimes(market_data)  # features.index
            mf_full = mf_detector.expand_to_full_index(mf_labels, idx, method='ffill').astype(int)
            results['regime_mf'] = mf_full
            if verbose: print("[mf] done.")
        except Exception as e:
            warnings.warn(f"Multifractal detector failed: {e}")
            results['regime_mf'] = np.nan

    # -------------------------
    # 5) Consensus / ensemble labeling
    # -------------------------
    # Build a simple voting consensus: collect detectors that produced labels
    det_cols = [c for c in results.columns if c.startswith('regime_')]
    def _consensus_row(row):
        votes = []
        weights = []
        for col in det_cols:
            v = row.get(col, np.nan)
            if pd.isna(v):
                continue
            # compute weight: use stability if available (hmm_boot_agree used), else 1
            w = 1.0
            if col == 'regime_hmm' and 'hmm_boot_agree' in row.index:
                # if NaN, fallback to 1
                w = float(row.get('hmm_boot_agree', 0.5) or 0.5)
            votes.append(int(v))
            weights.append(float(w))
        if len(votes) == 0:
            return (np.nan, 0.0)
        # weighted majority: compute weighted counts per label
        labels = np.array(votes)
        u_labels = np.unique(labels)
        score_map = {}
        for ul in u_labels:
            score_map[int(ul)] = float(np.sum([w for lab,w in zip(votes,weights) if lab==ul]))
        # pick top
        best_label = max(score_map.items(), key=lambda x: x[1])[0]
        strength = score_map[best_label] / float(np.sum(weights))
        return (best_label, strength)

    cons = results.apply(_consensus_row, axis=1)
    cons_df = pd.DataFrame(list(cons), index=results.index, columns=['consensus_regime','consensus_strength'])
    results = pd.concat([results, cons_df], axis=1)

    # add metadata columns: how many detectors voted
    results['num_detectors'] = results[det_cols].notna().sum(axis=1)

    # Save to file
    results.to_parquet('regimes_all.parquet', index=True, compression='gzip')
    if verbose: print("Saved regimes_all.parquet with columns:", list(results.columns))

    return results
