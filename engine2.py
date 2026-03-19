#!/usr/bin/env python3
"""
engine_force_regime6_adapt.py

Minimal engine that *adapts* to regime6 exports (without modifying regime6).
- Deterministic seed
- SPY 1997-01-01 -> 2024-01-01
- Backtests simple strategy set
- Computes per-regime & global PNL / ALPHA / SHARPE leaderboards
- Writes CSV reports to ./reports_regime6_adapt/

Behavior:
 - Imports regime6 and tries several common export names to recover:
     1) detect_regimes_api(...)
     2) run_all_detectors(...) or run_detectors(...)
     3) module-level variables: PER_METHOD_REGIMES, ENSEMBLE_REGIMES, REGIMES_BY_METHOD
 - Fails with clear error if no regimes found.
"""
import os
import sys
import math
import logging
import random
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# local modules you must have in same project
from data_manager import DataManager
from strategy_zoo import StrategyRegistry, BuyAndHold, SMACrossover, BollingerMeanReversion, RSIMomentum
from backtest_engine import BacktestEngine, BacktestResult

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Deterministic seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Config
TICKER = "SPY"
START_DATE = "1997-01-01"
END_DATE = "2024-01-01"
INITIAL_CAPITAL = 100_000
OUT_DIR = "reports_regime6_adapt"
os.makedirs(OUT_DIR, exist_ok=True)

# Try to import regime6 (must exist)
try:
    import regime6
except Exception as e:
    raise RuntimeError("regime6.py must be importable and present in the working directory.") from e


# --------------------
# Helpers to extract regimes from regime6
# --------------------
def try_detect_api_via_function(market_index: pd.DatetimeIndex) -> Tuple[Dict[str, pd.Series], pd.Series, Optional[pd.Series]]:
    """
    Attempt to call functions on regime6 to obtain per-method regimes and an ensemble.
    Tries a few likely function names and argument patterns.
    Returns: (per_method_dict, ensemble_series, confidence_series_or_none)
    Raises RuntimeError if unsuccessful.
    """
    candidates = [
        ('detect_regimes_api', ('ticker', 'start', 'end', 'n_states', 'methods', 'seed', 'save_outputs', 'output_dir')),
        ('detect_regimes', ('data', 'n_states', 'methods', 'seed')),
        ('run_all_detectors', ('data', 'methods', 'seed')),
        ('run_detectors', ('data', 'methods', 'seed'))
    ]

    for name, _args in candidates:
        fn = getattr(regime6, name, None)
        if callable(fn):
            LOG.info("Found regime6 function: %s — attempting call", name)
            try:
                # attempt common signatures:
                # 1) detect_regimes_api(ticker=..., start=..., end=..., n_states=..., methods=..., seed=..., save_outputs=..., output_dir=...)
                # 2) detect_regimes(data=..., n_states=..., methods=..., seed=...)
                # 3) run_all_detectors(data=..., methods=..., seed=...)
                kwargs = {}
                if 'ticker' in fn.__code__.co_varnames:
                    kwargs.update({'ticker': TICKER, 'start': START_DATE, 'end': END_DATE, 'n_states': 3, 'methods': ['hmm','gmm','kmeans'], 'seed': SEED, 'save_outputs': False, 'output_dir': None})
                    res = fn(**{k: v for k, v in kwargs.items() if k in fn.__code__.co_varnames})
                elif 'data' in fn.__code__.co_varnames:
                    # prepare a small market frame to pass
                    dm = DataManager(cache_dir='./data_cache')
                    df = dm.fetch_data(symbol=TICKER, start_date=START_DATE, end_date=END_DATE)
                    kwargs = {'data': df, 'methods': ['hmm','gmm','kmeans'], 'seed': SEED, 'n_states': 3}
                    res = fn(**{k: v for k, v in kwargs.items() if k in fn.__code__.co_varnames})
                else:
                    # fallback: try no args
                    res = fn()
                # interpret res
                res_dict = {}
                ensemble = None
                confidence = None
                if isinstance(res, dict):
                    # expected keys: 'per_method', 'ensemble', 'confidence', 'ic_scores' etc.
                    per_method = res.get('per_method') or res.get('per_method_series') or {}
                    if isinstance(per_method, dict):
                        # ensure series align to market_index
                        per_method_aligned = {}
                        for k, s in per_method.items():
                            try:
                                s2 = s.reindex(market_index).ffill().astype(int)
                            except Exception:
                                s2 = pd.Series(s, index=market_index)[:len(market_index)].astype(int)
                            per_method_aligned[k] = s2
                        res_dict = per_method_aligned
                    # ensemble
                    ensemble = res.get('ensemble') or res.get('ensemble_regimes') or res.get('ensemble_series')
                    confidence = res.get('confidence') or res.get('agreement')
                elif isinstance(res, tuple) and len(res) >= 2:
                    # maybe (per_method, ensemble, confidence)
                    per_method, ensemble = res[0], res[1]
                    if len(res) >= 3:
                        confidence = res[2]
                else:
                    LOG.debug("regime6.%s returned unexpected type %s", name, type(res))
                    continue

                # normalize ensemble & confidence to pd.Series aligned to market_index
                if isinstance(ensemble, pd.Series):
                    ensemble = ensemble.reindex(market_index).ffill().astype(int)
                elif isinstance(ensemble, (dict, list, np.ndarray)):
                    try:
                        ensemble = pd.Series(ensemble, index=market_index).ffill().astype(int)
                    except Exception:
                        ensemble = None

                if isinstance(confidence, pd.Series):
                    confidence = confidence.reindex(market_index).ffill()
                elif isinstance(confidence, (list, np.ndarray)):
                    try:
                        confidence = pd.Series(confidence, index=market_index).reindex(market_index).ffill()
                    except Exception:
                        confidence = None

                return res_dict, ensemble, confidence
            except Exception as e:
                LOG.warning("Call to regime6.%s failed: %s", name, e)
                continue
    # nothing worked
    raise RuntimeError("Could not call any regime6 detector functions successfully.")


def try_extract_module_vars(market_index: pd.DatetimeIndex) -> Tuple[Dict[str, pd.Series], pd.Series, Optional[pd.Series]]:
    """
    Try to find module-level variables exported by regime6 with likely names.
    """
    per_method_names = ['PER_METHOD_REGIMES', 'per_method', 'REGIMES_BY_METHOD', 'regimes_by_method']
    ensemble_names = ['ENSEMBLE_REGIMES', 'ensemble', 'ENSEMBLE', 'ENSEMBLE_SERIES', 'regimes_ensemble']
    confidence_names = ['ENSEMBLE_CONFIDENCE', 'CONFIDENCE', 'ensemble_confidence', 'agreement']

    per_method = {}
    for name in per_method_names:
        if hasattr(regime6, name):
            obj = getattr(regime6, name)
            if isinstance(obj, dict):
                for k, s in obj.items():
                    try:
                        s2 = s.reindex(market_index).ffill().astype(int)
                    except Exception:
                        s2 = pd.Series(s, index=market_index)[:len(market_index)].astype(int)
                    per_method[k] = s2
                LOG.info("Found per-method regimes in regime6.%s", name)
                break

    ensemble = None
    for name in ensemble_names:
        if hasattr(regime6, name):
            obj = getattr(regime6, name)
            if isinstance(obj, pd.Series):
                ensemble = obj.reindex(market_index).ffill().astype(int)
            else:
                try:
                    ensemble = pd.Series(obj, index=market_index).reindex(market_index).ffill().astype(int)
                except Exception:
                    ensemble = None
            LOG.info("Found ensemble regimes in regime6.%s", name)
            break

    confidence = None
    for name in confidence_names:
        if hasattr(regime6, name):
            obj = getattr(regime6, name)
            try:
                confidence = pd.Series(obj, index=market_index).reindex(market_index).ffill()
            except Exception:
                confidence = None
            LOG.info("Found ensemble confidence in regime6.%s", name)
            break

    if not per_method and ensemble is None:
        raise RuntimeError("No per-method or ensemble regime data found as module variables in regime6.")
    return per_method, ensemble, confidence


def obtain_regimes_strict(market_index: pd.DatetimeIndex) -> Tuple[Dict[str, pd.Series], pd.Series, Optional[pd.Series]]:
    """
    Try multiple strategies to obtain regime outputs from regime6.
    This does not modify regime6.
    """
    # 1) Try calling a function API
    try:
        pm, ensemble, conf = try_detect_api_via_function(market_index)
        if ensemble is not None:
            LOG.info("Obtained regimes via function call.")
            return pm, ensemble, conf
    except Exception as e:
        LOG.info("Function API attempt failed: %s", e)

    # 2) Try module vars
    try:
        pm, ensemble, conf = try_extract_module_vars(market_index)
        LOG.info("Obtained regimes via module variables.")
        return pm, ensemble, conf
    except Exception as e:
        LOG.info("Module variable attempt failed: %s", e)

    # Nothing worked -> raise (we must use regime6)
    raise RuntimeError("Failed to obtain regimes from regime6 via any supported path. Inspect regime6 exports.")


# --------------------
# Backtesting wrapper
# --------------------
def run_backtests(registry: StrategyRegistry, market_data: pd.DataFrame, benchmark_data: pd.DataFrame) -> Dict[str, BacktestResult]:
    engine = BacktestEngine(benchmark_symbol=TICKER)
    results = {}
    for s in registry.strategies.values():
        LOG.info("Backtesting %s...", s.name)
        res = engine.run_backtest(s, market_data, initial_capital=INITIAL_CAPITAL, benchmark_data=benchmark_data)
        # align series
        try:
            res.returns = res.returns.reindex(market_data.index).fillna(0)
        except Exception:
            LOG.warning("Could not align returns for %s", s.name)
        if getattr(res, 'alpha_returns', None) is not None:
            try:
                res.alpha_returns = res.alpha_returns.reindex(market_data.index).fillna(0)
            except Exception:
                LOG.warning("Could not align alpha_returns for %s", s.name)
        results[s.name] = res
    return results


# --------------------
# Compute simple regression alpha & beta (fallback)
# --------------------
def compute_regression_alpha_beta(strategy_ret: pd.Series, benchmark_ret: pd.Series) -> Tuple[float, float]:
    """
    Compute OLS slope (beta) and intercept (alpha_daily) using valid overlapping dates.
    Returns (alpha_daily, beta). alpha_daily is in daily return units.
    """
    s = strategy_ret.reindex(benchmark_ret.index).dropna()
    b = benchmark_ret.reindex(s.index).dropna()
    if s.empty or b.empty:
        return float('nan'), float('nan')
    # demean? no — simple OLS: s = alpha + beta * b + eps
    X = np.vstack([np.ones(len(b)), b.values]).T
    y = s.values
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha = float(coef[0])
        beta = float(coef[1])
        return alpha, beta
    except Exception:
        return float('nan'), float('nan')


# --------------------
# Build master table aligned to market index
# --------------------
def build_master_table(all_results: Dict[str, BacktestResult], market_index: pd.DatetimeIndex, ensemble: pd.Series, ensemble_confidence: Optional[pd.Series]):
    rows = []
    # compute benchmark returns if present in any result or assume we can fetch from DataManager
    # prefer benchmark inside first result; else fetch via DataManager
    benchmark_ret = None
    for res in all_results.values():
        if getattr(res, 'benchmark_returns', None) is not None:
            benchmark_ret = res.benchmark_returns.reindex(market_index).fillna(0)
            break
    if benchmark_ret is None:
        dm = DataManager(cache_dir="./data_cache")
        mkt = dm.fetch_data(symbol=TICKER, start_date=START_DATE, end_date=END_DATE)
        benchmark_ret = mkt['Adj Close'].pct_change().reindex(market_index).fillna(0)

    for name, res in all_results.items():
        # prefer returns/alpha if present, else attempt to compute alpha via regression later
        r = getattr(res, 'returns', None)
        a = getattr(res, 'alpha_returns', None)
        if r is None:
            LOG.warning("Result for %s missing returns. Skipping.", name)
            continue
        r = r.reindex(market_index).fillna(0).astype(float)
        if a is not None:
            a = a.reindex(market_index).fillna(0).astype(float)
        else:
            a = None
        # compute daily beta/alpha fallback if needed
        if a is None:
            alpha_daily, beta = compute_regression_alpha_beta(r, benchmark_ret)
            # create alpha series by subtracting beta*benchmark
            a = (r - beta * benchmark_ret).fillna(0).astype(float)
        for dt in market_index:
            regime_val = int(ensemble.loc[dt]) if (ensemble is not None and dt in ensemble.index and not pd.isna(ensemble.loc[dt])) else None
            stress_flag = False
            if ensemble_confidence is not None and dt in ensemble_confidence.index:
                stress_flag = bool(ensemble_confidence.loc[dt] <= 0.5)
            rows.append({
                'date': pd.Timestamp(dt),
                'strategy': name,
                'return': float(r.loc[dt]) if dt in r.index else float('nan'),
                'alpha': float(a.loc[dt]) if dt in a.index else float('nan'),
                'regime': regime_val,
                'stress': stress_flag
            })
    master = pd.DataFrame(rows)
    master = master.sort_values(['strategy', 'date']).reset_index(drop=True)
    return master


# --------------------
# Metrics: per regime and global
# --------------------
def compute_performance_tables(master: pd.DataFrame, periods_per_year: int = 252):
    def ann_from_mean(mean_daily):
        return (1 + mean_daily) ** periods_per_year - 1

    per_regime_rows = []
    for (strategy, regime), grp in master.groupby(['strategy','regime']):
        valid_ret = grp['return'].dropna()
        valid_alpha = grp['alpha'].dropna()
        n = len(valid_ret)
        mean_daily = float(valid_ret.mean()) if n else float('nan')
        mean_alpha = float(valid_alpha.mean()) if len(valid_alpha) else float('nan')
        vol = float(valid_ret.std() * math.sqrt(periods_per_year)) if n else float('nan')
        annual = ann_from_mean(mean_daily) if n else float('nan')
        sharpe = float(annual / vol) if (n and vol and vol != 0) else float('nan')
        per_regime_rows.append({
            'strategy': strategy,
            'regime': regime,
            'n_days': int(n),
            'mean_daily_return': mean_daily,
            'mean_daily_alpha': mean_alpha,
            'annual_return': annual,
            'annual_vol': vol,
            'sharpe': sharpe
        })
    per_regime_df = pd.DataFrame(per_regime_rows)

    # global
    global_rows = []
    for strategy, grp in master.groupby('strategy'):
        valid_ret = grp['return'].dropna()
        valid_alpha = grp['alpha'].dropna()
        n = len(valid_ret)
        mean_daily = float(valid_ret.mean()) if n else float('nan')
        mean_alpha = float(valid_alpha.mean()) if len(valid_alpha) else float('nan')
        vol = float(valid_ret.std() * math.sqrt(periods_per_year)) if n else float('nan')
        annual = ann_from_mean(mean_daily) if n else float('nan')
        sharpe = float(annual / vol) if (n and vol and vol != 0) else float('nan')
        global_rows.append({
            'strategy': strategy,
            'n_days': int(n),
            'mean_daily_return': mean_daily,
            'mean_daily_alpha': mean_alpha,
            'annual_return': annual,
            'annual_vol': vol,
            'sharpe': sharpe
        })
    global_df = pd.DataFrame(global_rows).sort_values('sharpe', ascending=False)
    return per_regime_df, global_df


# --------------------
# Pretty leaderboard printing and winner-agreement analysis
# --------------------
def print_and_save_leaderboards(per_regime_df: pd.DataFrame, global_df: pd.DataFrame):
    print("\nGLOBAL leaderboard (top 10 by sharpe):")
    if not global_df.empty:
        print(global_df[['strategy','annual_return','annual_vol','sharpe']].head(10).to_string(index=False))
    else:
        print("No global data")

    metrics_map = {
        'pnl': ('mean_daily_return', True),
        'alpha': ('mean_daily_alpha', True),
        'sharpe': ('sharpe', True)
    }

    winners = {}
    regimes = sorted(list(per_regime_df['regime'].dropna().unique()))
    for r in regimes:
        df_r = per_regime_df[per_regime_df['regime'] == r].copy()
        if df_r.empty:
            continue
        winners[r] = {}
        print(f"\n--- Regime {r} leaderboard ---")
        for metric_name, (col, high_is_better) in metrics_map.items():
            df_sorted = df_r.sort_values(col, ascending=not high_is_better)
            top = df_sorted.head(1)
            if top.empty:
                winner = None
                val = float('nan')
            else:
                winner = top.iloc[0]['strategy']
                val = top.iloc[0][col]
            winners[r][metric_name] = winner
            print(f"Top {metric_name:6s}: {winner:30s} | {col}={val:.6g}")
    # agreement
    print("\nWinner agreement by regime:")
    for r, mapping in winners.items():
        uniq = set([v for v in mapping.values() if v is not None])
        if len(uniq) == 1:
            print(f"Regime {r}: same winner for pnl/alpha/sharpe -> {uniq.pop()}")
        else:
            print(f"Regime {r}: different winners -> {mapping}")

    # Save CSVs
    per_regime_df.to_csv(os.path.join(OUT_DIR, "per_regime_metrics_regime6_adapt.csv"), index=False)
    global_df.to_csv(os.path.join(OUT_DIR, "global_metrics_regime6_adapt.csv"), index=False)
    print(f"\nSaved CSV reports to {OUT_DIR}/")


# --------------------
# Main
# --------------------
def main():
    LOG.info("Engine start — trying to obtain regimes from regime6 (no changes to regime6)")

    # prepare data & strategies
    dm = DataManager(cache_dir="./data_cache")
    LOG.info("Fetching market data %s %s->%s", TICKER, START_DATE, END_DATE)
    market_data = dm.fetch_data(symbol=TICKER, start_date=START_DATE, end_date=END_DATE)

    strat_reg = StrategyRegistry()
    strat_reg.register(BuyAndHold())
    strat_reg.register(SMACrossover(fast_period=20, slow_period=50))
    strat_reg.register(SMACrossover(fast_period=50, slow_period=200))
    strat_reg.register(BollingerMeanReversion(period=20, num_std=2.0))
    strat_reg.register(RSIMomentum(period=14))

    # obtain regimes (strictly from regime6)
    market_index = market_data.index
    per_method_map, ensemble_series, ensemble_conf = obtain_regimes_strict(market_index)

    LOG.info("Per-method detectors found: %s", list(per_method_map.keys()))
    LOG.info("Ensemble regimes unique values: %s", sorted(list(ensemble_series.unique())) if ensemble_series is not None else "None")

    # backtests
    LOG.info("Running backtests for strategies...")
    all_results = run_backtests(strat_reg, market_data, market_data.copy())

    # build master table
    LOG.info("Building master table aligned to market index...")
    master = build_master_table(all_results, market_index, ensemble_series, ensemble_conf)
    master_fp = os.path.join(OUT_DIR, "master_regime6_adapt.parquet")
    master.to_parquet(master_fp, index=False)
    LOG.info("Wrote master -> %s rows=%d", master_fp, len(master))

    # metrics
    per_regime_df, global_df = compute_performance_tables(master)
    print_and_save_leaderboards(per_regime_df, global_df)

    LOG.info("Engine complete.")


if __name__ == "__main__":
    main()
