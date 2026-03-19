"""
Main Engine v1.2 - Bayesian ELO as the primary ranking system

This is the updated top-level engine that uses the BayesianELORanking
implementation (with uncertainty tracking) as the ranking system for
alpha and tail-survival tournaments.

Drop this file next to your existing modules (data_manager, regime_detector,
strategy_zoo, backtest_engine, matchmaker, ranking, analyzer) and run.
"""

from data_manager import DataManager
from regime_detector import HMMRegimeDetector, RegimeDetectorRegistry, VolatilityRegimeDetector, HMMRegimeDetector, ChangePointRegimeDetector, MultifractalRegimeDetector
from regime_ensemble import run_all_detectors
from strategy_zoo import StrategyRegistry, BuyAndHold, SMACrossover, BollingerMeanReversion, RSIMomentum
from backtest_engine import BacktestEngine, BacktestResult
from matchmaker import RoundRobinMatcher
from ranking import RankingManager, BayesianELORanking  # <- use Bayesian ranking
from analyzer import MetaELOAnalyzer
from itertools import combinations
from utils import align_series_to_index, sanity_check_market_data

import pandas as pd
import numpy as np
import ruptures as rpt
from typing import Dict, Optional

# New imports for safe snapshot & saving
import os
import glob
import math
import logging

logger = logging.getLogger(__name__)

# Try to import the new detect_regimes_api from regime5 (if available)
try:
    from regime6 import detect_regimes_api
    HAVE_DETECT_API = True
except Exception:
    detect_regimes_api = None
    HAVE_DETECT_API = False

# ---------------------------
# Helper: comparison function
# ---------------------------
def _compare_strategies_alpha(
    result_a: BacktestResult,
    result_b: BacktestResult,
    metric: str = 'alpha'
) -> float:
    """
    Compare strategies on ALPHA (market-neutral returns), not raw returns.
    Returns:
        1.0 if A > B, 0.0 if B > A, 0.5 if tie.
    """
    if metric == 'alpha':
        # Compare on alpha returns if available
        if getattr(result_a, 'alpha_returns', None) is not None and getattr(result_b, 'alpha_returns', None) is not None:
            alpha_a = float(np.nanmean(result_a.alpha_returns))
            alpha_b = float(np.nanmean(result_b.alpha_returns))
        else:
            # Fallback to raw returns (not ideal but safe)
            alpha_a = float(np.nanmean(result_a.returns))
            alpha_b = float(np.nanmean(result_b.returns))

    elif metric == 'tail_survival':
        tail_a = getattr(result_a, 'tail_metrics', {}) or {}
        tail_b = getattr(result_b, 'tail_metrics', {}) or {}

        # We define "survival score" as stress_mean (higher is better) - choose robust fallback
        alpha_a = float(tail_a.get('stress_mean', -np.inf))
        alpha_b = float(tail_b.get('stress_mean', -np.inf))

    else:
        # Generic metric access (from result.metrics)
        alpha_a = float(result_a.metrics.get(metric, -np.inf))
        alpha_b = float(result_b.metrics.get(metric, -np.inf))

    # Comparison with tolerance
    if np.isnan(alpha_a) and np.isnan(alpha_b):
        return 0.5
    if np.isnan(alpha_a):
        return 0.0
    if np.isnan(alpha_b):
        return 1.0

    if alpha_a > alpha_b:
        return 1.0
    elif alpha_b > alpha_a:
        return 0.0
    else:
        return 0.5

# ---------------------------
# Rating history snapshot + safe saving
# ---------------------------
# Buffer and config
rating_history = []  # list of dicts
FLUSH_EVERY = 250  # flush every N snapshots to parquet chunks
_out_chunk_dir = 'rating_history_chunks'
os.makedirs(_out_chunk_dir, exist_ok=True)
_flush_counter = 0

def _safe_iter_players(ranking_obj):
    """Return iterable of (name, player) pairs for several possible ranking implementations."""
    if ranking_obj is None:
        return []
    # Common: ranking_obj.players is a dict
    if hasattr(ranking_obj, 'players') and isinstance(ranking_obj.players, dict):
        return ranking_obj.players.items()
    # Common alternative: ranking_obj.get_players() -> dict or list
    if hasattr(ranking_obj, 'get_players'):
        try:
            players = ranking_obj.get_players()
            if isinstance(players, dict):
                return players.items()
            if isinstance(players, (list, tuple)):
                out = []
                for p in players:
                    name = getattr(p, 'name', getattr(p, 'strategy', None))
                    if name is None:
                        continue
                    out.append((name, p))
                return out
        except Exception:
            pass
    # fallback: 'ratings' attribute
    if hasattr(ranking_obj, 'ratings') and isinstance(ranking_obj.ratings, dict):
        return ranking_obj.ratings.items()
    return []

def snapshot_ratings(rm, metric, regime, ts):
    """Append ratings snapshot for metric/regime at timestamp ts (pd.Timestamp or convertible)."""
    global _flush_counter
    ts = pd.Timestamp(ts)
    ranking_obj = rm.rankings.get(metric, {}).get(regime)
    if ranking_obj is None:
        return

    for name, player in _safe_iter_players(ranking_obj):
        try:
            mu = float(getattr(player, 'mu', getattr(player, 'rating', float('nan'))))
        except Exception:
            mu = float('nan')
        try:
            sigma = float(getattr(player, 'sigma', getattr(player, 'uncertainty', float('nan'))))
        except Exception:
            sigma = float('nan')
        # optional fields if present
        try:
            conservative = float(getattr(player, 'conservative_rating', math.nan))
        except Exception:
            conservative = math.nan
        try:
            games = int(getattr(player, 'games', getattr(player, 'matches', math.nan)))
        except Exception:
            games = None

        rating_history.append({
            'date': ts,
            'metric': metric,
            'regime': regime,
            'strategy': name,
            'mu': mu,
            'sigma': sigma,
            'conservative_rating': conservative,
            'games': games
        })

    _flush_counter += 1
    # Periodic flush to parquet to avoid memory spikes
    if _flush_counter >= FLUSH_EVERY:
        _flush_counter = 0
        df_chunk = pd.DataFrame(rating_history)
        if not df_chunk.empty:
            fname = os.path.join(_out_chunk_dir, f'rating_chunk_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.parquet')
            df_chunk.to_parquet(fname, index=False, compression='gzip')
            rating_history.clear()

def save_final_rating_history(out_filename='rating_history.parquet'):
    """Combine chunks and in-memory buffer, write a single parquet file."""
    frames = []
    chunk_files = sorted(glob.glob(os.path.join(_out_chunk_dir, '*.parquet')))
    for cf in chunk_files:
        try:
            frames.append(pd.read_parquet(cf))
        except Exception:
            try:
                frames.append(pd.read_pickle(cf))
            except Exception:
                pass
    if rating_history:
        frames.append(pd.DataFrame(rating_history))
    if not frames:
        print("No rating history to save.")
        return
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values(['date','metric','regime','strategy']).reset_index(drop=True)
    df_all.to_parquet(out_filename, index=False, compression='gzip')
    print(f"Saved final rating history to {out_filename} (rows={len(df_all)})")

# ---------------------------
# Main run loop
# ---------------------------
def run():
    print("="*100)
    print("Trading Strategy ELO Engine v1.2 - Bayesian ELO (with Uncertainty) + Tail Risk")
    print("="*100)

    # Initialize components
    print("\n[Step 1] Initializing components...")
    data_manager = DataManager(cache_dir='./data_cache')
    regime_detector = HMMRegimeDetector(num_regimes=3)
    strategy_registry = StrategyRegistry()

    # Backtest engine (expects to support benchmark_data and to populate alpha/beta on results)
    backtest_engine = BacktestEngine(benchmark_symbol='SPY')

    # Matchmaker: round-robin across strategies and regimes
    matchmaker = RoundRobinMatcher(include_global=True)

    # Use the BayesianELORanking as the ranking backend for the RankingManager.
    ranking_manager = RankingManager(
        ranking_class=BayesianELORanking,
        initial_mu=1500,
        initial_sigma=350,
        min_sigma=50,
        tau=1.0,
        base_k=32
    )

    # Register strategies (simple set for demo)
    print("\n[Step 2] Registering strategies...")
    strategy_registry.register(BuyAndHold())
    strategy_registry.register(SMACrossover(fast_period=20, slow_period=50))
    strategy_registry.register(SMACrossover(fast_period=50, slow_period=200))
    strategy_registry.register(BollingerMeanReversion(period=20, num_std=2.0))
    strategy_registry.register(RSIMomentum(period=14))

    # Experiment parameters
    target_symbol = 'SPY'
    benchmark_symbol = 'SPY'
    start_date = '1997-01-01'
    end_date = '2024-01-01'
    initial_capital = 100_000

    try:
        # Fetch market and benchmark data
        print(f"\n[Step 3] Fetching market data for {target_symbol}...")
        market_data = data_manager.fetch_data(symbol=target_symbol, start_date=start_date, end_date=end_date)

        sanity_check_market_data(market_data, require_cols=['close', 'volume'], nan_threshold=0.05)

        # For now benchmark = target; production could fetch separately
        benchmark_data = market_data.copy()

        print("\n[Step 4] Detecting regimes (try new detect_regimes_api first, then fallback to registry.detect_regimes_auto)...")

        # First attempt: call the simple deterministic detect_regimes_api if available
        ensemble_regimes = None
        ensemble_confidence = None
        per_method = {}
        ic_scores = {}

        if HAVE_DETECT_API:
            try:
                print("Attempting detect_regimes_api from regime6.py ...")
                api_res = detect_regimes_api(
                    ticker=target_symbol,
                    start=start_date,
                    end=end_date,
                    n_states=3,
                    methods=['hmm', 'gmm', 'kmeans'],
                    seed=42,
                    save_outputs=True,
                    output_dir='regimes_engine_outputs'
                )
                per_method = api_res.get('per_method', {}) or {}
                ensemble_regimes = api_res.get('ensemble')
                ensemble_confidence = api_res.get('confidence')
                print("detect_regimes_api succeeded. Methods:", list(per_method.keys()))
            except Exception as ex:
                logger.warning(f"detect_regimes_api failed: {ex}")
                ensemble_regimes = None
                ensemble_confidence = None
                per_method = {}

        # Fallback: if we didn't get ensemble results, run the registry auto detection as before
        if ensemble_regimes is None or (hasattr(ensemble_regimes, 'empty') and ensemble_regimes.empty):
            try:
                print("Falling back to registry-based detection...")
                # Create registry and detectors (tweak params if you want)
                registry = RegimeDetectorRegistry()

                # Volatility detector (fast, interpretable)
                vol_det = VolatilityRegimeDetector(window=20, num_regimes=2, method='quantile')
                registry.register(vol_det)

                # HMM detector (robustified). We'll register but selection happens in detect_regimes_auto
                hmm_det = HMMRegimeDetector(num_regimes=3, covariance_type='full', n_iter=200, n_init=4)
                registry.register(hmm_det)

                # Change point detector (requires ruptures)
                try:
                    cp_det = ChangePointRegimeDetector(method='pelt', penalty=10, min_size=20)
                    registry.register(cp_det)
                except Exception as ex:
                    logger.warning(f"ChangePoint detector unavailable: {ex}")

                # Multifractal detector (Hurst, tail)
                mf_det = MultifractalRegimeDetector(window=60, num_regimes=3)
                registry.register(mf_det)

                # Decide which methods to run (None -> all registered)
                methods_to_run = registry.list_all()

                # HMM model selection and changepoint grid
                hmm_states = [2, 3]               # search candidate numbers of states
                hmm_cov_types = ['full', 'diag']  # try a couple covariance types
                changepoint_penalty_grid = list(np.logspace(0, 2, 12))

                auto_res = registry.detect_regimes_auto(
                    data=market_data,
                    methods=methods_to_run,
                    hmm_states=hmm_states,
                    hmm_cov_types=hmm_cov_types,
                    model_selection_ic='bic',              # 'bic' recommended by default
                    changepoint_penalty_grid=changepoint_penalty_grid,
                    expand_to_index=market_data.index,     # return series aligned to market_data index
                    ensemble_method='voting'               # majority vote ensemble (confidence returned too)
                )

                per_method = auto_res.get('per_method', {}) or {}
                ic_scores = auto_res.get('ic_scores', {}) or {}
                ensemble_regimes = auto_res.get('ensemble')
                ensemble_confidence = auto_res.get('confidence')
            except Exception as ex:
                logger.warning(f"Registry detect_regimes_auto failed: {ex}")
                ensemble_regimes = None
                ensemble_confidence = None
                per_method = {}

        # Persist outputs for audibility & debugging
        for name, series in per_method.items():
            try:
                fname = f"regimes_{name}.csv"
                series.to_frame(name='regime').reset_index().to_csv(fname, index=False)
            except Exception as ex:
                logger.debug(f"Could not save regimes for {name}: {ex}")

        # Save ensemble and confidence
        if ensemble_regimes is not None and hasattr(ensemble_regimes, "empty") and not ensemble_regimes.empty:
            try:
                ensemble_regimes.to_frame(name='regime').reset_index().to_csv('regimes_ensemble.csv', index=False)
            except Exception:
                pass
        if ensemble_confidence is not None and hasattr(ensemble_confidence, "empty") and not ensemble_confidence.empty:
            try:
                ensemble_confidence.to_frame(name='confidence').reset_index().to_csv('regimes_ensemble_confidence.csv', index=False)
            except Exception:
                pass

        # Use ensemble for matchmaking (matchmaker expects a pd.Series)
        if ensemble_regimes is None or (hasattr(ensemble_regimes, "empty") and ensemble_regimes.empty):
            # fallback: use volatility detector alone (robust fallback) if vol_det exists
            print("[Step 4] Ensemble empty — falling back to volatility regimes")
            try:
                regimes = vol_det.detect_regimes(market_data)
                regimes = vol_det.expand_to_full_index(regimes, market_data.index, method='ffill')
            except Exception:
                regimes = pd.Series(index=market_data.index, data=[0]*len(market_data), name='regime')
        else:
            regimes = ensemble_regimes.copy()

        # Print quick diagnostics
        print("Detected regime methods:", list(per_method.keys()))
        try:
            unique_vals = sorted(list(regimes.unique()))
        except Exception:
            unique_vals = []
        print("Ensemble regimes unique:", unique_vals)
        if ensemble_confidence is not None:
            print("Ensemble confidence (sample):")
            try:
                print(ensemble_confidence.dropna().head(10))
            except Exception:
                pass


        # Run backtests with benchmark data - populate dictionary of results
        print("\n[Step 5] Running backtests (alpha/beta separation enabled in engine)...")
        all_results: Dict[str, BacktestResult] = {}
        strategies_list = [s for s in strategy_registry.strategies.values()]

        for strategy in strategies_list:
            print(f"   Backtesting {strategy.name} ...", end='', flush=True)
            result: BacktestResult = backtest_engine.run_backtest(
                strategy,
                market_data,
                initial_capital=initial_capital,
                benchmark_data=benchmark_data  # engine should compute alpha/beta if implemented
            )
            all_results[strategy.name] = result

            try:
                # Reindex to the master calendar, then fill any new missing values with 0
                result.returns = result.returns.reindex(market_data.index).fillna(0)
            except Exception:
                # keep original if alignment fails but log
                print(f"Warning: failed to align returns for {strategy.name}")

            if getattr(result, 'alpha_returns', None) is not None:
                try:
                    # Do the same for alpha returns
                    result.alpha_returns = result.alpha_returns.reindex(market_data.index).fillna(0)
                except Exception:
                    print(f"Warning: failed to align alpha_returns for {strategy.name}")    

            print(" done.")

            # Print quick diagnostics when available
            beta_str = f"{result.beta:6.3f}" if getattr(result, 'beta', None) is not None else "  n/a "
            tail_metrics = getattr(result, 'tail_metrics', {}) or {}
            stress_mean = tail_metrics.get('stress_mean', np.nan)
            stress_days = tail_metrics.get('stress_positive_days', None)
            stress_days_str = f"{int(stress_days):2d}" if stress_days is not None else " n/a"
            print(f"      Beta: {beta_str} | StressMean: {stress_mean:7.4f} | StressDays: {stress_days_str}")


                    # -----------------------
            # Post-backtest: validate regimes and generate report
            # -----------------------
            print("\n[Step 5.1] Validating regimes against strategy performance...")

            # Build strategy_returns dictionary required by validation functions
            strategy_returns = {}
            for name, res in all_results.items():
                # prefer alpha_returns if present (market-neutral), else plain returns
                if getattr(res, 'alpha_returns', None) is not None:
                    strategy_returns[name] = res.alpha_returns.reindex(market_data.index).fillna(0)
                else:
                    strategy_returns[name] = res.returns.reindex(market_data.index).fillna(0)

        # Generate matchups (informational - keep for compatibility)
        print("\n[Step 6] Generating round-robin matchups (per regime + global)...")
        matchups = matchmaker.generate_matchups(strategies_list, market_data, regimes, asset=target_symbol)

        # -------------------------------
        # NEW: Run daily tournaments
        # -------------------------------
        print("\n[Step 7] Running DAILY Bayesian ELO tournaments (alpha preferred)...")

        # --- compute stress days from benchmark returns (e.g., worst 5% days)
        benchmark_ret = benchmark_data['close'].pct_change().dropna()
        stress_q = 0.05
        stress_threshold = benchmark_ret.quantile(stress_q)   # e.g., 5th percentile
        stress_mask = benchmark_ret <= stress_threshold       # Series indexed by date: True on stress days

        # Optionally expand to full index if needed
        #stress_mask = stress_mask.reindex(market_data.index).fillna(False)
        stress_mask = stress_mask.reindex(market_data.index).fillna(False).astype(bool)

        # Build evaluation dates: intersection of available dates across strategies (alpha preferred)
        all_dates = None
        for res in all_results.values():
            if getattr(res, 'alpha_returns', None) is not None:
                idx = res.alpha_returns.dropna().index
            else:
                idx = res.returns.dropna().index
            if all_dates is None:
                all_dates = set(idx)
            else:
                all_dates = all_dates.intersection(set(idx))

        if not all_dates:
            raise RuntimeError("No overlapping evaluation dates found across strategy results.")

        eval_dates = sorted(list(all_dates))
        print(f"  Evaluation dates (count): {len(eval_dates)}")

        # Precompute pairs
        pairs = list(combinations(strategies_list, 2))

        match_counter = 0
        # Loop through dates and update ranking per pair per day
        for date in eval_dates:
            regime_label = None
            try:
                if regimes is not None and date in regimes.index:
                    regime_label = regimes.loc[date]
            except Exception:
                regime_label = None

            # Build daily scores (alpha preferred)
            scores = {}
            for strat in strategies_list:
                res = all_results[strat.name]
                score = np.nan
                if getattr(res, 'alpha_returns', None) is not None:
                    try:
                        score = float(res.alpha_returns.loc[date])
                    except Exception:
                        score = np.nan
                if np.isnan(score):
                    try:
                        score = float(res.returns.loc[date])
                    except Exception:
                        score = np.nan
                scores[strat.name] = score

            # Pairwise comparisons
            for strat_a, strat_b in pairs:
                sa = scores.get(strat_a.name, np.nan)
                sb = scores.get(strat_b.name, np.nan)

                if np.isnan(sa) or np.isnan(sb):
                    # skip incomplete data for this date
                    continue

                eps = 1e-12
                if sa > sb + eps:
                    outcome = 1.0
                elif sb > sa + eps:
                    outcome = 0.0
                else:
                    outcome = 0.5

                ts = pd.Timestamp(date)
                ranking_manager.update(
                    strategy_a_name=strat_a.name,
                    strategy_b_name=strat_b.name,
                    outcome=outcome,
                    metric='alpha',
                    regime=regime_label,
                    timestamp=ts
                )

                # Also update GLOBAL (regime=None) so we accumulate an overall rating.
                # This keeps per-regime learning AND a running global leaderboard.
                ranking_manager.update(
                    strategy_a_name=strat_a.name,
                    strategy_b_name=strat_b.name,
                    outcome=outcome,
                    metric='alpha',
                    regime=None,
                    timestamp=ts
                )

                match_counter += 1

            snapshot_ratings(ranking_manager, 'alpha', regime_label, ts)
            snapshot_ratings(ranking_manager, 'alpha', None, ts)
            # also snapshot tail_survival if you want
            snapshot_ratings(ranking_manager, 'tail_survival', regime_label, ts)
            snapshot_ratings(ranking_manager, 'tail_survival', None, ts)

            # -----------------------------
            # Per-day tail_survival updates
            # -----------------------------
            # If today is a market stress day, reward strategies that survived (positive returns).
            is_stress = bool(stress_mask.loc[date]) if date in stress_mask.index else False

            if is_stress:
                # build tail_scores dict same shape as 'scores'
                tail_scores = {}
                for strat in strategies_list:
                    res = all_results[strat.name]
                    s_val = np.nan
                    if getattr(res, 'alpha_returns', None) is not None:
                        try:
                            s_val = float(res.alpha_returns.loc[date])
                        except Exception:
                            s_val = np.nan
                    if np.isnan(s_val):
                        try:
                            s_val = float(res.returns.loc[date])
                        except Exception:
                            s_val = np.nan
                    tail_scores[strat.name] = s_val

                # pairwise tail updates (reward positive survival on stress day)
                for strat_a, strat_b in pairs:
                    sa = tail_scores.get(strat_a.name, np.nan)
                    sb = tail_scores.get(strat_b.name, np.nan)
                    # skip if either missing
                    if np.isnan(sa) or np.isnan(sb):
                        continue

                    # Outcome logic: prefer strategy with higher return on stress day.
                    eps = 1e-12
                    if sa > sb + eps:
                        outcome = 1.0
                    elif sb > sa + eps:
                        outcome = 0.0
                    else:
                        outcome = 0.5

                    ts = pd.Timestamp(date)
                    ranking_manager.update(
                        strategy_a_name=strat_a.name,
                        strategy_b_name=strat_b.name,
                        outcome=outcome,
                        metric='tail_survival',
                        regime=regime_label,   # keep regime-specific tail learning
                        timestamp=ts
                    )

                    # also update global tail_survival
                    ranking_manager.update(
                        strategy_a_name=strat_a.name,
                        strategy_b_name=strat_b.name,
                        outcome=outcome,
                        metric='tail_survival',
                        regime=None,
                        timestamp=ts
                    )

                # ---------------------------
        # AGGREGATE & REPORT (drop-in)
        # Run this AFTER your tournaments/backtests and before final leaderboards
        # Produces: master.parquet, per_regime_metrics.csv, regime_predictiveness.csv
        # ---------------------------

        import pathlib
        from collections import defaultdict
        from math import isnan

        OUT_DIR = pathlib.Path("reports")
        OUT_DIR.mkdir(exist_ok=True)

        def safe_series(s):
            # ensure index is datetime and dtype float
            return s.reindex(market_data.index).astype(float)

        # 1) Build master table: rows = (date, strategy), columns = returns, alpha_returns, regime, stress_flag
        rows = []
        for name, res in all_results.items():
            # prefer alpha_returns if present for "alpha" metric, else raw returns
            raw = getattr(res, 'returns', None)
            alpha = getattr(res, 'alpha_returns', None)
            # align to master calendar
            if raw is None:
                continue
            raw = raw.reindex(market_data.index).fillna(0).astype(float)
            if alpha is not None:
                alpha = alpha.reindex(market_data.index).fillna(0).astype(float)
            else:
                alpha = raw.copy()  # fallback: raw returns serve as alpha
            for dt in market_data.index:
                rows.append({
                    'date': pd.Timestamp(dt),
                    'strategy': name,
                    'return': float(raw.loc[dt]) if dt in raw.index else float('nan'),
                    'alpha': float(alpha.loc[dt]) if dt in alpha.index else float('nan'),
                    'regime': int(regimes.loc[dt]) if (regimes is not None and dt in regimes.index and not pd.isna(regimes.loc[dt])) else None,
                    'stress': bool(ensemble_confidence.loc[dt] <= 0.5) if (ensemble_confidence is not None and dt in ensemble_confidence.index and not pd.isna(ensemble_confidence.loc[dt])) else False
                })

        master = pd.DataFrame(rows)
        master = master.sort_values(['strategy','date']).reset_index(drop=True)

        # Save canonical master file (parquet is single-file, columnar, fast to read)
        master_fp = OUT_DIR / "master.parquet"
        master.to_parquet(master_fp, index=False, compression='gzip')
        print(f"[REPORT] Wrote master table -> {master_fp} (rows={len(master)})")

        # 2) Per-regime performance metrics for each strategy
        # Metrics: n_days, mean_return, annualized_return, vol, sharpe (ann), max_drawdown, win_rate
        def annualize_return(mean_daily_return, periods_per_year=252):
            return (1 + mean_daily_return) ** periods_per_year - 1

        def compute_max_drawdown(series):
            # series is cumulative returns series (1 + r).cumprod()
            wealth = (1 + series.fillna(0)).cumprod()
            peak = wealth.cummax()
            dd = (wealth - peak) / peak
            return float(dd.min())

        def compute_metrics(group):
            # group is DataFrame for one strategy + regime
            valid = group['return'].dropna()
            n = len(valid)
            mean_ret = float(valid.mean()) if n else float('nan')
            vol = float(valid.std() * np.sqrt(252)) if n else float('nan')
            annual_ret = annualize_return(mean_ret) if n else float('nan')
            sharpe = float(annual_ret / vol) if (n and vol and vol != 0) else float('nan')
            win_rate = float((valid > 0).sum() / n) if n else float('nan')
            try:
                mdd = compute_max_drawdown(valid)
            except Exception:
                mdd = float('nan')
            return {
                'n_days': int(n),
                'mean_daily_return': mean_ret,
                'annual_return': annual_ret,
                'annual_vol': vol,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'max_drawdown': mdd
            }

        per_regime_rows = []
        for strategy, df_s in master.groupby('strategy'):
            for regime_label, df_sr in df_s.groupby('regime'):
                metrics = compute_metrics(df_sr)
                row = {'strategy': strategy, 'regime': regime_label}
                row.update(metrics)
                per_regime_rows.append(row)

        per_regime_df = pd.DataFrame(per_regime_rows)
        per_regime_fp = OUT_DIR / "per_regime_metrics.csv"
        per_regime_df.to_csv(per_regime_fp, index=False)
        print(f"[REPORT] Wrote per-regime metrics -> {per_regime_fp} (rows={len(per_regime_df)})")

        # 3) Overall per-strategy metrics (global)
        global_rows = []
        for strategy, df_s in master.groupby('strategy'):
            metrics = compute_metrics(df_s)
            row = {'strategy': strategy}
            row.update(metrics)
            global_rows.append(row)
        global_df = pd.DataFrame(global_rows).sort_values('sharpe', ascending=False)
        global_fp = OUT_DIR / "global_metrics.csv"
        global_df.to_csv(global_fp, index=False)
        print(f"[REPORT] Wrote global metrics -> {global_fp}")

        # 4) Predictiveness: do regimes change strategy ranking? (Spearman rank-correlation approach)
        # For each regime, rank strategies by mean alpha (or mean return), compute pairwise Spearman correlations across regimes,
        # Predictiveness = 1 - mean_correlation (0..1)
        from scipy.stats import spearmanr
        regime_rankings = {}
        for regime in sorted(master['regime'].dropna().unique()):
            mask = master['regime'] == regime
            sub = master[mask]
            perf = sub.groupby('strategy')['alpha'].mean().dropna()
            if len(perf) >= 3:
                ranked = perf.sort_values(ascending=False).index.tolist()
                regime_rankings[regime] = ranked

        corrs = []
        regimes_list = list(regime_rankings.items())
        for i in range(len(regimes_list)):
            for j in range(i+1, len(regimes_list)):
                r1, ranks1 = regimes_list[i]
                r2, ranks2 = regimes_list[j]
                common = list(set(ranks1).intersection(set(ranks2)))
                if len(common) < 3:
                    continue
                pos1 = [ranks1.index(s) for s in common]
                pos2 = [ranks2.index(s) for s in common]
                c,_ = spearmanr(pos1, pos2)
                if not pd.isna(c):
                    corrs.append(c)
        predictiveness = float(max(0.0, min(1.0, 1.0 - (float(np.mean(corrs)) if corrs else 0.0))))
        pred_fp = OUT_DIR / "regime_predictiveness.csv"
        pd.DataFrame([{
            'predictiveness': predictiveness,
            'n_regimes': len(regime_rankings),
            'n_pairs_considered': len(corrs)
        }]).to_csv(pred_fp, index=False)
        print(f"[REPORT] Wrote predictiveness -> {pred_fp} (value={predictiveness:.3f})")

        # 5) Write consolidated leaderboards + rating history (single CSVs)
        try:
            # rating_history is the in-memory list of dicts; convert and save
            rh_df = pd.DataFrame(rating_history)
            if not rh_df.empty:
                rh_fp = OUT_DIR / "rating_history_snapshot.parquet"
                rh_df.to_parquet(rh_fp, index=False, compression='gzip')
                print(f"[REPORT] Wrote rating history snapshot -> {rh_fp} (rows={len(rh_df)})")
        except Exception as e:
            print("[REPORT] Could not save rating_history:", e)

        # 6) Summary report
        summary = {
            'master_rows': len(master),
            'per_regime_rows': len(per_regime_df),
            'global_strategies': len(global_df),
            'predictiveness': predictiveness
        }
        summary_fp = OUT_DIR / "report_summary.json"
        import json
        summary_fp.write_text(json.dumps(summary, indent=2))
        print(f"[REPORT] Summary -> {summary_fp}")

        # Done. Print top-level pointers for convenience
        print("\n[REPORT] Files written to ./reports/")
        print("  - master.parquet         : per-date/per-strategy canonical table")
        print("  - per_regime_metrics.csv : per-strategy per-regime metrics (Sharpe, vol, MDD, etc.)")
        print("  - global_metrics.csv     : global (all-time) metrics per strategy")
        print("  - regime_predictiveness.csv : how informative regimes are (0..1)")
        print("  - rating_history_snapshot.parquet : ranking snapshots")

        print(f"\nDaily tournament complete. Total daily matches processed: {match_counter}")

        # Save rating history (final) before printing final leaderboards
        save_final_rating_history('rating_history.parquet')

        # Final leaderboards: use BayesianELORanking's uncertainty-aware leaderboard where available
        print("\n" + "="*100)
        print("FINAL BAYESIAN ELO LEADERBOARDS (Alpha & Tail Survival)")
        print("="*100)

        # Alpha (global)
        print("\n--- BAYESIAN ALPHA LEADERBOARD (global) ---")
        # RankingManager stores ranking objects in rankings[metric][regime]
        alpha_global_ranking_system = ranking_manager.rankings.get('alpha', {}).get(None)
        if alpha_global_ranking_system is not None and hasattr(alpha_global_ranking_system, 'get_leaderboard_with_uncertainty'):
            print(alpha_global_ranking_system.get_leaderboard_with_uncertainty(sort_by='conservative_rating', top_n=None).to_string(index=True))
        else:
            print(ranking_manager.get_leaderboard(metric='alpha', regime=None).to_string())

        # Tail survival (global)
        print("\n--- BAYESIAN TAIL SURVIVAL LEADERBOARD (global) ---")
        tail_global_ranking_system = ranking_manager.rankings.get('tail_survival', {}).get(None)
        if tail_global_ranking_system is not None and hasattr(tail_global_ranking_system, 'get_leaderboard_with_uncertainty'):
            print(tail_global_ranking_system.get_leaderboard_with_uncertainty(sort_by='conservative_rating', top_n=None).to_string(index=True))
        else:
            print(ranking_manager.get_leaderboard(metric='tail_survival', regime=None).to_string())

        # Per-regime leaderboards (alpha)
        regimes_seen = sorted(list(regimes.unique())) if hasattr(regimes, 'unique') else []
        for regime_label in regimes_seen:
            print(f"\n--- BAYESIAN ALPHA LEADERBOARD (Regime {regime_label}) ---")
            ranking_sys = ranking_manager.rankings.get('alpha', {}).get(regime_label)
            if ranking_sys is not None and hasattr(ranking_sys, 'get_leaderboard_with_uncertainty'):
                print(ranking_sys.get_leaderboard_with_uncertainty(sort_by='conservative_rating').to_string(index=True))
            else:
                df = ranking_manager.get_leaderboard(metric='alpha', regime=regime_label)
                if not df.empty:
                    print(df.to_string())
                else:
                    print(" (no data) ")

        # Meta-analysis using MetaELOAnalyzer (works over RankingManager)
        print("\n" + "="*100)
        print("META ANALYSIS (Bayesian ELO)")
        print("="*100)

        meta_analyzer = MetaELOAnalyzer(ranking_manager)

        print("\n--- ELO VELOCITY (Alpha metric, global) ---")
        velocity = meta_analyzer.get_velocity_ranking(metric='alpha', regime=None)
        if isinstance(velocity, pd.DataFrame) and not velocity.empty:
            print(velocity.to_string())
        else:
            print("(no velocity data)")

        print("\n--- STRATEGY LIFECYCLE (Alpha metric, global) ---")
        lifecycle = meta_analyzer.get_lifecycle_report(metric='alpha', regime=None)
        if isinstance(lifecycle, pd.DataFrame) and not lifecycle.empty:
            print(lifecycle.to_string())
        else:
            print("(no lifecycle data)")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

        # Save partial rating history on error too
        try:
            save_final_rating_history('rating_history_partial_on_error.parquet')
        except Exception:
            # fallback: pickle small buffer if parquet fails
            df_rating_history = pd.DataFrame(rating_history)
            df_rating_history.to_pickle('rating_history_error.pkl')

if __name__ == "__main__":
    print("="*100)
    print("CHOOSE TESTING MODE:")
    print("1. Sensitivity Analysis (Multiple Training Windows) - RECOMMENDED")
    print("2. Walk-Forward ELO Validation (Single Window) - NO LOOK-AHEAD BIAS")
    print("3. Original Engine (HAS LOOK-AHEAD BIAS) - FOR COMPARISON ONLY")
    print("="*100)
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\nRunning Sensitivity Analysis...")
        from walk_forward_engine import run_sensitivity_analysis
        run_sensitivity_analysis()
    elif choice == "2":
        print("\nRunning Walk-Forward ELO Validation...")
        from walk_forward_engine import run_walk_forward_validation
        run_walk_forward_validation()
    elif choice == "3":
        print("\nRunning Original Engine (WARNING: Has look-ahead bias)...")
        run()
    else:
        print("Invalid choice. Running Sensitivity Analysis by default...")
        from walk_forward_engine import run_sensitivity_analysis
        run_sensitivity_analysis()
