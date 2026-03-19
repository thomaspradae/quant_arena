#!/usr/bin/env python3
"""
Walk-Forward ELO Engine v1.0 - No Look-Ahead Bias

This implements proper walk-forward testing where:
1. Train ELO on period T using only data up to T
2. Test predictions on period T+1
3. Regime detection uses only past data
4. No circular logic or look-ahead bias

Key principle: At any point in time, we only use information that would have been
available at that time in real trading.
"""

# (Mocks and imports for standalone execution)
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass  # CORRECTED: Added the missing import
from itertools import combinations
from abc import ABC, abstractmethod
from scipy import stats
import yfinance as yf
import json

# --- Mock/Placeholder Classes ---
# In your real project, these would be imported from your other files.
class Strategy(ABC):
    def __init__(self): self.name = self.__class__.__name__; self.strategy_type = "unknown"
    @abstractmethod
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series: pass
    def warmup_period(self) -> int: return 0
class BuyAndHold(Strategy):
    def __init__(self): super().__init__(); self.name="BuyAndHold"; self.strategy_type="passive"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series: return pd.Series(capital, index=data.index)
class TrendFollowing(Strategy):
    def __init__(self, lookback: int): super().__init__(); self.lookback=lookback; self.name=f"TrendFollow_{lookback}d"; self.strategy_type="momentum"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        ma = data['Close'].rolling(self.lookback).mean()
        return (data['Close'] > ma).astype(float) * capital
    def warmup_period(self) -> int: return self.lookback
class MeanReversion(Strategy):
    def __init__(self, lookback: int): super().__init__(); self.lookback=lookback; self.name=f"MeanReversion_{lookback}d"; self.strategy_type="mean_reversion"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        ret = data['Close'].pct_change()
        z = (ret - ret.rolling(self.lookback).mean())/ret.rolling(self.lookback).std()
        return (z < -1.5).astype(float) * capital
    def warmup_period(self) -> int: return self.lookback
class LowVolatility(Strategy):
    def __init__(self, lookback:int): super().__init__(); self.lookback=lookback; self.name=f"LowVol_{lookback}d"; self.strategy_type="low_volatility"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series: return pd.Series(capital, index=data.index) # Dummy
    def warmup_period(self) -> int: return 252+self.lookback
class VolatilityBreakout(Strategy):
    def __init__(self, lookback:int): super().__init__(); self.lookback=lookback; self.name=f"VolBreakout_{lookback}d"; self.strategy_type="volatility_breakout"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series: return pd.Series(0, index=data.index) # Dummy
    def warmup_period(self) -> int: return 252+self.lookback
class FadeExtremes(Strategy):
    def __init__(self, lookback:int): super().__init__(); self.lookback=lookback; self.name=f"FadeExtremes_{lookback}d"; self.strategy_type="contrarian"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series: return pd.Series(0, index=data.index) # Dummy
    def warmup_period(self) -> int: return self.lookback
class MomentumCrossover(Strategy):
    def __init__(self, fast:int, slow:int): super().__init__(); self.fast=fast; self.slow=slow; self.name=f"MomXover_{fast}_{slow}"; self.strategy_type="momentum"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series: return pd.Series(capital, index=data.index) # Dummy
    def warmup_period(self) -> int: return self.slow
class RSIMeanReversion(Strategy):
    def __init__(self, period:int): super().__init__(); self.period=period; self.name=f"RSI_{period}"; self.strategy_type="mean_reversion"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series: return pd.Series(capital, index=data.index) # Dummy
    def warmup_period(self) -> int: return self.period
class RangeBreakout(Strategy):
    def __init__(self, lookback:int): super().__init__(); self.lookback=lookback; self.name=f"RangeBreak_{lookback}d"; self.strategy_type="breakout"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series: return pd.Series(0, index=data.index) # Dummy
    def warmup_period(self) -> int: return 126+self.lookback
class StrategyRegistry:
    def __init__(self): self.strategies = {}
    def register(self, s): self.strategies[s.name] = s
    def create_default_universe(self):
        self.register(BuyAndHold())
        self.register(TrendFollowing(50)); self.register(MeanReversion(20))
        self.register(LowVolatility(20)); self.register(VolatilityBreakout(20))
        self.register(FadeExtremes(63)); self.register(MomentumCrossover(20,50))
        self.register(RSIMeanReversion(14)); self.register(RangeBreakout(20))
        return self
class DataManager:
    def __init__(self, **kwargs): pass
    def fetch_data(self, symbol, start_date, end_date):
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        sanity_check_market_data(df); return df
class BacktestResult:
    def __init__(self, returns, **kwargs): self.returns = returns; self.alpha_returns=returns
class BacktestEngine:
    def __init__(self, **kwargs): pass
    def run_backtest(self, strat, market_data, capital, benchmark):
        pos = strat.generate_positions(market_data, capital)
        ret = market_data['Close'].pct_change()
        strat_ret = (pos.shift(1)/capital) * ret
        return BacktestResult(returns=strat_ret.fillna(0))
class BayesianELORanking:
    def __init__(self, mu=1500, sigma=350, **kwargs): self.mu=mu; self.sigma=sigma
    def update(self, other, out):
        if out==1.0: self.mu +=5
        elif out==0.0: self.mu -=5
    def get_rating(self, *args, **kwargs): return self.mu
    def get_uncertainty(self, *args, **kwargs): return self.sigma
class RankingManager:
    def __init__(self, ranking_class, **kwargs): self.rc=ranking_class; self.kw=kwargs; self.rk={'alpha':{None:{}}}
    def update(self, s1,s2,out,met,reg,ts):
        if s1 not in self.rk[met][None]: self.rk[met][None][s1] = self.rc(**self.kw)
        if s2 not in self.rk[met][None]: self.rk[met][None][s2] = self.rc(**self.kw)
        self.rk[met][None][s1].update(self.rk[met][None][s2], out)
        self.rk[met][None][s2].update(self.rk[met][None][s1], 1-out)
    def get_leaderboard(self, met, reg):
        rks={n:r.get_rating() for n,r in self.rk[met][None].items()}
        if not rks: return pd.DataFrame(columns=['strategy','rating'])
        return pd.DataFrame(list(rks.items()), columns=['strategy','rating']).sort_values('rating', ascending=False).reset_index(drop=True)
class VolatilityRegimeDetector:
    def __init__(self, **kwargs): pass
    def detect_regimes(self, data): return pd.Series(np.random.randint(0,2,len(data)), index=data.index)
    def expand_to_full_index(self, s, idx, **kwargs): return s.reindex(idx, method='ffill')
def sanity_check_market_data(df, **kwargs):
    if df.empty: raise ValueError("Data empty")
def adapt_strategies(s, c): return s
# --- End Mocks ---

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardResult:
    split_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    top_3_strategies: List[str]
    bottom_3_strategies: List[str]
    top_3_return: float
    bottom_3_return: float
    all_return: float
    spread: float
    elo_accuracy: float

class WalkForwardELOEngine:
    def __init__(self, n_splits: int = 5, min_train_days: int = 252, test_days: int = 63, ranking_kwargs: Optional[Dict] = None):
        self.n_splits = n_splits
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.ranking_kwargs = ranking_kwargs or {}
        self.data_manager = DataManager(cache_dir='./data_cache')
        self.backtest_engine = BacktestEngine(benchmark_symbol='SPY')
        self.strategy_registry = StrategyRegistry().create_default_universe()
        self.results: List[WalkForwardResult] = []
    
    def run_walk_forward_validation(self, symbol: str, start_date: str, end_date: str, initial_capital: float = 100_000) -> pd.DataFrame:
        print("="*100)
        print("WALK-FORWARD ELO VALIDATION - NO LOOK-AHEAD BIAS")
        print("="*100)
        self.adapted_strategies = adapt_strategies(list(self.strategy_registry.strategies.values()), capital=initial_capital)
        print(f"\n[Step 1] Fetching market data for {symbol}...")
        market_data = self.data_manager.fetch_data(symbol=symbol, start_date=start_date, end_date=end_date)
        all_dates = market_data.index
        print(f"Total trading days: {len(all_dates)}")
        print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
        
        split_boundaries = self._calculate_rolling_split_boundaries(all_dates)
        print(f"\n[Step 2] Running {len(split_boundaries)} walk-forward splits...")
        
        for split_idx, (train_start_idx, train_end_idx, test_start_idx, test_end_idx) in enumerate(split_boundaries):
            print(f"\n=== SPLIT {split_idx + 1}/{len(split_boundaries)} ===")
            train_dates = all_dates[train_start_idx : train_end_idx + 1]
            test_dates = all_dates[test_start_idx : test_end_idx + 1]
            print(f"Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
            print(f"Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
            
            result = self._run_single_split(split_idx, train_dates, test_dates, market_data, market_data.copy(), initial_capital)
            self.results.append(result)
            
            print(f"Top-3 ELO strategies: {result.top_3_strategies}")
            print(f"Test returns - Top-3: {result.top_3_return:.2%}, Bottom-3: {result.bottom_3_return:.2%}")
            print(f"Spread: {result.spread:.2%}, ELO Accuracy: {result.elo_accuracy:.1%}")
            
        return self._aggregate_results()

    def _calculate_rolling_split_boundaries(self, all_dates: pd.DatetimeIndex) -> List[Tuple[int, int, int]]:
        total_days = len(all_dates)
        train_window = 2 * 252
        step_size = self.test_days
        splits_possible = (total_days - train_window) // step_size
        if splits_possible < 2: raise ValueError(f"Not enough data for rolling window. Need > {train_window + 2*step_size} days")
        
        boundaries = []
        for i in range(splits_possible):
            train_start_idx = i * step_size
            train_end_idx = train_start_idx + train_window - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + self.test_days - 1, total_days - 1)
            if test_end_idx <= test_start_idx or train_end_idx >= total_days: break
            boundaries.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))
        return boundaries

    def _run_single_split(self, split_idx, train_dates, test_dates, market_data, benchmark_data, capital) -> WalkForwardResult:
        print("  Training ELO on historical data...")
        train_market_data = market_data.loc[train_dates]
        regimes = self._detect_regimes_no_lookahead(train_market_data)
        train_results = self._run_backtests_on_period(train_market_data, benchmark_data.loc[train_dates], capital)
        ranking_manager = self._train_elo_ranking(train_results, regimes, train_dates)
        
        leaderboard = ranking_manager.get_leaderboard('alpha', None)
        if leaderboard.empty: raise RuntimeError(f"No ELO rankings for split {split_idx}")
        
        top_3, bottom_3 = leaderboard.head(3)['strategy'].tolist(), leaderboard.tail(3)['strategy'].tolist()
        
        print("  Testing on future data...")
        test_market_data = market_data.loc[test_dates]
        test_results = self._run_backtests_on_period(test_market_data, benchmark_data.loc[test_dates], capital)
        test_returns = {name: res.returns.mean()*252 for name, res in test_results.items()}
        
        top_3_ret = np.mean([test_returns.get(s,0) for s in top_3])
        bot_3_ret = np.mean([test_returns.get(s,0) for s in bottom_3])
        
        return WalkForwardResult(
            split_idx, train_dates[0], train_dates[-1], test_dates[0], test_dates[-1],
            top_3, bottom_3, top_3_ret, bot_3_ret, np.mean(list(test_returns.values())),
            top_3_ret - bot_3_ret, self._calculate_elo_accuracy(test_results, top_3, bottom_3, test_dates)
        )

    def _detect_regimes_no_lookahead(self, data: pd.DataFrame) -> pd.Series:
        try:
            detector = VolatilityRegimeDetector(window=20, num_regimes=2, method='quantile')
            regimes = detector.detect_regimes(data)
            return detector.expand_to_full_index(regimes, data.index, method='ffill')
        except Exception as e:
            return pd.Series(0, index=data.index, name='regime')

    def _run_backtests_on_period(self, market_data, benchmark_data, capital) -> Dict[str, BacktestResult]:
        results = {}
        for strat in self.adapted_strategies:
            try:
                res = self.backtest_engine.run_backtest(strat, market_data, capital, benchmark_data)
                res.returns = res.returns.reindex(market_data.index).fillna(0)
                results[strat.name] = res
            except Exception as e:
                results[strat.name] = BacktestResult(returns=pd.Series(0, index=market_data.index))
        return results

    def _train_elo_ranking(self, results, regimes, dates) -> RankingManager:
        rm = RankingManager(BayesianELORanking, **self.ranking_kwargs)
        pairs = list(combinations(list(results.keys()), 2))
        for date in dates:
            regime = regimes.get(date)
            scores = {name: res.returns.get(date,0.0) for name,res in results.items()}
            for s1,s2 in pairs:
                if np.isnan(scores[s1]) or np.isnan(scores[s2]): continue
                outcome = 0.5 if scores[s1]==scores[s2] else (1.0 if scores[s1]>scores[s2] else 0.0)
                rm.update(s1,s2,outcome,'alpha',regime,date)
                rm.update(s1,s2,outcome,'alpha',None,date)
        return rm

    def _calculate_elo_accuracy(self, test_results, top_3, bottom_3, test_dates) -> float:
        if not top_3 or not bottom_3: return 0.0
        wins = 0
        for date in test_dates:
            top_avg = np.mean([test_results[s].returns.get(date,0) for s in top_3])
            bot_avg = np.mean([test_results[s].returns.get(date,0) for s in bottom_3])
            if top_avg > bot_avg: wins+=1
        return wins/len(test_dates) if len(test_dates)>0 else 0.0

    def _aggregate_results(self) -> pd.DataFrame:
        df = pd.DataFrame([r.__dict__ for r in self.results])
        print("\n" + "="*100)
        print("WALK-FORWARD ELO VALIDATION RESULTS")
        print("="*100)
        print(f"Average Spread (Top-3 vs Bottom-3): {df['spread'].mean():.2%}")
        print(f"Positive Spreads: {(df['spread'] > 0).sum()}/{len(df)} splits")
        counts = pd.Series([s for sublist in df['top_3_strategies'] for s in sublist]).value_counts()
        print("\nMost frequently in top-3:")
        for strat, count in counts.head(5).items(): print(f"  {strat}: {count}/{len(df)} splits")
        return df

def compute_elo_predictive_power(validation_results):
    spreads = validation_results['spread']
    if len(spreads) < 2: return {}
    mean_spread = spreads.mean()
    t_stat, p_value = stats.ttest_1samp(spreads, 0, alternative='greater')
    win_rate = (spreads > 0).mean()
    print("\n" + "="*80); print("ELO PREDICTIVE POWER TEST"); print("="*80)
    print(f"Mean spread (Top-3 vs Bottom-3): {mean_spread:.2%}")
    print(f"T-statistic: {t_stat:.2f}"); print(f"P-value (one-sided): {p_value:.4f}")
    print(f"Win rate: {win_rate:.1%} ({(spreads > 0).sum()}/{len(spreads)} splits)")
    if p_value < 0.05 and mean_spread > 0: print("\n✓ RESULT: ELO HAS PREDICTIVE POWER")
    else: print("\n✗ RESULT: ELO DOES NOT PREDICT FUTURE PERFORMANCE")
    return {'mean_spread': mean_spread, 't_stat': t_stat, 'p_value': p_value, 'win_rate': win_rate}

def run_walk_forward_validation():
    engine = WalkForwardELOEngine(n_splits=50, min_train_days=252, test_days=21)
    results_df = engine.run_walk_forward_validation()
    stats_results = compute_elo_predictive_power(results_df)
    results_df.to_csv('walk_forward_elo_results.csv', index=False)
    with open('elo_predictive_power_stats.json', 'w') as f: json.dump(stats_results, f, indent=2)
    return results_df, stats_results

if __name__ == "__main__":
    run_walk_forward_validation()