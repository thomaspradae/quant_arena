"""
Walk-Forward ELO Engine v2.0 - With Statistical Validation

This final version integrates a rigorous statistical test (one-sided t-test) 
to validate whether the ELO rankings have genuine predictive power. It uses
a rolling-window approach to test across the entire dataset.

Key principle: At any point in time, we only use information that would have been
available at that time in real trading.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass # CORRECTED: Added the missing import
from itertools import combinations
from abc import ABC, abstractmethod
from scipy import stats
import yfinance as yf
import json

# --- Mock Classes (for standalone execution) ---
# In your real project, you would import these from your other files.

class Strategy(ABC):
    def __init__(self):
        self.name: str = self.__class__.__name__
        self.strategy_type: str = "unknown"
    @abstractmethod
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        pass
    def warmup_period(self) -> int:
        return 0

class BuyAndHold(Strategy):
    def __init__(self):
        super().__init__()
        self.strategy_type = "passive"
        self.name = "BuyAndHold"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        return pd.Series(capital, index=data.index)

class TrendFollowing(Strategy):
    def __init__(self, lookback: int = 50):
        super().__init__()
        self.strategy_type = "momentum"
        self.lookback = lookback
        self.name = f"TrendFollow_{lookback}d"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        ma = data['Close'].rolling(self.lookback).mean()
        signal = (data['Close'] > ma).astype(float)
        return signal * capital
    def warmup_period(self) -> int:
        return self.lookback

class MeanReversion(Strategy):
    def __init__(self, lookback: int = 20):
        super().__init__()
        self.strategy_type = "mean_reversion"
        self.lookback = lookback
        self.name = f"MeanReversion_{lookback}d"
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        returns = data['Close'].pct_change()
        z_score = (returns - returns.rolling(self.lookback).mean()) / returns.rolling(self.lookback).std()
        signal = (z_score < -1.5).astype(float)
        return signal * capital
    def warmup_period(self) -> int:
        return self.lookback

class StrategyRegistry:
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
    def register(self, strategy: Strategy):
        self.strategies[strategy.name] = strategy
    def create_default_universe(self):
        self.register(BuyAndHold())
        self.register(TrendFollowing(50))
        self.register(TrendFollowing(200))
        self.register(MeanReversion(20))
        self.register(MeanReversion(60))
        return self
    def get_all_strategies(self) -> List[Strategy]:
        return list(self.strategies.values())

def adapt_strategies(strategies: List[Strategy], capital: float) -> List[Strategy]:
    return strategies

class DataManager:
    def __init__(self, cache_dir='./data_cache'):
        pass
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        sanity_check_market_data(df)
        return df

class BacktestResult:
    def __init__(self, returns: pd.Series, alpha_returns: Optional[pd.Series] = None, **kwargs):
        self.returns = returns
        self.alpha_returns = alpha_returns if alpha_returns is not None else returns

class BacktestEngine:
    def __init__(self, benchmark_symbol: str = 'SPY'):
        pass
    def run_backtest(self, strategy: Strategy, market_data: pd.DataFrame, initial_capital: float, benchmark_data: pd.DataFrame) -> BacktestResult:
        positions = strategy.generate_positions(market_data, initial_capital)
        returns = market_data['Close'].pct_change()
        strategy_returns = (positions.shift(1) / initial_capital) * returns
        return BacktestResult(returns=strategy_returns.fillna(0))

class BayesianELORanking:
    def __init__(self, mu: float = 1500, sigma: float = 350, **kwargs):
        self.mu = mu
        self.sigma = sigma
    def update(self, other_rating, outcome: float):
        if outcome == 1.0:
            self.mu += 5
        elif outcome == 0.0:
            self.mu -= 5
    def get_rating(self) -> float:
        return self.mu

class RankingManager:
    def __init__(self, ranking_class, **kwargs):
        self.ranking_class = ranking_class
        self.kwargs = kwargs
        self.rankings: Dict[str, Dict[Optional[int], Dict[str, BayesianELORanking]]] = {'alpha': {None: {}}}
    def update(self, strategy_a_name: str, strategy_b_name: str, outcome: float, metric: str, regime: Optional[int], timestamp: pd.Timestamp):
        if strategy_a_name not in self.rankings[metric][None]:
            self.rankings[metric][None][strategy_a_name] = self.ranking_class(**self.kwargs)
        if strategy_b_name not in self.rankings[metric][None]:
            self.rankings[metric][None][strategy_b_name] = self.ranking_class(**self.kwargs)
        
        self.rankings[metric][None][strategy_a_name].update(self.rankings[metric][None][strategy_b_name], outcome)
        self.rankings[metric][None][strategy_b_name].update(self.rankings[metric][None][strategy_a_name], 1 - outcome)
    def get_leaderboard(self, metric: str, regime: Optional[int]) -> pd.DataFrame:
        ranks = {name: ranker.get_rating() for name, ranker in self.rankings[metric][None].items()}
        if not ranks:
            return pd.DataFrame(columns=['strategy', 'rating'])
        return pd.DataFrame(list(ranks.items()), columns=['strategy', 'rating']).sort_values('rating', ascending=False).reset_index(drop=True)

class VolatilityRegimeDetector:
    def __init__(self, window: int, num_regimes: int, method: str): pass
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series: return pd.Series(np.random.randint(0, 2, len(data)), index=data.index)
    def expand_to_full_index(self, series: pd.Series, index: pd.Index, method: str) -> pd.Series: return series.reindex(index, method='ffill')

def sanity_check_market_data(df: pd.DataFrame, **kwargs):
    if df.empty:
        raise ValueError("Market data is empty.")

# --- End of Mock Classes ---

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
    def __init__(
        self,
        train_window_days: int = 504,
        test_days: int = 63,
        ranking_kwargs: Optional[Dict] = None
    ):
        self.train_window_days = train_window_days
        self.test_days = test_days
        self.ranking_kwargs = ranking_kwargs or {}
        self.data_manager = DataManager()
        self.backtest_engine = BacktestEngine()
        self.strategy_registry = StrategyRegistry().create_default_universe()
        self.results: List[WalkForwardResult] = []
    
    def run_walk_forward_validation(
        self,
        symbol: str = 'SPY',
        start_date: str = '1997-01-01',
        end_date: str = '2024-01-01',
        initial_capital: float = 100_000
    ) -> pd.DataFrame:
        print("="*100)
        print("WALK-FORWARD ELO VALIDATION - NO LOOK-AHEAD BIAS")
        print("="*100)
        
        self.adapted_strategies = adapt_strategies(
            list(self.strategy_registry.strategies.values()), 
            capital=initial_capital
        )
        
        print(f"\n[Step 1] Fetching market data for {symbol}...")
        market_data = self.data_manager.fetch_data(symbol=symbol, start_date=start_date, end_date=end_date)
        all_dates = market_data.index
        
        split_boundaries = self._calculate_rolling_split_boundaries(all_dates)
        print(f"\n[Step 2] Running {len(split_boundaries)} walk-forward splits...")
        
        for split_idx, (train_start_idx, train_end_idx, test_start_idx, test_end_idx) in enumerate(split_boundaries):
            print(f"\n=== SPLIT {split_idx + 1}/{len(split_boundaries)} ===")
            
            train_dates = all_dates[train_start_idx : train_end_idx + 1]
            test_dates = all_dates[test_start_idx : test_end_idx + 1]
            
            print(f"Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
            print(f"Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
            
            result = self._run_single_split(split_idx, train_dates, test_dates, market_data, market_data, initial_capital)
            self.results.append(result)
            
            print(f"Top-3 ELO strategies: {result.top_3_strategies}")
            print(f"Test returns - Top-3: {result.top_3_return:.2%}, Bottom-3: {result.bottom_3_return:.2%}")
            print(f"Spread: {result.spread:.2%}, ELO Accuracy: {result.elo_accuracy:.1%}")
        
        results_df = self._aggregate_results()
        compute_elo_predictive_power(results_df)
        return results_df
    
    def _calculate_rolling_split_boundaries(self, all_dates: pd.DatetimeIndex) -> List[Tuple[int, int, int, int]]:
        total_days = len(all_dates)
        step_size = self.test_days
        splits_possible = (total_days - self.train_window_days) // step_size
        
        if splits_possible < 2:
            raise ValueError(f"Not enough data for rolling window testing. Need at least {self.train_window_days + 2 * step_size} days")
        
        boundaries = []
        for i in range(splits_possible):
            train_start_idx = i * step_size
            train_end_idx = train_start_idx + self.train_window_days - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + self.test_days - 1, total_days - 1)
            
            if test_end_idx <= test_start_idx or train_end_idx >= total_days: break
            boundaries.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))
        
        return boundaries
    
    def _run_single_split(self, split_idx, train_dates, test_dates, market_data, benchmark_data, initial_capital) -> WalkForwardResult:
        train_market_data = market_data.loc[train_dates]
        
        print("  Training ELO on historical data...")
        regimes = self._detect_regimes_no_lookahead(train_market_data)
        train_results = self._run_backtests_on_period(train_market_data, benchmark_data.loc[train_dates], initial_capital)
        ranking_manager = self._train_elo_ranking(train_results, regimes, train_dates)
        
        train_leaderboard = ranking_manager.get_leaderboard('alpha', None)
        if train_leaderboard.empty: raise RuntimeError(f"No ELO rankings generated for split {split_idx}")
        
        top_3, bottom_3 = train_leaderboard.head(3)['strategy'].tolist(), train_leaderboard.tail(3)['strategy'].tolist()
        
        print("  Testing on future data...")
        test_market_data = market_data.loc[test_dates]
        test_results = self._run_backtests_on_period(test_market_data, benchmark_data.loc[test_dates], initial_capital)
        
        test_returns = {name: res.returns.mean() * 252 for name, res in test_results.items()}
        
        top_3_return = np.mean([test_returns.get(s, 0) for s in top_3])
        bottom_3_return = np.mean([test_returns.get(s, 0) for s in bottom_3])
        
        return WalkForwardResult(
            split_idx, train_dates[0], train_dates[-1], test_dates[0], test_dates[-1],
            top_3, bottom_3, top_3_return, bottom_3_return, np.mean(list(test_returns.values())),
            top_3_return - bottom_3_return, self._calculate_elo_accuracy(test_results, top_3, bottom_3, test_dates)
        )
    
    def _detect_regimes_no_lookahead(self, data: pd.DataFrame) -> pd.Series:
        try:
            vol_detector = VolatilityRegimeDetector(window=20, num_regimes=2, method='quantile')
            regimes = vol_detector.detect_regimes(data)
            return vol_detector.expand_to_full_index(regimes, data.index, method='ffill')
        except Exception as e:
            return pd.Series(0, index=data.index, name='regime')
    
    def _run_backtests_on_period(self, market_data, benchmark_data, initial_capital) -> Dict[str, BacktestResult]:
        results = {}
        for strategy in self.adapted_strategies:
            try:
                result = self.backtest_engine.run_backtest(strategy, market_data, initial_capital, benchmark_data)
                result.returns = result.returns.reindex(market_data.index).fillna(0)
                results[strategy.name] = result
            except Exception as e:
                results[strategy.name] = BacktestResult(returns=pd.Series(0, index=market_data.index))
        return results
    
    def _train_elo_ranking(self, results, regimes, dates) -> RankingManager:
        ranking_manager = RankingManager(BayesianELORanking, **self.ranking_kwargs)
        pairs = list(combinations(list(results.keys()), 2))
        
        for date in dates:
            regime_label = regimes.get(date)
            scores = {name: res.returns.get(date, 0.0) for name, res in results.items()}
            for s1, s2 in pairs:
                if np.isnan(scores[s1]) or np.isnan(scores[s2]): continue
                outcome = 0.5 if scores[s1] == scores[s2] else (1.0 if scores[s1] > scores[s2] else 0.0)
                ranking_manager.update(s1, s2, outcome, 'alpha', regime_label, date)
                ranking_manager.update(s1, s2, outcome, 'alpha', None, date)
        return ranking_manager
    
    def _calculate_elo_accuracy(self, test_results, top_3, bottom_3, test_dates) -> float:
        if not top_3 or not bottom_3: return 0.0
        wins = 0
        for date in test_dates:
            top_3_avg = np.mean([test_results[s].returns.get(date, 0) for s in top_3])
            bottom_3_avg = np.mean([test_results[s].returns.get(date, 0) for s in bottom_3])
            if top_3_avg > bottom_3_avg: wins += 1
        return wins / len(test_dates) if len(test_dates) > 0 else 0.0
    
    def _aggregate_results(self) -> pd.DataFrame:
        df = pd.DataFrame([r.__dict__ for r in self.results])
        print("\n" + "="*100)
        print("WALK-FORWARD ELO VALIDATION RESULTS")
        print("="*100)
        print(f"\nAverage Spread (Top-3 vs Bottom-3): {df['spread'].mean():.2%}")
        print(f"Average ELO Accuracy: {df['elo_accuracy'].mean():.1%}")
        print(f"Positive Spreads: {(df['spread'] > 0).sum()}/{len(df)} splits")
        all_top = [s for sublist in df['top_3_strategies'] for s in sublist]
        strategy_counts = pd.Series(all_top).value_counts()
        print("\nMost frequently in top-3:")
        for strategy, count in strategy_counts.head(5).items():
            print(f"  {strategy}: {count}/{len(df)} splits")
        return df

def compute_elo_predictive_power(validation_results: pd.DataFrame):
    spreads = validation_results['spread']
    if len(spreads) < 2: return {}
    
    mean_spread = spreads.mean()
    t_stat, p_value = stats.ttest_1samp(spreads, 0, alternative='greater')
    win_rate = (spreads > 0).mean()
    
    print("\n" + "="*80)
    print("ELO PREDICTIVE POWER TEST")
    print("="*80)
    print(f"\nMean spread (Top-3 vs Bottom-3): {mean_spread:.2%}")
    print(f"T-statistic: {t_stat:.2f}")
    print(f"P-value (one-sided): {p_value:.4f}")
    print(f"Win rate: {win_rate:.1%} ({(spreads > 0).sum()}/{len(spreads)} splits)")
    
    if p_value < 0.05 and mean_spread > 0:
        print("\n✓ RESULT: ELO HAS PREDICTIVE POWER")
    else:
        print("\n✗ RESULT: ELO DOES NOT PREDICT FUTURE PERFORMANCE")
    
    return {'mean_spread': mean_spread, 't_stat': t_stat, 'p_value': p_value, 'win_rate': win_rate}

def run_sensitivity_analysis():
    all_stats = {}
    training_windows = [252, 504, 756] # 1, 2, 3 years

    print("="*100)
    print("SENSITIVITY ANALYSIS: ELO PREDICTIVE POWER VS. TRAINING WINDOW")
    print("="*100)

    for window in training_windows:
        print(f"\n{'='*25} TESTING {window}-DAY TRAINING WINDOW {'='*25}")
        engine = WalkForwardELOEngine(train_window_days=window, test_days=63)
        results_df = engine.run_walk_forward_validation()
        
        if not results_df.empty:
            stats_results = compute_elo_predictive_power(results_df)
            all_stats[f'{window}d_window'] = stats_results

    print("\n" + "="*100)
    print("FINAL SENSITIVITY ANALYSIS SUMMARY")
    print("="*100)
    
    if all_stats:
        summary_df = pd.DataFrame(all_stats).T
        print(summary_df.to_string())
        summary_df.to_csv('sensitivity_analysis_results.csv')
        print("\nFull sensitivity analysis saved to sensitivity_analysis_results.csv")

if __name__ == "__main__":
    run_sensitivity_analysis()