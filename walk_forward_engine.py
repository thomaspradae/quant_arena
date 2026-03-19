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

from data_manager import DataManager
from regime_detector import HMMRegimeDetector, RegimeDetectorRegistry, VolatilityRegimeDetector, ChangePointRegimeDetector, MultifractalRegimeDetector
from strategy_zoo import StrategyRegistry, BuyAndHold, TrendFollowing, MeanReversion, LowVolatility, VolatilityBreakout, FadeExtremes, MomentumCrossover, RSIMeanReversion, RangeBreakout
from strategy_adapter import adapt_strategies
from backtest_engine import BacktestEngine, BacktestResult
from ranking import RankingManager, BayesianELORanking
from itertools import combinations
from utils import align_series_to_index, sanity_check_market_data

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardResult:
    """Results from a single walk-forward split"""
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
    elo_accuracy: float  # How often top-3 ELO strategies beat bottom-3 in test period


class WalkForwardELOEngine:
    """
    Walk-forward ELO testing engine that eliminates look-ahead bias.
    
    Key features:
    - Proper train/test splits
    - Regime detection using only past data
    - ELO training on historical periods only
    - Testing on future periods
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        min_train_days: int = 252,  # Minimum 1 year of training data
        test_days: int = 63,  # ~3 months test periods
        ranking_kwargs: Optional[Dict] = None
    ):
        """
        Args:
            n_splits: Number of walk-forward splits
            min_train_days: Minimum training period length
            test_days: Length of each test period
            ranking_kwargs: Arguments for BayesianELORanking
        """
        self.n_splits = n_splits
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.ranking_kwargs = ranking_kwargs or {
            'initial_mu': 1500,
            'initial_sigma': 350,
            'min_sigma': 50,
            'tau': 1.0,
            'base_k': 32
        }
        
        # Initialize components
        self.data_manager = DataManager(cache_dir='./data_cache')
        self.backtest_engine = BacktestEngine(benchmark_symbol='SPY')
        self.strategy_registry = StrategyRegistry().create_default_universe()
        
        # We'll adapt strategies later when we have the capital parameter
        
        # Results storage
        self.results: List[WalkForwardResult] = []
    
    def run_walk_forward_validation(
        self,
        symbol: str = 'SPY',
        start_date: str = '1997-01-01',
        end_date: str = '2024-01-01',
        initial_capital: float = 100_000
    ) -> pd.DataFrame:
        """
        Run complete walk-forward ELO validation.
        
        Returns:
            DataFrame with results from each split
        """
        print("="*100)
        print("WALK-FORWARD ELO VALIDATION - NO LOOK-AHEAD BIAS")
        print("="*100)
        
        # Adapt strategies for backtest engine compatibility
        self.adapted_strategies = adapt_strategies(
            list(self.strategy_registry.strategies.values()), 
            capital=initial_capital
        )
        
        # Fetch market data
        print(f"\n[Step 1] Fetching market data for {symbol}...")
        market_data = self.data_manager.fetch_data(
            symbol=symbol, 
            start_date=start_date, 
            end_date=end_date
        )
        
        sanity_check_market_data(market_data, require_cols=['close', 'volume'], nan_threshold=0.05)
        benchmark_data = market_data.copy()
        
        # Get all available dates
        all_dates = market_data.index
        total_days = len(all_dates)
        
        print(f"Total trading days: {total_days}")
        print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
        
        # Calculate split boundaries using rolling windows to cover full dataset
        split_boundaries = self._calculate_rolling_split_boundaries(all_dates)
        
        print(f"\n[Step 2] Running {len(split_boundaries)} walk-forward splits...")
        
        for split_idx, (train_end_idx, test_start_idx, test_end_idx) in enumerate(split_boundaries):
            print(f"\n=== SPLIT {split_idx + 1}/{len(split_boundaries)} ===")
            
            # Define periods (rolling window)
            train_start_idx = split_idx * self.test_days
            train_dates = all_dates[train_start_idx:train_end_idx + 1]
            test_dates = all_dates[test_start_idx:test_end_idx + 1]
            
            train_start = train_dates[0]
            train_end = train_dates[-1]
            test_start = test_dates[0]
            test_end = test_dates[-1]
            
            print(f"Train: {train_start.date()} to {train_end.date()} ({len(train_dates)} days)")
            print(f"Test:  {test_start.date()} to {test_end.date()} ({len(test_dates)} days)")
            
            # Run this split
            result = self._run_single_split(
                split_idx=split_idx,
                train_dates=train_dates,
                test_dates=test_dates,
                market_data=market_data,
                benchmark_data=benchmark_data,
                initial_capital=initial_capital
            )
            
            self.results.append(result)
            
            # Print results
            print(f"Top-3 ELO strategies: {result.top_3_strategies}")
            print(f"Test returns - Top-3: {result.top_3_return:.2%}, Bottom-3: {result.bottom_3_return:.2%}")
            print(f"Spread: {result.spread:.2%}, ELO Accuracy: {result.elo_accuracy:.1%}")
        
        # Aggregate results
        return self._aggregate_results()
    
    def _calculate_split_boundaries(self, all_dates: pd.DatetimeIndex) -> List[Tuple[int, int, int]]:
        """Calculate train/test boundaries for walk-forward splits"""
        total_days = len(all_dates)
        
        # Use 2 years minimum training
        min_train_size = max(self.min_train_days, 2 * 252)  # 2 years minimum training
        
        # Calculate how many splits we can fit in the remaining data
        remaining_days = total_days - min_train_size
        splits_possible = min(self.n_splits, remaining_days // self.test_days)
        
        if splits_possible < 2:
            raise ValueError(f"Not enough data for walk-forward testing. Need at least {min_train_size + 2 * self.test_days} days")
        
        boundaries = []
        for i in range(splits_possible):
            # Each split uses more training data (expanding window)
            train_end_idx = min_train_size + i * self.test_days - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + self.test_days - 1, total_days - 1)
            
            if test_end_idx <= test_start_idx:
                break  # No more valid splits
                
            boundaries.append((train_end_idx, test_start_idx, test_end_idx))
        
        return boundaries
    
    def _calculate_rolling_split_boundaries(self, all_dates: pd.DatetimeIndex) -> List[Tuple[int, int, int]]:
        """Calculate train/test boundaries using rolling windows to cover full dataset"""
        total_days = len(all_dates)
        
        # Use 2 years training window
        train_window = 2 * 252  # 2 years
        
        # Calculate how many splits we can fit
        step_size = self.test_days  # Step forward by test period
        splits_possible = (total_days - train_window) // step_size
        
        if splits_possible < 2:
            raise ValueError(f"Not enough data for rolling window testing. Need at least {train_window + 2 * step_size} days")
        
        boundaries = []
        for i in range(splits_possible):
            # Rolling window: each split uses a fixed training window
            train_start_idx = i * step_size
            train_end_idx = train_start_idx + train_window - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + self.test_days - 1, total_days - 1)
            
            if test_end_idx <= test_start_idx or train_end_idx >= total_days:
                break  # No more valid splits
                
            boundaries.append((train_end_idx, test_start_idx, test_end_idx))
        
        return boundaries
    
    def _run_single_split(
        self,
        split_idx: int,
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
        market_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        initial_capital: float
    ) -> WalkForwardResult:
        """Run a single walk-forward split"""
        
        # Step 1: Train ELO on training period only
        print("  Training ELO on historical data...")
        
        # Get training data
        train_market_data = market_data.loc[train_dates]
        train_benchmark_data = benchmark_data.loc[train_dates]
        
        # Detect regimes using ONLY training data (no look-ahead)
        regimes = self._detect_regimes_no_lookahead(train_market_data)
        
        # Run backtests on training period only
        train_results = self._run_backtests_on_period(
            train_market_data, train_benchmark_data, initial_capital
        )
        
        # Train ELO ranking system
        ranking_manager = self._train_elo_ranking(
            train_results, regimes, train_dates
        )
        
        # Get final ELO rankings
        train_leaderboard = ranking_manager.get_leaderboard('alpha', None)
        
        if train_leaderboard.empty:
            raise RuntimeError(f"No ELO rankings generated for split {split_idx}")
        
        # Identify top and bottom strategies
        top_3 = train_leaderboard.head(3)['strategy'].tolist()
        bottom_3 = train_leaderboard.tail(3)['strategy'].tolist()
        
        # Step 2: Test on future period
        print("  Testing on future data...")
        
        # Get test data
        test_market_data = market_data.loc[test_dates]
        test_benchmark_data = benchmark_data.loc[test_dates]
        
        # Run backtests on test period
        test_results = self._run_backtests_on_period(
            test_market_data, test_benchmark_data, initial_capital
        )
        
        # Calculate test period returns
        test_returns = {}
        for name, result in test_results.items():
            if getattr(result, 'alpha_returns', None) is not None:
                test_returns[name] = result.alpha_returns.mean() * 252  # Annualized
            else:
                test_returns[name] = result.returns.mean() * 252
        
        # Calculate performance metrics
        top_3_return = np.mean([test_returns.get(s, 0) for s in top_3])
        bottom_3_return = np.mean([test_returns.get(s, 0) for s in bottom_3])
        all_return = np.mean(list(test_returns.values()))
        spread = top_3_return - bottom_3_return
        
        # Calculate ELO accuracy (how often top-3 beat bottom-3)
        elo_accuracy = self._calculate_elo_accuracy(
            test_results, top_3, bottom_3, test_dates
        )
        
        return WalkForwardResult(
            split_idx=split_idx,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
            top_3_strategies=top_3,
            bottom_3_strategies=bottom_3,
            top_3_return=top_3_return,
            bottom_3_return=bottom_3_return,
            all_return=all_return,
            spread=spread,
            elo_accuracy=elo_accuracy
        )
    
    def _detect_regimes_no_lookahead(self, data: pd.DataFrame) -> pd.Series:
        """Detect regimes using only the provided data (no future information)"""
        try:
            # Use simple volatility-based regime detection for robustness
            vol_detector = VolatilityRegimeDetector(window=20, num_regimes=2, method='quantile')
            regimes = vol_detector.detect_regimes(data)
            regimes = vol_detector.expand_to_full_index(regimes, data.index, method='ffill')
            return regimes
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}. Using single regime.")
            return pd.Series(index=data.index, data=[0] * len(data), name='regime')
    
    def _run_backtests_on_period(
        self, 
        market_data: pd.DataFrame, 
        benchmark_data: pd.DataFrame, 
        initial_capital: float
    ) -> Dict[str, BacktestResult]:
        """Run backtests for all strategies on a specific period"""
        results = {}
        
        for strategy in self.adapted_strategies:
            try:
                result = self.backtest_engine.run_backtest(
                    strategy,
                    market_data,
                    initial_capital=initial_capital,
                    benchmark_data=benchmark_data
                )
                
                # Align returns to market data index
                result.returns = result.returns.reindex(market_data.index).fillna(0)
                if getattr(result, 'alpha_returns', None) is not None:
                    result.alpha_returns = result.alpha_returns.reindex(market_data.index).fillna(0)
                
                results[strategy.name] = result
                
            except Exception as e:
                logger.warning(f"Backtest failed for {strategy.name}: {e}")
                # Create dummy result
                dummy_returns = pd.Series(0, index=market_data.index)
                dummy_positions = pd.Series(0, index=market_data.index)
                dummy_trades = pd.DataFrame()
                dummy_equity = pd.Series(100000, index=market_data.index)
                
                results[strategy.name] = BacktestResult(
                    returns=dummy_returns,
                    positions=dummy_positions,
                    trades=dummy_trades,
                    equity_curve=dummy_equity,
                    metrics={},
                    alpha_returns=dummy_returns,
                    beta=0.0,
                    tail_metrics={}
                )
        
        return results
    
    def _train_elo_ranking(
        self, 
        results: Dict[str, BacktestResult], 
        regimes: pd.Series, 
        dates: pd.DatetimeIndex
    ) -> RankingManager:
        """Train ELO ranking system on historical data only"""
        
        ranking_manager = RankingManager(
            ranking_class=BayesianELORanking,
            **self.ranking_kwargs
        )
        
        # Get strategy names
        strategy_names = list(results.keys())
        pairs = list(combinations(strategy_names, 2))
        
        # Update ELO with training data
        for date in dates:
            regime_label = None
            try:
                if date in regimes.index:
                    regime_label = int(regimes.loc[date])
            except Exception:
                regime_label = None
            
            # Get returns for this date
            scores = {}
            for name, result in results.items():
                score = np.nan
                if getattr(result, 'alpha_returns', None) is not None:
                    try:
                        score = float(result.alpha_returns.loc[date])
                    except Exception:
                        score = np.nan
                if np.isnan(score):
                    try:
                        score = float(result.returns.loc[date])
                    except Exception:
                        score = np.nan
                scores[name] = score
            
            # Pairwise matches
            for s1, s2 in pairs:
                if np.isnan(scores[s1]) or np.isnan(scores[s2]):
                    continue
                
                eps = 1e-12
                if scores[s1] > scores[s2] + eps:
                    outcome = 1.0
                elif scores[s2] > scores[s1] + eps:
                    outcome = 0.0
                else:
                    outcome = 0.5
                
                ranking_manager.update(
                    strategy_a_name=s1,
                    strategy_b_name=s2,
                    outcome=outcome,
                    metric='alpha',
                    regime=regime_label,
                    timestamp=pd.Timestamp(date)
                )
                
                # Also update global ranking
                ranking_manager.update(
                    strategy_a_name=s1,
                    strategy_b_name=s2,
                    outcome=outcome,
                    metric='alpha',
                    regime=None,
                    timestamp=pd.Timestamp(date)
                )
        
        return ranking_manager
    
    def _calculate_elo_accuracy(
        self, 
        test_results: Dict[str, BacktestResult], 
        top_3: List[str], 
        bottom_3: List[str], 
        test_dates: pd.DatetimeIndex
    ) -> float:
        """Calculate how often top-3 ELO strategies beat bottom-3 in test period"""
        
        if not top_3 or not bottom_3:
            return 0.0
        
        wins = 0
        total_comparisons = 0
        
        for date in test_dates:
            # Get daily returns
            daily_returns = {}
            for name, result in test_results.items():
                if getattr(result, 'alpha_returns', None) is not None:
                    try:
                        daily_returns[name] = float(result.alpha_returns.loc[date])
                    except Exception:
                        daily_returns[name] = 0.0
                else:
                    try:
                        daily_returns[name] = float(result.returns.loc[date])
                    except Exception:
                        daily_returns[name] = 0.0
            
            # Compare top-3 vs bottom-3
            top_3_avg = np.mean([daily_returns.get(s, 0) for s in top_3])
            bottom_3_avg = np.mean([daily_returns.get(s, 0) for s in bottom_3])
            
            if top_3_avg > bottom_3_avg:
                wins += 1
            total_comparisons += 1
        
        return wins / total_comparisons if total_comparisons > 0 else 0.0
    
    def _aggregate_results(self) -> pd.DataFrame:
        """Aggregate results from all splits"""
        if not self.results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        results_data = []
        for result in self.results:
            results_data.append({
                'split': result.split_idx + 1,
                'train_start': result.train_start,
                'train_end': result.train_end,
                'test_start': result.test_start,
                'test_end': result.test_end,
                'train_days': (result.train_end - result.train_start).days,
                'test_days': (result.test_end - result.test_start).days,
                'top_3_strategies': ', '.join(result.top_3_strategies),
                'bottom_3_strategies': ', '.join(result.bottom_3_strategies),
                'top_3_return': result.top_3_return,
                'bottom_3_return': result.bottom_3_return,
                'all_return': result.all_return,
                'spread': result.spread,
                'elo_accuracy': result.elo_accuracy
            })
        
        df = pd.DataFrame(results_data)
        
        # Print summary
        print("\n" + "="*100)
        print("WALK-FORWARD ELO VALIDATION RESULTS")
        print("="*100)
        
        print(f"\nAverage Spread (Top-3 vs Bottom-3): {df['spread'].mean():.2%}")
        print(f"Average ELO Accuracy: {df['elo_accuracy'].mean():.1%}")
        print(f"Positive Spreads: {(df['spread'] > 0).sum()}/{len(df)} splits")
        print(f"ELO Accuracy > 50%: {(df['elo_accuracy'] > 0.5).sum()}/{len(df)} splits")
        
        # Strategy frequency analysis
        all_top_strategies = []
        for strategies_str in df['top_3_strategies']:
            all_top_strategies.extend(strategies_str.split(', '))
        
        strategy_counts = pd.Series(all_top_strategies).value_counts()
        print(f"\nMost frequently in top-3:")
        for strategy, count in strategy_counts.head(5).items():
            print(f"  {strategy}: {count}/{len(df)} splits")
        
        return df


def compute_elo_predictive_power(validation_results):
    """
    Does ELO ranking predict future performance?
    """
    spreads = validation_results['spread']
    
    # Test if spread is positive on average
    mean_spread = spreads.mean()
    std_spread = spreads.std()
    t_stat = mean_spread / (std_spread / np.sqrt(len(spreads)))
    
    # One-sided t-test (H1: spread > 0)
    p_value = 1 - stats.t.cdf(t_stat, df=len(spreads)-1)
    
    # Win rate (% of splits where top-3 beat bottom-3)
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
        print(f"  Top-3 ELO strategies outperform by {mean_spread:.2%} on average (p={p_value:.3f})")
    else:
        print("\n✗ RESULT: ELO DOES NOT PREDICT FUTURE PERFORMANCE")
    
    return {
        'mean_spread': mean_spread,
        't_stat': t_stat,
        'p_value': p_value,
        'win_rate': win_rate
    }


def run_walk_forward_validation():
    """Main function to run walk-forward ELO validation"""
    
    # Initialize engine with more splits to use full dataset
    engine = WalkForwardELOEngine(
        n_splits=50,  # Start with 50 splits for testing
        min_train_days=252,  # 1 year minimum training
        test_days=21,  # ~1 month test periods (shorter to fit more splits)
    )
    
    # Run validation
    results_df = engine.run_walk_forward_validation(
        symbol='SPY',
        start_date='1997-01-01',
        end_date='2024-01-01',
        initial_capital=100_000
    )
    
    # Run statistical significance test
    stats_results = compute_elo_predictive_power(results_df)
    
    # Save results
    results_df.to_csv('walk_forward_elo_results.csv', index=False)
    print(f"\nResults saved to walk_forward_elo_results.csv")
    
    # Save statistical results
    import json
    with open('elo_predictive_power_stats.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    print(f"Statistical results saved to elo_predictive_power_stats.json")
    
    return results_df, stats_results


if __name__ == "__main__":
    run_walk_forward_validation()
