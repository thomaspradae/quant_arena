"""
ELO Full Evolution Tracker - Complete ELO Analysis 1997-2024

This script tracks ELO evolution across the ENTIRE dataset from 1997-2024,
not just training windows, to show complete strategy performance and ELO evolution.
"""

from data_manager import DataManager
from regime_detector import VolatilityRegimeDetector
from strategy_zoo import StrategyRegistry
from strategy_adapter import adapt_strategies
from backtest_engine import BacktestEngine, BacktestResult
from ranking import RankingManager, BayesianELORanking
from itertools import combinations
from utils import align_series_to_index, sanity_check_market_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class DailyELOData:
    """Daily ELO tracking data"""
    date: pd.Timestamp
    strategy: str
    mu: float
    sigma: float
    match_count: int
    wins: int
    losses: int
    draws: int
    daily_return: float
    cumulative_return: float
    regime: int

@dataclass
class MatchResult:
    """Individual match result"""
    date: pd.Timestamp
    strategy_a: str
    strategy_b: str
    outcome: float  # 1.0 = A wins, 0.0 = B wins, 0.5 = draw
    strategy_a_return: float
    strategy_b_return: float
    strategy_a_mu_before: float
    strategy_a_mu_after: float
    strategy_b_mu_before: float
    strategy_b_mu_after: float
    strategy_a_sigma_before: float
    strategy_a_sigma_after: float
    strategy_b_sigma_before: float
    strategy_b_sigma_after: float
    regime: int

class ELOFullEvolutionTracker:
    """Tracks ELO evolution across the ENTIRE dataset 1997-2024"""
    
    def __init__(
        self,
        ranking_kwargs: Optional[Dict] = None
    ):
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
        
        # Tracking data
        self.daily_elo_data: List[DailyELOData] = []
        self.match_results: List[MatchResult] = []
        self.strategy_performance: Dict[str, Dict] = {}
        
    def run_full_analysis(
        self,
        symbol: str = 'SPY',
        start_date: str = '1997-01-01',
        end_date: str = '2024-01-01',
        initial_capital: float = 100_000
    ) -> Dict:
        """Run complete ELO evolution analysis across ENTIRE dataset"""
        
        print("="*100)
        print("ELO FULL EVOLUTION TRACKER - COMPLETE 1997-2024 ANALYSIS")
        print("="*100)
        
        # Adapt strategies
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
        
        # Initialize strategy tracking
        strategy_names = [s.name for s in self.adapted_strategies]
        self._initialize_strategy_tracking(strategy_names, initial_capital)
        
        print(f"\n[Step 2] Running backtests on ENTIRE dataset...")
        
        # Run backtests on ENTIRE dataset
        all_results = self._run_backtests_on_period(
            market_data, benchmark_data, initial_capital
        )
        
        print(f"\n[Step 3] Detecting regimes across ENTIRE dataset...")
        
        # Detect regimes across ENTIRE dataset
        regimes = self._detect_regimes_no_lookahead(market_data)
        
        print(f"\n[Step 4] Training ELO across ENTIRE dataset with daily tracking...")
        
        # Train ELO with detailed tracking across ENTIRE dataset
        ranking_manager = self._train_elo_with_full_tracking(
            all_results, regimes, all_dates
        )
        
        # Calculate final performance metrics
        self._calculate_final_metrics()
        
        # Create visualizations
        self._create_visualizations()
        
        # Export data
        self._export_data()
        
        return self._get_summary()
    
    def _initialize_strategy_tracking(self, strategy_names: List[str], initial_capital: float):
        """Initialize tracking for all strategies"""
        for name in strategy_names:
            self.strategy_performance[name] = {
                'initial_capital': initial_capital,
                'current_capital': initial_capital,
                'total_return_pct': 0.0,
                'total_return_money': 0.0,
                'daily_returns': [],
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_matches': 0,
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'final_mu': 1500.0,
                'final_sigma': 350.0,
                'mu_evolution': [],
                'sigma_evolution': []
            }
    
    def _detect_regimes_no_lookahead(self, data: pd.DataFrame) -> pd.Series:
        """Detect regimes using only the provided data"""
        try:
            vol_detector = VolatilityRegimeDetector(window=20, num_regimes=2, method='quantile')
            regimes = vol_detector.detect_regimes(data)
            regimes = vol_detector.expand_to_full_index(regimes, data.index, method='ffill')
            return regimes
        except Exception as e:
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
                
                # Update strategy performance tracking
                daily_returns = result.returns
                self.strategy_performance[strategy.name]['daily_returns'] = daily_returns.tolist()
                
                # Calculate total return
                total_return = (1 + daily_returns).prod() - 1
                self.strategy_performance[strategy.name]['current_capital'] = initial_capital * (1 + total_return)
                self.strategy_performance[strategy.name]['total_return_pct'] = total_return
                self.strategy_performance[strategy.name]['total_return_money'] = (
                    self.strategy_performance[strategy.name]['current_capital'] - initial_capital
                )
                
                print(f"  {strategy.name}: {total_return:.1%} total return")
                
            except Exception as e:
                print(f"  {strategy.name}: Backtest failed - {e}")
                # Create dummy result
                dummy_returns = pd.Series(0, index=market_data.index)
                dummy_positions = pd.Series(0, index=market_data.index)
                dummy_trades = pd.DataFrame()
                dummy_equity = pd.Series(initial_capital, index=market_data.index)
                
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
    
    def _train_elo_with_full_tracking(
        self, 
        results: Dict[str, BacktestResult], 
        regimes: pd.Series, 
        dates: pd.DatetimeIndex
    ) -> RankingManager:
        """Train ELO ranking system with detailed tracking across ENTIRE dataset"""
        
        ranking_manager = RankingManager(
            ranking_class=BayesianELORanking,
            **self.ranking_kwargs
        )
        
        # Get strategy names
        strategy_names = list(results.keys())
        pairs = list(combinations(strategy_names, 2))
        
        print(f"  Tracking {len(dates)} days with {len(pairs)} strategy pairs per day...")
        
        # Update ELO with training data and track everything
        for i, date in enumerate(dates):
            if i % 1000 == 0:
                print(f"    Processing day {i+1}/{len(dates)}: {date.date()}")
            
            regime_label = None
            try:
                if date in regimes.index:
                    regime_label = int(regimes.loc[date])
            except Exception:
                regime_label = 0
            
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
            
            # Pairwise matches with detailed tracking
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
                
                # Get ratings before update
                s1_rating_before = None
                s2_rating_before = None
                if 'alpha' in ranking_manager.rankings and None in ranking_manager.rankings['alpha']:
                    s1_rating_before = ranking_manager.rankings['alpha'][None].ratings.get(s1)
                    s2_rating_before = ranking_manager.rankings['alpha'][None].ratings.get(s2)
                
                s1_mu_before = s1_rating_before[0] if s1_rating_before else 1500
                s1_sigma_before = s1_rating_before[1] if s1_rating_before else 350
                s2_mu_before = s2_rating_before[0] if s2_rating_before else 1500
                s2_sigma_before = s2_rating_before[1] if s2_rating_before else 350
                
                # Update ELO
                ranking_manager.update(
                    strategy_a_name=s1,
                    strategy_b_name=s2,
                    outcome=outcome,
                    metric='alpha',
                    regime=regime_label,
                    timestamp=pd.Timestamp(date)
                )
                
                # Get ratings after update
                s1_rating_after = None
                s2_rating_after = None
                if 'alpha' in ranking_manager.rankings and None in ranking_manager.rankings['alpha']:
                    s1_rating_after = ranking_manager.rankings['alpha'][None].ratings.get(s1)
                    s2_rating_after = ranking_manager.rankings['alpha'][None].ratings.get(s2)
                
                s1_mu_after = s1_rating_after[0] if s1_rating_after else 1500
                s1_sigma_after = s1_rating_after[1] if s1_rating_after else 350
                s2_mu_after = s2_rating_after[0] if s2_rating_after else 1500
                s2_sigma_after = s2_rating_after[1] if s2_rating_after else 350
                
                # Record match result
                match_result = MatchResult(
                    date=date,
                    strategy_a=s1,
                    strategy_b=s2,
                    outcome=outcome,
                    strategy_a_return=scores[s1],
                    strategy_b_return=scores[s2],
                    strategy_a_mu_before=s1_mu_before,
                    strategy_a_mu_after=s1_mu_after,
                    strategy_b_mu_before=s2_mu_before,
                    strategy_b_mu_after=s2_mu_after,
                    strategy_a_sigma_before=s1_sigma_before,
                    strategy_a_sigma_after=s1_sigma_after,
                    strategy_b_sigma_before=s2_sigma_before,
                    strategy_b_sigma_after=s2_sigma_after,
                    regime=regime_label
                )
                self.match_results.append(match_result)
                
                # Update strategy performance tracking
                self._update_strategy_match_stats(s1, outcome, s1_mu_after, s1_sigma_after)
                self._update_strategy_match_stats(s2, 1.0 - outcome, s2_mu_after, s2_sigma_after)
                
                # Also update global ranking
                ranking_manager.update(
                    strategy_a_name=s1,
                    strategy_b_name=s2,
                    outcome=outcome,
                    metric='alpha',
                    regime=None,
                    timestamp=pd.Timestamp(date)
                )
            
            # Record daily ELO data for all strategies
            for name in strategy_names:
                rating = None
                if 'alpha' in ranking_manager.rankings and None in ranking_manager.rankings['alpha']:
                    rating = ranking_manager.rankings['alpha'][None].ratings.get(name)
                mu = rating[0] if rating else 1500
                sigma = rating[1] if rating else 350
                
                daily_return = scores.get(name, 0.0)
                
                daily_data = DailyELOData(
                    date=date,
                    strategy=name,
                    mu=mu,
                    sigma=sigma,
                    match_count=self.strategy_performance[name]['total_matches'],
                    wins=self.strategy_performance[name]['wins'],
                    losses=self.strategy_performance[name]['losses'],
                    draws=self.strategy_performance[name]['draws'],
                    daily_return=daily_return,
                    cumulative_return=0.0,  # Will be calculated later
                    regime=regime_label
                )
                self.daily_elo_data.append(daily_data)
        
        return ranking_manager
    
    def _update_strategy_match_stats(self, strategy: str, outcome: float, mu: float, sigma: float):
        """Update strategy match statistics"""
        self.strategy_performance[strategy]['total_matches'] += 1
        self.strategy_performance[strategy]['final_mu'] = mu
        self.strategy_performance[strategy]['final_sigma'] = sigma
        
        if outcome > 0.5:
            self.strategy_performance[strategy]['wins'] += 1
        elif outcome < 0.5:
            self.strategy_performance[strategy]['losses'] += 1
        else:
            self.strategy_performance[strategy]['draws'] += 1
    
    def _calculate_final_metrics(self):
        """Calculate final performance metrics for all strategies"""
        
        for name, perf in self.strategy_performance.items():
            if perf['daily_returns']:
                returns = np.array(perf['daily_returns'])
                
                # Sharpe ratio
                if returns.std() > 0:
                    perf['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
                
                # Win rate
                if perf['total_matches'] > 0:
                    perf['win_rate'] = perf['wins'] / perf['total_matches']
                
                # Max drawdown
                cumulative = (1 + returns).cumprod()
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                perf['max_drawdown'] = drawdown.min()
    
    def _create_visualizations(self):
        """Create ELO evolution visualizations"""
        
        # Convert daily data to DataFrame
        daily_df = pd.DataFrame([
            {
                'date': d.date,
                'strategy': d.strategy,
                'mu': d.mu,
                'sigma': d.sigma,
                'daily_return': d.daily_return,
                'regime': d.regime
            }
            for d in self.daily_elo_data
        ])
        
        if daily_df.empty:
            print("No data to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(20, 15))
        
        # Plot 1: ELO Evolution (Mu) - Full Dataset
        for strategy in daily_df['strategy'].unique():
            strategy_data = daily_df[daily_df['strategy'] == strategy]
            axes[0].plot(strategy_data['date'], strategy_data['mu'], label=strategy, alpha=0.7, linewidth=1)
        
        axes[0].set_title('ELO Rating Evolution (Mu) - Complete Dataset 1997-2024', fontsize=14)
        axes[0].set_ylabel('ELO Rating (Mu)')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty Evolution (Sigma) - Full Dataset
        for strategy in daily_df['strategy'].unique():
            strategy_data = daily_df[daily_df['strategy'] == strategy]
            axes[1].plot(strategy_data['date'], strategy_data['sigma'], label=strategy, alpha=0.7, linewidth=1)
        
        axes[1].set_title('ELO Uncertainty Evolution (Sigma) - Complete Dataset 1997-2024', fontsize=14)
        axes[1].set_ylabel('ELO Uncertainty (Sigma)')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Regime Detection - Full Dataset
        regime_data = daily_df.groupby('date')['regime'].first()
        axes[2].plot(regime_data.index, regime_data.values, color='red', alpha=0.7, linewidth=1)
        axes[2].set_title('Market Regime Detection - Complete Dataset 1997-2024', fontsize=14)
        axes[2].set_ylabel('Regime (0=Low Vol, 1=High Vol)')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('elo_full_evolution_1997_2024.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create performance comparison plot
        self._create_performance_comparison_plot()
    
    def _create_performance_comparison_plot(self):
        """Create performance comparison visualization"""
        
        # Prepare data
        strategies = list(self.strategy_performance.keys())
        sharpe_ratios = [self.strategy_performance[s]['sharpe_ratio'] for s in strategies]
        total_returns = [self.strategy_performance[s]['total_return_pct'] for s in strategies]
        final_elos = [self.strategy_performance[s]['final_mu'] for s in strategies]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sharpe vs Total Return
        axes[0, 0].scatter(sharpe_ratios, total_returns, s=100, alpha=0.7)
        for i, strategy in enumerate(strategies):
            axes[0, 0].annotate(strategy, (sharpe_ratios[i], total_returns[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].set_ylabel('Total Return %')
        axes[0, 0].set_title('Sharpe Ratio vs Total Return (1997-2024)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Final ELO vs Total Return
        axes[0, 1].scatter(final_elos, total_returns, s=100, alpha=0.7, color='green')
        for i, strategy in enumerate(strategies):
            axes[0, 1].annotate(strategy, (final_elos[i], total_returns[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_xlabel('Final ELO Rating')
        axes[0, 1].set_ylabel('Total Return %')
        axes[0, 1].set_title('Final ELO vs Total Return (1997-2024)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe vs Final ELO
        axes[1, 0].scatter(sharpe_ratios, final_elos, s=100, alpha=0.7, color='orange')
        for i, strategy in enumerate(strategies):
            axes[1, 0].annotate(strategy, (sharpe_ratios[i], final_elos[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Sharpe Ratio')
        axes[1, 0].set_ylabel('Final ELO Rating')
        axes[1, 0].set_title('Sharpe Ratio vs Final ELO (1997-2024)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance ranking comparison
        sharpe_rank = [sorted(sharpe_ratios, reverse=True).index(x) + 1 for x in sharpe_ratios]
        elo_rank = [sorted(final_elos, reverse=True).index(x) + 1 for x in final_elos]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, sharpe_rank, width, label='Sharpe Rank', alpha=0.7)
        axes[1, 1].bar(x + width/2, elo_rank, width, label='ELO Rank', alpha=0.7)
        axes[1, 1].set_xlabel('Strategies')
        axes[1, 1].set_ylabel('Rank (1=Best)')
        axes[1, 1].set_title('Strategy Rankings: Sharpe vs ELO (1997-2024)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(strategies, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('strategy_performance_full_1997_2024.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _export_data(self):
        """Export all tracking data to files"""
        
        # Export daily ELO data
        daily_df = pd.DataFrame([
            {
                'date': d.date,
                'strategy': d.strategy,
                'mu': d.mu,
                'sigma': d.sigma,
                'match_count': d.match_count,
                'wins': d.wins,
                'losses': d.losses,
                'draws': d.draws,
                'daily_return': d.daily_return,
                'cumulative_return': d.cumulative_return,
                'regime': d.regime
            }
            for d in self.daily_elo_data
        ])
        daily_df.to_csv('elo_full_evolution_1997_2024.csv', index=False)
        
        # Export match results
        match_df = pd.DataFrame([
            {
                'date': m.date,
                'strategy_a': m.strategy_a,
                'strategy_b': m.strategy_b,
                'outcome': m.outcome,
                'strategy_a_return': m.strategy_a_return,
                'strategy_b_return': m.strategy_b_return,
                'strategy_a_mu_before': m.strategy_a_mu_before,
                'strategy_a_mu_after': m.strategy_a_mu_after,
                'strategy_b_mu_before': m.strategy_b_mu_before,
                'strategy_b_mu_after': m.strategy_b_mu_after,
                'strategy_a_sigma_before': m.strategy_a_sigma_before,
                'strategy_a_sigma_after': m.strategy_a_sigma_after,
                'strategy_b_sigma_before': m.strategy_b_sigma_before,
                'strategy_b_sigma_after': m.strategy_b_sigma_after,
                'regime': m.regime
            }
            for m in self.match_results
        ])
        match_df.to_csv('elo_match_results_full_1997_2024.csv', index=False)
        
        # Export strategy performance summary
        perf_df = pd.DataFrame([
            {
                'strategy': name,
                'initial_capital': perf['initial_capital'],
                'final_capital': perf['current_capital'],
                'total_return_pct': perf['total_return_pct'],
                'total_return_money': perf['total_return_money'],
                'sharpe_ratio': perf['sharpe_ratio'],
                'max_drawdown': perf['max_drawdown'],
                'win_rate': perf['win_rate'],
                'total_matches': perf['total_matches'],
                'wins': perf['wins'],
                'losses': perf['losses'],
                'draws': perf['draws'],
                'final_mu': perf['final_mu'],
                'final_sigma': perf['final_sigma']
            }
            for name, perf in self.strategy_performance.items()
        ])
        perf_df.to_csv('strategy_performance_full_1997_2024.csv', index=False)
        
        # Export summary JSON
        summary = self._get_summary()
        with open('elo_full_analysis_summary_1997_2024.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nData exported to:")
        print(f"  - elo_full_evolution_1997_2024.csv")
        print(f"  - elo_match_results_full_1997_2024.csv") 
        print(f"  - strategy_performance_full_1997_2024.csv")
        print(f"  - elo_full_analysis_summary_1997_2024.json")
        print(f"  - elo_full_evolution_1997_2024.png")
        print(f"  - strategy_performance_full_1997_2024.png")
    
    def _get_summary(self) -> Dict:
        """Get analysis summary"""
        
        # Calculate correlations
        strategies = list(self.strategy_performance.keys())
        sharpe_ratios = [self.strategy_performance[s]['sharpe_ratio'] for s in strategies]
        total_returns = [self.strategy_performance[s]['total_return_pct'] for s in strategies]
        final_elos = [self.strategy_performance[s]['final_mu'] for s in strategies]
        
        sharpe_elo_corr = np.corrcoef(sharpe_ratios, final_elos)[0, 1]
        sharpe_return_corr = np.corrcoef(sharpe_ratios, total_returns)[0, 1]
        elo_return_corr = np.corrcoef(final_elos, total_returns)[0, 1]
        
        return {
            'analysis_date': datetime.now().isoformat(),
            'total_days': len(self.daily_elo_data) // len(strategies) if strategies else 0,
            'total_matches': len(self.match_results),
            'strategies_analyzed': len(strategies),
            'correlations': {
                'sharpe_vs_elo': float(sharpe_elo_corr),
                'sharpe_vs_return': float(sharpe_return_corr),
                'elo_vs_return': float(elo_return_corr)
            },
            'top_performers': {
                'by_sharpe': sorted(strategies, key=lambda x: self.strategy_performance[x]['sharpe_ratio'], reverse=True)[:3],
                'by_return': sorted(strategies, key=lambda x: self.strategy_performance[x]['total_return_pct'], reverse=True)[:3],
                'by_elo': sorted(strategies, key=lambda x: self.strategy_performance[x]['final_mu'], reverse=True)[:3]
            },
            'strategy_performance': self.strategy_performance
        }


def run_elo_full_evolution_analysis():
    """Main function to run ELO full evolution analysis"""
    
    tracker = ELOFullEvolutionTracker()
    
    summary = tracker.run_full_analysis(
        symbol='SPY',
        start_date='1997-01-01',
        end_date='2024-01-01',
        initial_capital=100_000
    )
    
    print("\n" + "="*100)
    print("FULL ANALYSIS SUMMARY - 1997-2024")
    print("="*100)
    print(f"Total days analyzed: {summary['total_days']}")
    print(f"Total matches recorded: {summary['total_matches']}")
    print(f"Strategies analyzed: {summary['strategies_analyzed']}")
    
    print(f"\nCorrelations:")
    print(f"  Sharpe vs ELO: {summary['correlations']['sharpe_vs_elo']:.3f}")
    print(f"  Sharpe vs Return: {summary['correlations']['sharpe_vs_return']:.3f}")
    print(f"  ELO vs Return: {summary['correlations']['elo_vs_return']:.3f}")
    
    print(f"\nTop performers by Sharpe: {summary['top_performers']['by_sharpe']}")
    print(f"Top performers by Return: {summary['top_performers']['by_return']}")
    print(f"Top performers by ELO: {summary['top_performers']['by_elo']}")
    
    print(f"\nStrategy Performance (1997-2024):")
    for name, perf in summary['strategy_performance'].items():
        print(f"  {name}: {perf['total_return_pct']:.1%} return, {perf['sharpe_ratio']:.2f} Sharpe, {perf['final_mu']:.0f} ELO")
    
    return summary


if __name__ == "__main__":
    run_elo_full_evolution_analysis()










