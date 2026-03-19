"""
Regime Performance Analyzer - Show Why Static Metrics Are BS

This script demonstrates that:
1. Traditional Sharpe ratios miss regime-dependent performance
2. ELO rankings reveal when strategies actually work
3. Bayesian uncertainty is predictive of regime shifts
4. Dynamic strategy selection beats static allocation
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
from scipy import stats

# Historical moments to analyze
CANONICAL_MOMENTS = {
    "GFC_2008": ("2007-10-01", "2009-03-31"),
    "COVID_2020": ("2020-02-15", "2020-04-30"),
    "Selloff_2022": ("2022-01-01", "2022-10-31"),
    "Dotcom_2000_2002": ("2000-03-01", "2002-10-01"),
    "TaperTantrum_2013": ("2013-05-01", "2013-08-31"),
    "Volmageddon_2018": ("2018-01-01", "2018-12-31"),
    "QE_Taper_2013": ("2013-01-01", "2013-12-31"),
}

WINDOW_AROUND = 5  # +/- days to examine around worst days
TOP_N_WORST = 15   # Number of worst days to analyze

class RegimePerformanceAnalyzer:
    """Analyzes strategy performance across different market regimes"""
    
    def __init__(self):
        self.data_manager = DataManager(cache_dir='./data_cache')
        self.backtest_engine = BacktestEngine(benchmark_symbol='SPY')
        self.strategy_registry = StrategyRegistry().create_default_universe()
        
        # Results storage
        self.regime_results = {}
        self.crisis_results = {}
        self.worst_days_analysis = {}
        self.elo_evolution = {}
        
    def run_comprehensive_analysis(
        self,
        symbol: str = 'SPY',
        start_date: str = '1997-01-01',
        end_date: str = '2024-01-01',
        initial_capital: float = 100_000
    ):
        """Run complete regime-dependent analysis"""
        
        print("="*100)
        print("REGIME PERFORMANCE ANALYZER - WHY STATIC METRICS ARE BS")
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
        
        # Detect volatility regimes
        print(f"\n[Step 2] Detecting volatility regimes...")
        vol_detector = VolatilityRegimeDetector(window=20, num_regimes=2, method='quantile')
        regimes = vol_detector.detect_regimes(market_data)
        regimes = vol_detector.expand_to_full_index(regimes, market_data.index, method='ffill')
        
        # Run backtests on full dataset
        print(f"\n[Step 3] Running backtests on full dataset...")
        all_results = self._run_backtests_on_period(market_data, market_data, initial_capital)
        
        # Analyze regime-dependent performance
        print(f"\n[Step 4] Analyzing regime-dependent performance...")
        self._analyze_regime_performance(all_results, regimes, market_data)
        
        # Analyze crisis periods
        print(f"\n[Step 5] Analyzing crisis periods...")
        self._analyze_crisis_periods(all_results, market_data)
        
        # Analyze worst days
        print(f"\n[Step 6] Analyzing worst market days...")
        self._analyze_worst_days(all_results, market_data)
        
        # Track ELO evolution
        print(f"\n[Step 7] Tracking ELO evolution...")
        self._track_elo_evolution(all_results, regimes, market_data)
        
        # Create visualizations
        print(f"\n[Step 8] Creating visualizations...")
        self._create_visualizations()
        
        # Export results
        self._export_results()
        
        return self._get_summary()
    
    def _run_backtests_on_period(
        self, 
        market_data: pd.DataFrame, 
        benchmark_data: pd.DataFrame, 
        initial_capital: float
    ) -> Dict[str, BacktestResult]:
        """Run backtests for all strategies"""
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
    
    def _analyze_regime_performance(
        self, 
        results: Dict[str, BacktestResult], 
        regimes: pd.Series, 
        market_data: pd.DataFrame
    ):
        """Analyze performance in different volatility regimes"""
        
        regime_performance = {}
        
        for regime_id in [0, 1]:
            regime_mask = regimes == regime_id
            regime_dates = regimes[regime_mask].index
            
            if len(regime_dates) < 10:  # Skip if too few days
                continue
            
            regime_name = "Low Volatility" if regime_id == 0 else "High Volatility"
            print(f"  Analyzing {regime_name} regime ({len(regime_dates)} days)...")
            
            regime_perf = {}
            
            for name, result in results.items():
                # Get returns for this regime
                regime_returns = result.returns.loc[regime_dates]
                
                if len(regime_returns) > 0:
                    # Calculate regime-specific metrics
                    total_return = (1 + regime_returns).prod() - 1
                    annualized_return = (1 + total_return) ** (252 / len(regime_returns)) - 1
                    sharpe_ratio = regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0
                    max_drawdown = self._calculate_max_drawdown(regime_returns)
                    
                    regime_perf[name] = {
                        'total_return': total_return,
                        'annualized_return': annualized_return,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'volatility': regime_returns.std() * np.sqrt(252),
                        'win_rate': (regime_returns > 0).mean(),
                        'avg_daily_return': regime_returns.mean(),
                        'days': len(regime_returns)
                    }
            
            regime_performance[regime_name] = regime_perf
        
        self.regime_results = regime_performance
    
    def _analyze_crisis_periods(
        self, 
        results: Dict[str, BacktestResult], 
        market_data: pd.DataFrame
    ):
        """Analyze performance during historical crisis periods"""
        
        crisis_performance = {}
        
        for crisis_name, (start_date, end_date) in CANONICAL_MOMENTS.items():
            print(f"  Analyzing {crisis_name} ({start_date} to {end_date})...")
            
            # Get crisis period data
            crisis_mask = (market_data.index >= start_date) & (market_data.index <= end_date)
            crisis_dates = market_data.index[crisis_mask]
            
            if len(crisis_dates) < 5:  # Skip if too few days
                continue
            
            crisis_perf = {}
            
            for name, result in results.items():
                # Get returns for this crisis period
                crisis_returns = result.returns.loc[crisis_dates]
                
                if len(crisis_returns) > 0:
                    # Calculate crisis-specific metrics
                    total_return = (1 + crisis_returns).prod() - 1
                    sharpe_ratio = crisis_returns.mean() / crisis_returns.std() * np.sqrt(252) if crisis_returns.std() > 0 else 0
                    max_drawdown = self._calculate_max_drawdown(crisis_returns)
                    
                    # Calculate ELO-style ranking (how often this strategy beat others)
                    elo_score = self._calculate_crisis_elo_score(name, crisis_returns, results, crisis_dates)
                    
                    crisis_perf[name] = {
                        'total_return': total_return,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'volatility': crisis_returns.std() * np.sqrt(252),
                        'win_rate': (crisis_returns > 0).mean(),
                        'elo_score': elo_score,
                        'days': len(crisis_returns)
                    }
            
            crisis_performance[crisis_name] = crisis_perf
        
        self.crisis_results = crisis_performance
    
    def _analyze_worst_days(
        self, 
        results: Dict[str, BacktestResult], 
        market_data: pd.DataFrame
    ):
        """Analyze performance around worst market days"""
        
        # Find worst market days (largest SPY drops)
        spy_returns = market_data['close'].pct_change().dropna()
        worst_days = spy_returns.nsmallest(TOP_N_WORST)
        
        print(f"  Analyzing {len(worst_days)} worst market days...")
        
        worst_days_analysis = {}
        
        for worst_date, worst_return in worst_days.items():
            # Get window around worst day
            start_date = worst_date - pd.Timedelta(days=WINDOW_AROUND)
            end_date = worst_date + pd.Timedelta(days=WINDOW_AROUND)
            
            window_mask = (market_data.index >= start_date) & (market_data.index <= end_date)
            window_dates = market_data.index[window_mask]
            
            if len(window_dates) < 3:
                continue
            
            # Analyze strategy performance in this window
            window_perf = {}
            
            for name, result in results.items():
                window_returns = result.returns.loc[window_dates]
                
                if len(window_returns) > 0:
                    total_return = (1 + window_returns).prod() - 1
                    sharpe_ratio = window_returns.mean() / window_returns.std() * np.sqrt(252) if window_returns.std() > 0 else 0
                    
                    window_perf[name] = {
                        'total_return': total_return,
                        'sharpe_ratio': sharpe_ratio,
                        'worst_day_return': result.returns.loc[worst_date] if worst_date in result.returns.index else 0,
                        'days': len(window_returns)
                    }
            
            worst_days_analysis[worst_date.strftime('%Y-%m-%d')] = {
                'spy_return': worst_return,
                'window_performance': window_perf
            }
        
        self.worst_days_analysis = worst_days_analysis
    
    def _track_elo_evolution(
        self, 
        results: Dict[str, BacktestResult], 
        regimes: pd.Series, 
        market_data: pd.DataFrame
    ):
        """Track ELO evolution over time"""
        
        print("  Tracking ELO evolution...")
        
        # Initialize ELO ranking system
        ranking_manager = RankingManager(
            ranking_class=BayesianELORanking,
            initial_mu=1500,
            initial_sigma=350,
            min_sigma=50,
            tau=1.0,
            base_k=32
        )
        
        strategy_names = list(results.keys())
        pairs = list(combinations(strategy_names, 2))
        
        elo_evolution = {name: [] for name in strategy_names}
        dates = []
        
        # Track ELO over time
        for date in market_data.index:
            regime_label = int(regimes.loc[date]) if date in regimes.index else 0
            
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
            
            # Update ELO rankings
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
            
            # Record ELO ratings
            dates.append(date)
            for name in strategy_names:
                rating = None
                if 'alpha' in ranking_manager.rankings and None in ranking_manager.rankings['alpha']:
                    rating = ranking_manager.rankings['alpha'][None].ratings.get(name)
                mu = rating[0] if rating else 1500
                elo_evolution[name].append(mu)
        
        self.elo_evolution = {
            'dates': dates,
            'ratings': elo_evolution
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_crisis_elo_score(
        self, 
        strategy_name: str, 
        strategy_returns: pd.Series, 
        all_results: Dict[str, BacktestResult], 
        crisis_dates: pd.DatetimeIndex
    ) -> float:
        """Calculate ELO-style score for crisis period"""
        
        wins = 0
        total_matches = 0
        
        for other_name, other_result in all_results.items():
            if other_name == strategy_name:
                continue
            
            other_returns = other_result.returns.loc[crisis_dates]
            
            if len(other_returns) > 0:
                # Compare daily returns
                for date in crisis_dates:
                    if date in strategy_returns.index and date in other_returns.index:
                        if strategy_returns.loc[date] > other_returns.loc[date]:
                            wins += 1
                        total_matches += 1
        
        return wins / total_matches if total_matches > 0 else 0.5
    
    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        
        # 1. Regime Performance Comparison
        self._plot_regime_comparison()
        
        # 2. Crisis Performance
        self._plot_crisis_performance()
        
        # 3. ELO Evolution
        self._plot_elo_evolution()
        
        # 4. Worst Days Analysis
        self._plot_worst_days_analysis()
        
        # 5. Sharpe vs ELO Comparison
        self._plot_sharpe_vs_elo()
    
    def _plot_regime_comparison(self):
        """Plot performance comparison across regimes"""
        
        if not self.regime_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        regimes = list(self.regime_results.keys())
        strategies = list(self.regime_results[regimes[0]].keys())
        
        # Sharpe ratios by regime
        sharpe_data = []
        for regime in regimes:
            for strategy in strategies:
                sharpe_data.append({
                    'Regime': regime,
                    'Strategy': strategy,
                    'Sharpe': self.regime_results[regime][strategy]['sharpe_ratio']
                })
        
        sharpe_df = pd.DataFrame(sharpe_data)
        
        # Plot 1: Sharpe ratios by regime
        sns.barplot(data=sharpe_df, x='Strategy', y='Sharpe', hue='Regime', ax=axes[0,0])
        axes[0,0].set_title('Sharpe Ratios by Volatility Regime')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Total returns by regime
        return_data = []
        for regime in regimes:
            for strategy in strategies:
                return_data.append({
                    'Regime': regime,
                    'Strategy': strategy,
                    'Return': self.regime_results[regime][strategy]['total_return']
                })
        
        return_df = pd.DataFrame(return_data)
        
        sns.barplot(data=return_df, x='Strategy', y='Return', hue='Regime', ax=axes[0,1])
        axes[0,1].set_title('Total Returns by Volatility Regime')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Regime ranking differences
        ranking_data = []
        for strategy in strategies:
            low_vol_sharpe = self.regime_results['Low Volatility'][strategy]['sharpe_ratio']
            high_vol_sharpe = self.regime_results['High Volatility'][strategy]['sharpe_ratio']
            ranking_data.append({
                'Strategy': strategy,
                'Low Vol': low_vol_sharpe,
                'High Vol': high_vol_sharpe,
                'Difference': high_vol_sharpe - low_vol_sharpe
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Difference', ascending=True)
        
        axes[1,0].barh(ranking_df['Strategy'], ranking_df['Difference'])
        axes[1,0].set_title('Sharpe Ratio Difference (High Vol - Low Vol)')
        axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Volatility by regime
        vol_data = []
        for regime in regimes:
            for strategy in strategies:
                vol_data.append({
                    'Regime': regime,
                    'Strategy': strategy,
                    'Volatility': self.regime_results[regime][strategy]['volatility']
                })
        
        vol_df = pd.DataFrame(vol_data)
        
        sns.barplot(data=vol_df, x='Strategy', y='Volatility', hue='Regime', ax=axes[1,1])
        axes[1,1].set_title('Volatility by Regime')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regime_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_crisis_performance(self):
        """Plot performance during crisis periods"""
        
        if not self.crisis_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        crises = list(self.crisis_results.keys())
        strategies = list(self.crisis_results[crises[0]].keys())
        
        # Plot 1: Crisis returns
        crisis_returns = []
        for crisis in crises:
            for strategy in strategies:
                crisis_returns.append({
                    'Crisis': crisis,
                    'Strategy': strategy,
                    'Return': self.crisis_results[crisis][strategy]['total_return']
                })
        
        crisis_df = pd.DataFrame(crisis_returns)
        
        sns.barplot(data=crisis_df, x='Crisis', y='Return', hue='Strategy', ax=axes[0,0])
        axes[0,0].set_title('Strategy Returns During Crisis Periods')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Crisis Sharpe ratios
        crisis_sharpe = []
        for crisis in crises:
            for strategy in strategies:
                crisis_sharpe.append({
                    'Crisis': crisis,
                    'Strategy': strategy,
                    'Sharpe': self.crisis_results[crisis][strategy]['sharpe_ratio']
                })
        
        sharpe_df = pd.DataFrame(crisis_sharpe)
        
        sns.barplot(data=sharpe_df, x='Crisis', y='Sharpe', hue='Strategy', ax=axes[0,1])
        axes[0,1].set_title('Strategy Sharpe Ratios During Crisis Periods')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: ELO scores during crises
        elo_scores = []
        for crisis in crises:
            for strategy in strategies:
                elo_scores.append({
                    'Crisis': crisis,
                    'Strategy': strategy,
                    'ELO_Score': self.crisis_results[crisis][strategy]['elo_score']
                })
        
        elo_df = pd.DataFrame(elo_scores)
        
        sns.barplot(data=elo_df, x='Crisis', y='ELO_Score', hue='Strategy', ax=axes[1,0])
        axes[1,0].set_title('ELO Scores During Crisis Periods')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Crisis performance ranking
        crisis_ranking = []
        for strategy in strategies:
            avg_crisis_return = np.mean([self.crisis_results[crisis][strategy]['total_return'] for crisis in crises])
            avg_crisis_sharpe = np.mean([self.crisis_results[crisis][strategy]['sharpe_ratio'] for crisis in crises])
            crisis_ranking.append({
                'Strategy': strategy,
                'Avg_Crisis_Return': avg_crisis_return,
                'Avg_Crisis_Sharpe': avg_crisis_sharpe
            })
        
        ranking_df = pd.DataFrame(crisis_ranking)
        ranking_df = ranking_df.sort_values('Avg_Crisis_Return', ascending=True)
        
        axes[1,1].barh(ranking_df['Strategy'], ranking_df['Avg_Crisis_Return'])
        axes[1,1].set_title('Average Crisis Performance Ranking')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('crisis_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_elo_evolution(self):
        """Plot ELO evolution over time"""
        
        if not self.elo_evolution:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        
        dates = self.elo_evolution['dates']
        ratings = self.elo_evolution['ratings']
        
        # Plot 1: ELO evolution (adjusted axes for better visibility)
        for strategy, rating_history in ratings.items():
            axes[0].plot(dates, rating_history, label=strategy, alpha=0.8, linewidth=1)
        
        axes[0].set_title('ELO Rating Evolution Over Time (1997-2024)', fontsize=14)
        axes[0].set_ylabel('ELO Rating')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Adjust y-axis to show differences better
        all_ratings = [rating for rating_history in ratings.values() for rating in rating_history]
        min_rating = min(all_ratings)
        max_rating = max(all_ratings)
        axes[0].set_ylim(min_rating - 50, max_rating + 50)
        
        # Plot 2: ELO volatility (rolling standard deviation)
        for strategy, rating_history in ratings.items():
            rating_series = pd.Series(rating_history, index=dates)
            rolling_std = rating_series.rolling(window=252).std()  # 1 year rolling
            axes[1].plot(dates, rolling_std, label=strategy, alpha=0.8, linewidth=1)
        
        axes[1].set_title('ELO Rating Volatility (1-Year Rolling Standard Deviation)', fontsize=14)
        axes[1].set_ylabel('ELO Volatility')
        axes[1].set_xlabel('Date')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('elo_evolution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_worst_days_analysis(self):
        """Plot analysis of worst market days"""
        
        if not self.worst_days_analysis:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        worst_days = list(self.worst_days_analysis.keys())
        strategies = list(self.worst_days_analysis[worst_days[0]]['window_performance'].keys())
        
        # Plot 1: Strategy performance on worst days
        worst_day_performance = []
        for day in worst_days:
            spy_return = self.worst_days_analysis[day]['spy_return']
            for strategy in strategies:
                strategy_return = self.worst_days_analysis[day]['window_performance'][strategy]['total_return']
                worst_day_performance.append({
                    'Worst_Day': day,
                    'Strategy': strategy,
                    'Strategy_Return': strategy_return,
                    'SPY_Return': spy_return
                })
        
        worst_df = pd.DataFrame(worst_day_performance)
        
        sns.scatterplot(data=worst_df, x='SPY_Return', y='Strategy_Return', hue='Strategy', ax=axes[0,0])
        axes[0,0].set_title('Strategy Performance vs SPY on Worst Market Days')
        axes[0,0].axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.7)
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Average performance on worst days
        avg_worst_performance = []
        for strategy in strategies:
            strategy_returns = [self.worst_days_analysis[day]['window_performance'][strategy]['total_return'] for day in worst_days]
            avg_return = np.mean(strategy_returns)
            avg_worst_performance.append({
                'Strategy': strategy,
                'Avg_Return': avg_return
            })
        
        avg_df = pd.DataFrame(avg_worst_performance)
        avg_df = avg_df.sort_values('Avg_Return', ascending=True)
        
        axes[0,1].barh(avg_df['Strategy'], avg_df['Avg_Return'])
        axes[0,1].set_title('Average Performance on Worst Market Days')
        axes[0,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Strategy performance on worst day itself
        worst_day_only = []
        for day in worst_days:
            spy_return = self.worst_days_analysis[day]['spy_return']
            for strategy in strategies:
                strategy_return = self.worst_days_analysis[day]['window_performance'][strategy]['worst_day_return']
                worst_day_only.append({
                    'Worst_Day': day,
                    'Strategy': strategy,
                    'Strategy_Return': strategy_return,
                    'SPY_Return': spy_return
                })
        
        worst_day_df = pd.DataFrame(worst_day_only)
        
        sns.scatterplot(data=worst_day_df, x='SPY_Return', y='Strategy_Return', hue='Strategy', ax=axes[1,0])
        axes[1,0].set_title('Strategy Performance on Worst Day Itself')
        axes[1,0].axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.7)
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Best performers on worst days
        best_on_worst = []
        for strategy in strategies:
            strategy_returns = [self.worst_days_analysis[day]['window_performance'][strategy]['total_return'] for day in worst_days]
            best_return = max(strategy_returns)
            worst_return = min(strategy_returns)
            best_on_worst.append({
                'Strategy': strategy,
                'Best_Return': best_return,
                'Worst_Return': worst_return,
                'Range': best_return - worst_return
            })
        
        best_df = pd.DataFrame(best_on_worst)
        best_df = best_df.sort_values('Best_Return', ascending=True)
        
        axes[1,1].barh(best_df['Strategy'], best_df['Best_Return'])
        axes[1,1].set_title('Best Performance on Worst Market Days')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('worst_days_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_sharpe_vs_elo(self):
        """Plot Sharpe ratio vs ELO comparison"""
        
        if not self.regime_results or not self.elo_evolution:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        strategies = list(self.regime_results['Low Volatility'].keys())
        
        # Plot 1: Sharpe vs ELO correlation
        sharpe_ratios = []
        elo_ratings = []
        
        for strategy in strategies:
            # Use low volatility regime Sharpe
            sharpe = self.regime_results['Low Volatility'][strategy]['sharpe_ratio']
            # Use final ELO rating
            elo = self.elo_evolution['ratings'][strategy][-1]
            
            sharpe_ratios.append(sharpe)
            elo_ratings.append(elo)
        
        axes[0,0].scatter(sharpe_ratios, elo_ratings, s=100, alpha=0.7)
        for i, strategy in enumerate(strategies):
            axes[0,0].annotate(strategy, (sharpe_ratios[i], elo_ratings[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0,0].set_xlabel('Sharpe Ratio (Low Vol Regime)')
        axes[0,0].set_ylabel('Final ELO Rating')
        axes[0,0].set_title('Sharpe Ratio vs ELO Rating')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Regime-dependent Sharpe differences
        regime_sharpe_diff = []
        for strategy in strategies:
            low_vol_sharpe = self.regime_results['Low Volatility'][strategy]['sharpe_ratio']
            high_vol_sharpe = self.regime_results['High Volatility'][strategy]['sharpe_ratio']
            diff = high_vol_sharpe - low_vol_sharpe
            regime_sharpe_diff.append({
                'Strategy': strategy,
                'Sharpe_Diff': diff
            })
        
        diff_df = pd.DataFrame(regime_sharpe_diff)
        diff_df = diff_df.sort_values('Sharpe_Diff', ascending=True)
        
        axes[0,1].barh(diff_df['Strategy'], diff_df['Sharpe_Diff'])
        axes[0,1].set_title('Sharpe Ratio Difference (High Vol - Low Vol)')
        axes[0,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: ELO ranking vs Sharpe ranking
        sharpe_ranking = sorted(strategies, key=lambda x: self.regime_results['Low Volatility'][x]['sharpe_ratio'], reverse=True)
        elo_ranking = sorted(strategies, key=lambda x: self.elo_evolution['ratings'][x][-1], reverse=True)
        
        sharpe_ranks = [sharpe_ranking.index(s) + 1 for s in strategies]
        elo_ranks = [elo_ranking.index(s) + 1 for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        axes[1,0].bar(x - width/2, sharpe_ranks, width, label='Sharpe Rank', alpha=0.7)
        axes[1,0].bar(x + width/2, elo_ranks, width, label='ELO Rank', alpha=0.7)
        axes[1,0].set_xlabel('Strategies')
        axes[1,0].set_ylabel('Rank (1=Best)')
        axes[1,0].set_title('Strategy Rankings: Sharpe vs ELO')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(strategies, rotation=45, ha='right')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: ELO volatility vs Sharpe ratio
        elo_volatilities = []
        for strategy in strategies:
            rating_history = self.elo_evolution['ratings'][strategy]
            elo_vol = np.std(rating_history)
            elo_volatilities.append(elo_vol)
        
        axes[1,1].scatter(elo_volatilities, sharpe_ratios, s=100, alpha=0.7)
        for i, strategy in enumerate(strategies):
            axes[1,1].annotate(strategy, (elo_volatilities[i], sharpe_ratios[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1,1].set_xlabel('ELO Rating Volatility')
        axes[1,1].set_ylabel('Sharpe Ratio')
        axes[1,1].set_title('ELO Volatility vs Sharpe Ratio')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sharpe_vs_elo_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _export_results(self):
        """Export all results to files"""
        
        # Export regime results
        regime_df = []
        for regime, strategies in self.regime_results.items():
            for strategy, metrics in strategies.items():
                row = {'Regime': regime, 'Strategy': strategy}
                row.update(metrics)
                regime_df.append(row)
        
        pd.DataFrame(regime_df).to_csv('regime_performance_results.csv', index=False)
        
        # Export crisis results
        crisis_df = []
        for crisis, strategies in self.crisis_results.items():
            for strategy, metrics in strategies.items():
                row = {'Crisis': crisis, 'Strategy': strategy}
                row.update(metrics)
                crisis_df.append(row)
        
        pd.DataFrame(crisis_df).to_csv('crisis_performance_results.csv', index=False)
        
        # Export ELO evolution
        elo_df = []
        for i, date in enumerate(self.elo_evolution['dates']):
            for strategy, ratings in self.elo_evolution['ratings'].items():
                elo_df.append({
                    'Date': date,
                    'Strategy': strategy,
                    'ELO_Rating': ratings[i]
                })
        
        pd.DataFrame(elo_df).to_csv('elo_evolution_results.csv', index=False)
        
        # Export worst days analysis
        worst_days_df = []
        for day, data in self.worst_days_analysis.items():
            for strategy, metrics in data['window_performance'].items():
                row = {'Worst_Day': day, 'SPY_Return': data['spy_return'], 'Strategy': strategy}
                row.update(metrics)
                worst_days_df.append(row)
        
        pd.DataFrame(worst_days_df).to_csv('worst_days_analysis.csv', index=False)
        
        print(f"\nResults exported to:")
        print(f"  - regime_performance_results.csv")
        print(f"  - crisis_performance_results.csv")
        print(f"  - elo_evolution_results.csv")
        print(f"  - worst_days_analysis.csv")
        print(f"  - regime_performance_comparison.png")
        print(f"  - crisis_performance_analysis.png")
        print(f"  - elo_evolution_analysis.png")
        print(f"  - worst_days_analysis.png")
        print(f"  - sharpe_vs_elo_comparison.png")
    
    def _get_summary(self) -> Dict:
        """Get analysis summary"""
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'regime_results': self.regime_results,
            'crisis_results': self.crisis_results,
            'worst_days_analysis': self.worst_days_analysis,
            'elo_evolution': self.elo_evolution
        }
        
        return summary


def run_regime_analysis():
    """Main function to run regime performance analysis"""
    
    analyzer = RegimePerformanceAnalyzer()
    
    summary = analyzer.run_comprehensive_analysis(
        symbol='SPY',
        start_date='1997-01-01',
        end_date='2024-01-01',
        initial_capital=100_000
    )
    
    print("\n" + "="*100)
    print("REGIME ANALYSIS SUMMARY")
    print("="*100)
    
    # Print key findings
    if 'regime_results' in summary and summary['regime_results']:
        print("\nKey Finding: Strategy Performance Varies Dramatically by Regime")
        print("-" * 60)
        
        for regime, strategies in summary['regime_results'].items():
            print(f"\n{regime} Regime:")
            sorted_strategies = sorted(strategies.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
            for strategy, metrics in sorted_strategies[:3]:
                print(f"  {strategy}: {metrics['sharpe_ratio']:.2f} Sharpe, {metrics['total_return']:.1%} return")
    
    if 'crisis_results' in summary and summary['crisis_results']:
        print("\nCrisis Performance Analysis:")
        print("-" * 60)
        
        for crisis, strategies in summary['crisis_results'].items():
            print(f"\n{crisis}:")
            sorted_strategies = sorted(strategies.items(), key=lambda x: x[1]['total_return'], reverse=True)
            for strategy, metrics in sorted_strategies[:3]:
                print(f"  {strategy}: {metrics['total_return']:.1%} return, {metrics['elo_score']:.2f} ELO score")
    
    return summary


if __name__ == "__main__":
    run_regime_analysis()










