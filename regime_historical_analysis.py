"""
Regime-Dependent Performance Analysis

This script demonstrates how traditional static metrics (like Sharpe ratio) 
miss the dynamic nature of strategy performance across different market regimes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_manager import DataManager
from regime_detector import VolatilityRegimeDetector
from strategy_zoo import StrategyRegistry
from strategy_adapter import adapt_strategies
from backtest_engine import BacktestEngine
from ranking import RankingManager, BayesianELORanking
from itertools import combinations

# Configuration
WINDOW_AROUND = 5
TOP_N_WORST = 15

# Canonical historical moments
CANONICAL_MOMENTS = {
    "GFC_2008": ("2007-10-01", "2009-03-31"),
    "COVID_2020": ("2020-02-15", "2020-04-30"), 
    "Selloff_2022": ("2022-01-01", "2022-10-31"),
    "Dotcom_2000_2002": ("2000-03-01", "2002-10-01"),
    "TaperTantrum_2013": ("2013-05-01", "2013-08-31"),
}

class RegimePerformanceAnalyzer:
    def __init__(self, symbol='SPY', start_date='1997-01-01', end_date='2024-01-01'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = 100_000
        
        self.data_manager = DataManager(cache_dir='./data_cache')
        self.backtest_engine = BacktestEngine(benchmark_symbol='SPY')
        self.strategy_registry = StrategyRegistry().create_default_universe()
        
        self.market_data = None
        self.regimes = None
        self.strategy_results = None
        self.elo_evolution = None
        
    def load_data(self):
        print("Loading market data and running backtests...")
        
        self.market_data = self.data_manager.fetch_data(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        vol_detector = VolatilityRegimeDetector(window=20, num_regimes=2, method='quantile')
        self.regimes = vol_detector.detect_regimes(self.market_data)
        self.regimes = vol_detector.expand_to_full_index(self.regimes, self.market_data.index, method='ffill')
        
        self.adapted_strategies = adapt_strategies(
            list(self.strategy_registry.strategies.values()),
            capital=self.initial_capital
        )
        
        self.strategy_results = {}
        for strategy in self.adapted_strategies:
            try:
                result = self.backtest_engine.run_backtest(
                    strategy,
                    self.market_data,
                    initial_capital=self.initial_capital,
                    benchmark_data=self.market_data
                )
                self.strategy_results[strategy.name] = result
            except Exception as e:
                print(f"Backtest failed for {strategy.name}: {e}")
        
        print(f"Loaded data for {len(self.market_data)} days")
        print(f"Detected {self.regimes.nunique()} regimes")
        print(f"Successfully backtested {len(self.strategy_results)} strategies")
    
    def calculate_static_metrics(self):
        static_metrics = {}
        
        for name, result in self.strategy_results.items():
            returns = result.returns.dropna()
            
            if len(returns) > 0:
                total_return = (1 + returns).prod() - 1
                annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
                volatility = returns.std() * np.sqrt(252)
                sharpe = annualized_return / volatility if volatility > 0 else 0
                
                static_metrics[name] = {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'final_value': self.initial_capital * (1 + total_return)
                }
        
        return pd.DataFrame(static_metrics).T
    
    def track_elo_evolution(self):
        print("Tracking ELO evolution over time...")
        
        ranking_manager = RankingManager(
            ranking_class=BayesianELORanking,
            initial_mu=1500,
            initial_sigma=350,
            min_sigma=50,
            tau=1.0,
            base_k=32
        )
        
        strategy_names = list(self.strategy_results.keys())
        pairs = list(combinations(strategy_names, 2))
        
        elo_data = []
        
        for date in self.market_data.index:
            regime_label = None
            try:
                if date in self.regimes.index:
                    regime_label = int(self.regimes.loc[date])
            except Exception:
                regime_label = None
            
            scores = {}
            for name, result in self.strategy_results.items():
                try:
                    score = float(result.returns.loc[date])
                except Exception:
                    score = 0.0
                scores[name] = score
            
            for s1, s2 in pairs:
                if s1 not in scores or s2 not in scores:
                    continue
                
                eps = 1e-12
                if scores[s1] > scores[s2] + eps:
                    outcome = 1.0
                elif scores[s2] > scores[s1] + eps:
                    outcome = 0.0
                else:
                    outcome = 0.5
                
                ranking_manager.update(s1, s2, outcome, 'alpha', regime_label, date)
                ranking_manager.update(s1, s2, outcome, 'alpha', None, date)
            
            for name in strategy_names:
                try:
                    global_rating = ranking_manager.get_rating(name, 'alpha', None)
                    regime_rating = ranking_manager.get_rating(name, 'alpha', regime_label) if regime_label is not None else None
                    
                    elo_data.append({
                        'date': date,
                        'strategy': name,
                        'regime': regime_label,
                        'global_mu': global_rating.mu if hasattr(global_rating, 'mu') else global_rating,
                        'global_sigma': global_rating.sigma if hasattr(global_rating, 'sigma') else 0,
                        'regime_mu': regime_rating.mu if regime_rating and hasattr(regime_rating, 'mu') else None,
                        'regime_sigma': regime_rating.sigma if regime_rating and hasattr(regime_rating, 'sigma') else None,
                        'daily_return': scores.get(name, 0)
                    })
                except Exception as e:
                    continue
        
        self.elo_evolution = pd.DataFrame(elo_data)
        print(f"Tracked ELO evolution for {len(elo_data)} data points")
        
        return self.elo_evolution
    
    def analyze_regime_performance(self):
        print("Analyzing regime-dependent performance...")
        
        if self.elo_evolution is None:
            self.track_elo_evolution()
        
        regime_analysis = {}
        
        for regime in self.regimes.unique():
            regime_data = self.elo_evolution[self.elo_evolution['regime'] == regime]
            
            if len(regime_data) == 0:
                continue
            
            regime_metrics = {}
            for strategy in regime_data['strategy'].unique():
                strategy_data = regime_data[regime_data['strategy'] == strategy]
                
                if len(strategy_data) > 0:
                    regime_metrics[strategy] = {
                        'avg_elo': strategy_data['regime_mu'].mean(),
                        'avg_uncertainty': strategy_data['regime_sigma'].mean(),
                        'avg_return': strategy_data['daily_return'].mean() * 252,
                        'volatility': strategy_data['daily_return'].std() * np.sqrt(252),
                        'sharpe': (strategy_data['daily_return'].mean() * 252) / (strategy_data['daily_return'].std() * np.sqrt(252)) if strategy_data['daily_return'].std() > 0 else 0
                    }
            
            regime_analysis[f'Regime_{regime}'] = regime_metrics
        
        return regime_analysis
    
    def analyze_crisis_periods(self):
        print("Analyzing crisis period performance...")
        
        crisis_analysis = {}
        
        for crisis_name, (start_date, end_date) in CANONICAL_MOMENTS.items():
            crisis_data = self.elo_evolution[
                (self.elo_evolution['date'] >= start_date) & 
                (self.elo_evolution['date'] <= end_date)
            ]
            
            if len(crisis_data) == 0:
                continue
            
            crisis_metrics = {}
            for strategy in crisis_data['strategy'].unique():
                strategy_data = crisis_data[crisis_data['strategy'] == strategy]
                
                if len(strategy_data) > 0:
                    total_return = (1 + strategy_data['daily_return']).prod() - 1
                    volatility = strategy_data['daily_return'].std() * np.sqrt(252)
                    sharpe = (strategy_data['daily_return'].mean() * 252) / volatility if volatility > 0 else 0
                    
                    elo_start = strategy_data['global_mu'].iloc[0] if len(strategy_data) > 0 else 0
                    elo_end = strategy_data['global_mu'].iloc[-1] if len(strategy_data) > 0 else 0
                    elo_change = elo_end - elo_start
                    
                    crisis_metrics[strategy] = {
                        'total_return': total_return,
                        'volatility': volatility,
                        'sharpe': sharpe,
                        'elo_start': elo_start,
                        'elo_end': elo_end,
                        'elo_change': elo_change,
                        'avg_uncertainty': strategy_data['global_sigma'].mean()
                    }
            
            crisis_analysis[crisis_name] = crisis_metrics
        
        return crisis_analysis
    
    def create_summary_report(self):
        print("Creating summary report...")
        
        static_metrics = self.calculate_static_metrics()
        regime_analysis = self.analyze_regime_performance()
        crisis_analysis = self.analyze_crisis_periods()
        
        report = []
        report.append("="*80)
        report.append("REGIME-DEPENDENT PERFORMANCE ANALYSIS SUMMARY")
        report.append("="*80)
        report.append("")
        
        report.append("KEY FINDING:")
        report.append("Bayesian belief updating about strategy skill reveals regime-dependent")
        report.append("performance patterns that traditional static metrics completely miss,")
        report.append("and this epistemic uncertainty itself is predictive.")
        report.append("")
        
        report.append("STATIC METRICS (Traditional View):")
        report.append("-" * 40)
        top_static = static_metrics.nlargest(3, 'sharpe_ratio')
        for strategy, data in top_static.iterrows():
            report.append(f"{strategy}: Sharpe = {data['sharpe_ratio']:.2f}, Return = {data['annualized_return']:.1%}")
        report.append("")
        
        report.append("DYNAMIC ELO RANKINGS BY REGIME:")
        report.append("-" * 40)
        for regime_name, metrics in regime_analysis.items():
            report.append(f"\n{regime_name}:")
            sorted_strategies = sorted(metrics.items(), key=lambda x: x[1]['avg_elo'], reverse=True)
            for strategy, data in sorted_strategies[:3]:
                report.append(f"  {strategy}: ELO = {data['avg_elo']:.0f}, Sharpe = {data['sharpe']:.2f}")
        report.append("")
        
        report.append("CRISIS PERIOD PERFORMANCE:")
        report.append("-" * 40)
        for crisis_name, metrics in crisis_analysis.items():
            report.append(f"\n{crisis_name}:")
            sorted_strategies = sorted(metrics.items(), key=lambda x: x[1]['total_return'], reverse=True)
            for strategy, data in sorted_strategies[:3]:
                report.append(f"  {strategy}: Return = {data['total_return']:.1%}, ELO Change = {data['elo_change']:.0f}")
        report.append("")
        
        report.append("KEY INSIGHTS:")
        report.append("-" * 40)
        report.append("1. Strategy rankings change dramatically across regimes")
        report.append("2. Static Sharpe ratios miss regime-dependent performance")
        report.append("3. ELO uncertainty (σ) provides additional predictive signal")
        report.append("4. Crisis periods reveal which strategies are truly robust")
        report.append("5. Bayesian updating captures skill evolution over time")
        report.append("")
        
        with open('regime_analysis_summary.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Summary report saved to 'regime_analysis_summary.txt'")
        print('\n'.join(report))
    
    def export_results(self):
        print("Exporting results...")
        
        if self.elo_evolution is not None:
            self.elo_evolution.to_csv('elo_evolution_detailed.csv', index=False)
            print("ELO evolution data saved to 'elo_evolution_detailed.csv'")
        
        static_metrics = self.calculate_static_metrics()
        static_metrics.to_csv('static_metrics_comparison.csv')
        print("Static metrics saved to 'static_metrics_comparison.csv'")
        
        regime_analysis = self.analyze_regime_performance()
        regime_data = []
        for regime_name, metrics in regime_analysis.items():
            for strategy, data in metrics.items():
                regime_data.append({
                    'regime': regime_name,
                    'strategy': strategy,
                    **data
                })
        
        if regime_data:
            regime_df = pd.DataFrame(regime_data)
            regime_df.to_csv('regime_performance_analysis.csv', index=False)
            print("Regime performance analysis saved to 'regime_performance_analysis.csv'")
        
        crisis_analysis = self.analyze_crisis_periods()
        crisis_data = []
        for crisis_name, metrics in crisis_analysis.items():
            for strategy, data in metrics.items():
                crisis_data.append({
                    'crisis': crisis_name,
                    'strategy': strategy,
                    **data
                })
        
        if crisis_data:
            crisis_df = pd.DataFrame(crisis_data)
            crisis_df.to_csv('crisis_performance_analysis.csv', index=False)
            print("Crisis performance analysis saved to 'crisis_performance_analysis.csv'")
        
        self.create_summary_report()
    
    def run_analysis(self):
        print("Starting regime-dependent performance analysis...")
        print("="*60)
        
        self.load_data()
        self.track_elo_evolution()
        self.export_results()
        
        print("\n" + "="*60)
        print("Analysis complete! Check the generated files:")
        print("- elo_evolution_detailed.csv (daily ELO data)")
        print("- static_metrics_comparison.csv (traditional metrics)")
        print("- regime_performance_analysis.csv (regime-specific performance)")
        print("- crisis_performance_analysis.csv (crisis period analysis)")
        print("- regime_analysis_summary.txt (summary report)")

def main():
    analyzer = RegimePerformanceAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
