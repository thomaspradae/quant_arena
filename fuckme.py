"""
ELO Evolution Tracker - Dynamic Strategy Performance Analysis

This script tracks ELO evolution over time and demonstrates how strategy performance
varies by market regime, showing the limitations of static metrics like Sharpe ratio.

Key outputs:
1. Daily ELO evolution with mu/sigma values
2. Match results and score changes
3. Regime-dependent performance analysis
4. Historical crisis period analysis
5. Performance around worst trading days
6. Final strategy metrics (Sharpe, returns, etc.)
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
WINDOW_AROUND = 5            # +/- days to examine around each worst day
TOP_N_WORST = 15

# Canonical historical moments to report (start, end)
CANONICAL_MOMENTS = {
    "GFC_2008": ("2007-10-01", "2009-03-31"),
    "COVID_2020": ("2020-02-15", "2020-04-30"),
    "Selloff_2022": ("2022-01-01", "2022-10-31"),
    "Dotcom_2000_2002": ("2000-03-01", "2002-10-01"),
    "TaperTantrum_2013": ("2013-05-01", "2013-08-31"),
}

class ELOEvolutionTracker:
    """Track ELO evolution over time with detailed logging"""
    
    def __init__(self, symbol='SPY', start_date='1997-01-01', end_date='2024-01-01', initial_capital=100_000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Initialize components
        self.data_manager = DataManager(cache_dir='./data_cache')
        self.backtest_engine = BacktestEngine(benchmark_symbol='SPY')
        self.strategy_registry = StrategyRegistry().create_default_universe()
        
        # Results storage
        self.daily_elo_data = []
        self.match_records = []
        self.final_metrics = {}
        
    def run_evolution_tracking(self):
        """Run complete ELO evolution tracking"""
        
        print("="*100)
        print("ELO EVOLUTION TRACKING - DYNAMIC STRATEGY PERFORMANCE ANALYSIS")
        print("="*100)
        
        # Adapt strategies
        self.adapted_strategies = adapt_strategies(
            list(self.strategy_registry.strategies.values()), 
            capital=self.initial_capital
        )
        
        # Fetch market data
        print(f"\n[Step 1] Fetching market data for {self.symbol}...")
        market_data = self.data_manager.fetch_data(
            symbol=self.symbol, 
            start_date=self.start_date, 
            end_date=self.end_date
        )
        
        print(f"Total trading days: {len(market_data)}")
        print(f"Date range: {market_data.index[0].date()} to {market_data.index[-1].date()}")
        
        # Detect regimes
        print(f"\n[Step 2] Detecting market regimes...")
        regime_detector = VolatilityRegimeDetector(window=20, num_regimes=2, method='quantile')
        regimes = regime_detector.detect_regimes(market_data)
        regimes = regime_detector.expand_to_full_index(regimes, market_data.index, method='ffill')
        
        # Initialize ranking manager
        ranking_manager = RankingManager(
            ranking_class=BayesianELORanking,
            initial_mu=1500,
            initial_sigma=350,
            min_sigma=50,
            tau=1.0,
            base_k=32
        )
        
        # Get strategy names and pairs
        strategy_names = [s.name for s in self.adapted_strategies]
        pairs = list(combinations(strategy_names, 2))
        
        print(f"\n[Step 3] Running backtests and tracking ELO evolution...")
        
        # Track ELO evolution day by day
        for i, date in enumerate(market_data.index):
            if i % 500 == 0:
                print(f"  Processing day {i+1}/{len(market_data)}: {date.date()}")
            
            # Get regime for this date
            regime_label = int(regimes.loc[date]) if date in regimes.index else 0
            
            # Run backtests for this day
            daily_data = market_data.loc[[date]]
            daily_results = {}
            
            for strategy in self.adapted_strategies:
                try:
                    result = self.backtest_engine.run_backtest(
                        strategy, 
                        daily_data, 
                        initial_capital=self.initial_capital,
                        benchmark_data=daily_data
                    )
                    daily_results[strategy.name] = result
                except Exception as e:
                    # Create dummy result for failed backtests
                    dummy_returns = pd.Series(0, index=[date])
                    daily_results[strategy.name] = type('obj', (object,), {
                        'returns': dummy_returns,
                        'alpha_returns': dummy_returns
                    })()
            
            # Get returns for this date
            scores = {}
            for name, result in daily_results.items():
                score = np.nan
                if hasattr(result, 'alpha_returns') and result.alpha_returns is not None:
                    try:
                        score = float(result.alpha_returns.loc[date])
                    except:
                        score = np.nan
                if np.isnan(score):
                    try:
                        score = float(result.returns.loc[date])
                    except:
                        score = np.nan
                scores[name] = score
            
            # Update ELO rankings with pairwise matches
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
                
                # Get ELO ratings before update
                s1_rating_before = ranking_manager.get_rating(s1, 'alpha', regime_label)
                s2_rating_before = ranking_manager.get_rating(s2, 'alpha', regime_label)
                
                # Update ELO
                ranking_manager.update(
                    strategy_a_name=s1,
                    strategy_b_name=s2,
                    outcome=outcome,
                    metric='alpha',
                    regime=regime_label,
                    timestamp=pd.Timestamp(date)
                )
                
                # Get ELO ratings after update
                s1_rating_after = ranking_manager.get_rating(s1, 'alpha', regime_label)
                s2_rating_after = ranking_manager.get_rating(s2, 'alpha', regime_label)
                
                # Record match
                self.match_records.append({
                    'date': date,
                    'regime': regime_label,
                    'strategy_a': s1,
                    'strategy_b': s2,
                    'score_a': scores[s1],
                    'score_b': scores[s2],
                    'outcome': outcome,
                    'elo_a_before': s1_rating_before.mu if s1_rating_before else 1500,
                    'elo_b_before': s2_rating_before.mu if s2_rating_before else 1500,
                    'elo_a_after': s1_rating_after.mu if s1_rating_after else 1500,
                    'elo_b_after': s2_rating_after.mu if s2_rating_after else 1500,
                    'elo_change_a': (s1_rating_after.mu - s1_rating_before.mu) if s1_rating_after and s1_rating_before else 0,
                    'elo_change_b': (s2_rating_after.mu - s2_rating_before.mu) if s2_rating_after and s2_rating_before else 0,
                    'sigma_a': s1_rating_after.sigma if s1_rating_after else 350,
                    'sigma_b': s2_rating_after.sigma if s2_rating_after else 350
                })
            
            # Record daily ELO data for all strategies
            for strategy_name in strategy_names:
                rating = ranking_manager.get_rating(strategy_name, 'alpha', regime_label)
                if rating:
                    # Calculate cumulative returns
                    cumulative_return = 0
                    if i > 0:
                        # Get previous day's cumulative return
                        prev_data = [d for d in self.daily_elo_data if d['strategy'] == strategy_name and d['date'] < date]
                        if prev_data:
                            cumulative_return = prev_data[-1]['cumulative_return']
                    
                    # Add today's return
                    daily_return = scores.get(strategy_name, 0) * 100  # Convert to percentage
                    cumulative_return += daily_return
                    
                    self.daily_elo_data.append({
                        'date': date,
                        'strategy': strategy_name,
                        'regime': regime_label,
                        'mu': rating.mu,
                        'sigma': rating.sigma,
                        'daily_return': daily_return,
                        'cumulative_return': cumulative_return
                    })
        
        print(f"\n[Step 4] Calculating final metrics...")
        self._calculate_final_metrics(market_data)
        
        print(f"\n[Step 5] Exporting results...")
        self._export_results()
        
        return self.daily_elo_data, self.match_records, self.final_metrics
    
    def _calculate_final_metrics(self, market_data):
        """Calculate final strategy metrics"""
        
        daily_df = pd.DataFrame(self.daily_elo_data)
        
        for strategy in daily_df['strategy'].unique():
            strategy_data = daily_df[daily_df['strategy'] == strategy]
            
            # Calculate returns
            returns = strategy_data['daily_return'] / 100
            
            # Basic metrics
            total_return_pct = strategy_data['cumulative_return'].iloc[-1]
            total_return_money = self.initial_capital * (total_return_pct / 100)
            annualized_return = (1 + total_return_pct/100) ** (252/len(strategy_data)) - 1
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # ELO metrics
            final_elo = strategy_data['mu'].iloc[-1]
            final_uncertainty = strategy_data['sigma'].iloc[-1]
            elo_change = final_elo - strategy_data['mu'].iloc[0]
            
            # Worst drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            self.final_metrics[strategy] = {
                'total_return_pct': total_return_pct,
                'total_return_money': total_return_money,
                'annualized_return': annualized_return,
                'volatility_pct': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'final_elo': final_elo,
                'final_uncertainty': final_uncertainty,
                'elo_change': elo_change,
                'trading_days': len(strategy_data)
            }
    
    def _export_results(self):
        """Export all results to files"""
        
        # Export daily ELO evolution
        daily_df = pd.DataFrame(self.daily_elo_data)
        daily_df.to_csv('elo_daily_evolution.csv', index=False)
        print(f"Daily ELO evolution saved to elo_daily_evolution.csv")
        
        # Export match records
        match_df = pd.DataFrame(self.match_records)
        match_df.to_csv('elo_match_records.csv', index=False)
        print(f"Match records saved to elo_match_records.csv")
        
        # Export final metrics
        metrics_df = pd.DataFrame(self.final_metrics).T
        metrics_df.to_csv('final_strategy_metrics.csv')
        print(f"Final strategy metrics saved to final_strategy_metrics.csv")
        
        # Create comprehensive Excel report
        with pd.ExcelWriter('elo_evolution_analysis.xlsx', engine='openpyxl') as writer:
            # Daily evolution
            daily_df.to_excel(writer, sheet_name='Daily_ELO_Evolution', index=False)
            
            # Match records
            match_df.to_excel(writer, sheet_name='Match_Records', index=False)
            
            # Final metrics
            metrics_df.to_excel(writer, sheet_name='Final_Metrics')
            
            # Regime analysis
            regime_analysis = daily_df.groupby(['strategy', 'regime']).agg({
                'mu': ['mean', 'std', 'min', 'max'],
                'sigma': ['mean', 'std'],
                'daily_return': ['mean', 'std']
            }).round(3)
            regime_analysis.columns = ['_'.join(col).strip() for col in regime_analysis.columns]
            regime_analysis.to_excel(writer, sheet_name='Regime_Analysis')
        
        print(f"Comprehensive analysis saved to elo_evolution_analysis.xlsx")
    
    def create_visualizations(self):
        """Create visualizations showing ELO evolution and regime dependence"""
        
        print(f"\n[Step 6] Creating visualizations...")
        
        daily_df = pd.DataFrame(self.daily_elo_data)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('ELO Evolution and Regime-Dependent Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. ELO Evolution Over Time
        for strategy in daily_df['strategy'].unique():
            strategy_data = daily_df[daily_df['strategy'] == strategy]
            axes[0,0].plot(strategy_data['date'], strategy_data['mu'], label=strategy, alpha=0.7)
        axes[0,0].set_title('ELO Evolution Over Time')
        axes[0,0].set_ylabel('ELO Rating (μ)')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Uncertainty (Sigma) Evolution
        for strategy in daily_df['strategy'].unique():
            strategy_data = daily_df[daily_df['strategy'] == strategy]
            axes[0,1].plot(strategy_data['date'], strategy_data['sigma'], label=strategy, alpha=0.7)
        axes[0,1].set_title('Uncertainty (σ) Evolution Over Time')
        axes[0,1].set_ylabel('Uncertainty (σ)')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. ELO by Regime (Box Plot)
        regime_elo_data = []
        for _, row in daily_df.iterrows():
            regime_elo_data.append({
                'regime': f"Regime {row['regime']}",
                'strategy': row['strategy'],
                'elo': row['mu']
            })
        regime_elo_df = pd.DataFrame(regime_elo_data)
        
        sns.boxplot(data=regime_elo_df, x='regime', y='elo', hue='strategy', ax=axes[1,0])
        axes[1,0].set_title('ELO Distribution by Market Regime')
        axes[1,0].set_ylabel('ELO Rating (μ)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Final ELO vs Sharpe Ratio
        metrics_df = pd.DataFrame(self.final_metrics).T
        scatter = axes[1,1].scatter(metrics_df['sharpe_ratio'], metrics_df['final_elo'], 
                                   s=100, alpha=0.7, c=metrics_df['total_return_pct'], 
                                   cmap='viridis')
        axes[1,1].set_xlabel('Sharpe Ratio')
        axes[1,1].set_ylabel('Final ELO Rating')
        axes[1,1].set_title('Final ELO vs Sharpe Ratio\n(Color = Total Return %)')
        
        # Add strategy labels
        for strategy, row in metrics_df.iterrows():
            axes[1,1].annotate(strategy, (row['sharpe_ratio'], row['final_elo']), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=axes[1,1], label='Total Return (%)')
        
        # 5. Regime Performance Heatmap
        regime_performance = daily_df.groupby(['strategy', 'regime'])['mu'].mean().unstack()
        sns.heatmap(regime_performance, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[2,0])
        axes[2,0].set_title('Average ELO by Strategy and Regime')
        
        # 6. Cumulative Returns Comparison
        for strategy in daily_df['strategy'].unique():
            strategy_data = daily_df[daily_df['strategy'] == strategy]
            axes[2,1].plot(strategy_data['date'], strategy_data['cumulative_return'], 
                          label=strategy, alpha=0.8)
        axes[2,1].set_title('Cumulative Returns Over Time')
        axes[2,1].set_ylabel('Cumulative Return (%)')
        axes[2,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('elo_evolution_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to elo_evolution_analysis.png")
        
        plt.show()
    
    def analyze_regime_performance(self):
        """Analyze performance differences across regimes"""
        
        print(f"\n[Step 7] Analyzing regime-dependent performance...")
        
        daily_df = pd.DataFrame(self.daily_elo_data)
        metrics_df = pd.DataFrame(self.final_metrics).T
        
        print("\n" + "="*80)
        print("REGIME-DEPENDENT PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Calculate regime statistics
        regime_stats = daily_df.groupby(['strategy', 'regime']).agg({
            'mu': ['mean', 'std', 'min', 'max'],
            'daily_return': ['mean', 'std']
        }).round(3)
        
        print(f"\nELO Performance by Regime:")
        print(regime_stats.to_string())
        
        # Find strategies with highest regime variation
        regime_means = daily_df.groupby(['strategy', 'regime'])['mu'].mean().unstack()
        regime_variation = regime_means.max(axis=1) - regime_means.min(axis=1)
        
        print(f"\nStrategies with Highest ELO Variation Across Regimes:")
        for strategy, variation in regime_variation.sort_values(ascending=False).head(5).items():
            print(f"  {strategy}: {variation:.1f} ELO points variation")
        
        # Compare with static Sharpe ratios
        print(f"\nStatic vs Dynamic Performance Comparison:")
        print(f"{'Strategy':<20} {'Sharpe':<8} {'Final ELO':<10} {'ELO Variation':<15}")
        print("-" * 60)
        
        for strategy in regime_variation.index:
            sharpe = metrics_df.loc[strategy, 'sharpe_ratio']
            final_elo = metrics_df.loc[strategy, 'final_elo']
            variation = regime_variation[strategy]
            print(f"{strategy:<20} {sharpe:<8.2f} {final_elo:<10.1f} {variation:<15.1f}")
        
        return regime_stats, regime_variation


def main():
    """Main function to run ELO evolution tracking"""
    
    # Initialize tracker
    tracker = ELOEvolutionTracker(
        symbol='SPY',
        start_date='1997-01-01',
        end_date='2024-01-01',
        initial_capital=100_000
    )
    
    # Run tracking
    daily_data, match_records, final_metrics = tracker.run_evolution_tracking()
    
    # Create visualizations
    tracker.create_visualizations()
    
    # Analyze regime performance
    regime_stats, regime_variation = tracker.analyze_regime_performance()
    
    print("\n" + "="*100)
    print("ELO EVOLUTION TRACKING COMPLETE")
    print("="*100)
    print("Files created:")
    print("  - elo_daily_evolution.csv: Daily ELO ratings and returns")
    print("  - elo_match_records.csv: Individual match results and ELO changes")
    print("  - final_strategy_metrics.csv: Final performance metrics")
    print("  - elo_evolution_analysis.xlsx: Comprehensive Excel report")
    print("  - elo_evolution_analysis.png: Visualization")
    
    return daily_data, match_records, final_metrics


if __name__ == "__main__":
    main()