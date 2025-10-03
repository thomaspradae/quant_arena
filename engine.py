"""
Main Engine - Final Version with Meta-Analysis

This script now orchestrates the entire pipeline and adds a final meta-analysis step:
1. Loads market data.
2. Detects market regimes.
3. Initializes and backtests a universe of trading strategies.
4. Generates head-to-head matchups using a Matchmaker.
5. Simulates the tournament, comparing backtest results to determine outcomes.
6. Updates ELO ratings for each strategy using a Ranking Manager.
7. Displays the final ELO leaderboards.
8. Performs and displays a meta-analysis of the ELO dynamics (velocity, volatility, lifecycle).
"""

# We assume your component files are in the same directory or accessible.
from data_manager import DataManager
from regime_detector import HMMRegimeDetector
from strategy_zoo import StrategyRegistry, BuyAndHold, SMACrossover, BollingerMeanReversion, RSIMomentum, Strategy
from backtest_engine import BacktestEngine, BacktestResult
from matchmaker import RoundRobinMatcher, Matchup
from ranking import RankingManager, ELORanking
from analyzer import MetaELOAnalyzer

import pandas as pd
import numpy as np
from typing import Dict, List

def _compare_strategies(result_a: BacktestResult, result_b: BacktestResult, metric: str) -> float:
    """
    Compares two backtest results based on a given metric.

    Returns:
        1.0 if A wins, 0.0 if B wins, 0.5 for a tie.
    """
    metric_a = result_a.metrics.get(metric, -np.inf)
    metric_b = result_b.metrics.get(metric, -np.inf)

    if metric_a > metric_b:
        return 1.0  # A wins
    elif metric_b > metric_a:
        return 0.0  # B wins
    else:
        return 0.5  # Tie

def run():
    """
    The complete engine run function for the ELO tournament and meta-analysis.
    """
    print("="*80)
    print("Trading Strategy ELO Engine - Initializing Full Tournament & Meta-Analysis...")
    print("="*80)

    # --- 1. Initialize Components ---
    print("\n[Step 1] Initializing Core Components...")
    data_manager = DataManager(cache_dir='./data_cache')
    regime_detector = HMMRegimeDetector(num_regimes=3)
    strategy_registry = StrategyRegistry()
    backtest_engine = BacktestEngine()
    matchmaker = RoundRobinMatcher(include_global=True)
    # Refactor: Use RankingManager to handle multiple metrics/regimes
    ranking_manager = RankingManager(ranking_class=ELORanking, k_factor=32, initial_rating=1500)
    print("   -> Components Initialized: DataManager, RegimeDetector, StrategyRegistry, BacktestEngine, Matchmaker, RankingManager")

    # --- 2. Register Strategies ---
    print("\n[Step 2] Registering Strategy Universe...")
    strategy_registry.register(BuyAndHold())
    strategy_registry.register(SMACrossover(fast_period=20, slow_period=50))
    strategy_registry.register(SMACrossover(fast_period=50, slow_period=200))
    strategy_registry.register(BollingerMeanReversion(period=20, num_std=2.0))
    strategy_registry.register(RSIMomentum(period=14))
    print(f"   -> {len(strategy_registry.list_all())} strategies registered.")
    
    # --- 3. Define Experiment Parameters ---
    target_symbol = 'SPY'
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    initial_capital = 100_000
    comparison_metric = 'sharpe_ratio'

    try:
        # --- 4. Fetch Market Data ---
        print(f"\n[Step 3] Fetching data for '{target_symbol}'...")
        market_data_df = data_manager.fetch_data(symbol=target_symbol, start_date=start_date, end_date=end_date)
        print("   -> Data fetch successful.")

        # --- 5. Detect Regimes ---
        print(f"\n[Step 4] Detecting regimes in '{target_symbol}' data...")
        regimes = regime_detector.detect_regimes(market_data_df)
        print("   -> Regime detection successful.")
        
        # --- 6. Run Initial Backtests for All Strategies ---
        print("\n[Step 5] Running baseline backtests for all strategies...")
        all_results: Dict[str, BacktestResult] = {}
        strategies_list = [s for s in strategy_registry.strategies.values()]
        for strategy in strategies_list:
            result = backtest_engine.run_backtest(strategy, market_data_df, initial_capital=initial_capital)
            all_results[strategy.name] = result
        print(f"   -> Completed {len(all_results)} baseline backtests.")

        # --- 7. Generate Matchups ---
        print("\n[Step 6] Generating tournament matchups...")
        matchups = matchmaker.generate_matchups(strategies_list, market_data_df, regimes, asset=target_symbol)
        print(f"   -> Generated {len(matchups)} matchups across all regimes.")

        # --- 8. Run Tournament & Update ELO Ratings ---
        print("\n[Step 7] Running ELO tournament...")
        for matchup in matchups:
            strat_a = matchup.strategy_a
            strat_b = matchup.strategy_b
            result_a = all_results[strat_a.name]
            result_b = all_results[strat_b.name]
            outcome = _compare_strategies(result_a, result_b, comparison_metric)
            
            # Use the RankingManager to update ratings
            ranking_manager.update(
                strategy_a_name=strat_a.name, 
                strategy_b_name=strat_b.name, 
                outcome=outcome, 
                metric=comparison_metric,
                regime=matchup.regime
            )
        print("   -> Tournament complete. ELO ratings have been updated.")

        # --- 9. Display Final Leaderboards ---
        print("\n" + "="*80)
        print("FINAL ELO LEADERBOARDS")
        print("="*80)
        print(f"\nBased on metric: '{comparison_metric}'")

        print("\n--- Global Leaderboard ---")
        global_leaderboard = ranking_manager.get_leaderboard(metric=comparison_metric, regime=None)
        print(global_leaderboard.to_string())

        for regime_label in sorted(regimes.unique()):
            print(f"\n--- Regime {regime_label} Leaderboard ({regime_detector.regime_descriptions.get(regime_label, '')}) ---")
            regime_leaderboard = ranking_manager.get_leaderboard(metric=comparison_metric, regime=regime_label)
            if not regime_leaderboard.empty:
                print(regime_leaderboard.to_string())
            else:
                print("No matches played in this regime.")
        
        # --- 10. Perform and Display Meta-Analysis ---
        print("\n" + "="*80)
        print("META ELO ANALYSIS")
        print("="*80)

        meta_analyzer = MetaELOAnalyzer(ranking_manager)
        
        print(f"\n--- ELO Velocity Ranking (Momentum) ---")
        velocity_ranking = meta_analyzer.get_velocity_ranking(metric=comparison_metric, regime=None)
        print(velocity_ranking.to_string())
        
        print(f"\n--- ELO Volatility Ranking (Consistency) ---")
        volatility_ranking = meta_analyzer.get_volatility_ranking(metric=comparison_metric, regime=None)
        print(volatility_ranking.to_string())

        print(f"\n--- Strategy Lifecycle Report ---")
        lifecycle_report = meta_analyzer.get_lifecycle_report(metric=comparison_metric, regime=None)
        print(lifecycle_report.to_string())

        print("\n" + "="*80)

    except Exception as e:
        print(f"\n[ERROR] An error occurred during the engine run: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)

if __name__ == "__main__":
    run()

