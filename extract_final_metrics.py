"""
Extract Final Performance Metrics and Daily Match Results

This script extracts:
1. Final performance metrics (Sharpe ratio, returns, alpha) for each strategy
2. Daily "who beat who" results
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_sharpe_ratio(returns_series):
    """Calculate Sharpe ratio from returns series"""
    if len(returns_series) == 0 or returns_series.std() == 0:
        return 0.0
    return (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252))

def extract_final_metrics():
    """Extract final performance metrics for each strategy"""
    
    print("="*80)
    print("EXTRACTING FINAL PERFORMANCE METRICS")
    print("="*80)
    
    # Load the daily ELO evolution data
    try:
        daily_df = pd.read_csv('elo_daily_evolution.csv')
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        print(f"Loaded {len(daily_df)} daily ELO records")
        
    except FileNotFoundError:
        print("Error: elo_daily_evolution.csv not found. Please run elo_evolution_tracker.py first.")
        return
    
    # Load match records
    try:
        match_df = pd.read_csv('elo_match_records.csv')
        match_df['date'] = pd.to_datetime(match_df['date'])
        
        print(f"Loaded {len(match_df)} match records")
        
    except FileNotFoundError:
        print("Error: elo_match_records.csv not found. Please run elo_evolution_tracker.py first.")
        return
    
    # Calculate final metrics for each strategy
    final_metrics = []
    
    for strategy in daily_df['strategy'].unique():
        strategy_data = daily_df[daily_df['strategy'] == strategy]
        
        # Get final values
        final_elo = strategy_data['mu'].iloc[-1]
        final_uncertainty = strategy_data['sigma'].iloc[-1]
        final_cumulative_return = strategy_data['cumulative_return'].iloc[-1]
        final_matches = strategy_data['matches_played'].iloc[-1]
        final_wins = strategy_data['wins'].iloc[-1]
        final_losses = strategy_data['losses'].iloc[-1]
        final_draws = strategy_data['draws'].iloc[-1]
        
        # Calculate daily returns for Sharpe ratio
        daily_returns = strategy_data['daily_return'] / 100  # Convert percentage to decimal
        
        # Calculate metrics
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        win_rate = (final_wins / final_matches * 100) if final_matches > 0 else 0
        
        # Calculate total return in dollars (assuming $100,000 initial capital)
        initial_capital = 100000
        total_return_dollars = (final_cumulative_return / 100) * initial_capital
        
        # Calculate annualized return
        years = (strategy_data['date'].iloc[-1] - strategy_data['date'].iloc[0]).days / 365.25
        annualized_return = ((1 + final_cumulative_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate volatility
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Calculate alpha (excess return over market)
        # For simplicity, we'll use the cumulative return as alpha since we're using alpha returns in ELO
        alpha = final_cumulative_return
        
        final_metrics.append({
            'strategy': strategy,
            'final_elo_rating': round(final_elo, 2),
            'final_elo_uncertainty': round(final_uncertainty, 2),
            'total_return_pct': round(final_cumulative_return, 2),
            'total_return_dollars': round(total_return_dollars, 2),
            'annualized_return_pct': round(annualized_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'volatility_pct': round(volatility, 2),
            'alpha_pct': round(alpha, 2),
            'total_matches': final_matches,
            'wins': final_wins,
            'losses': final_losses,
            'draws': final_draws,
            'win_rate_pct': round(win_rate, 1)
        })
    
    # Create DataFrame and sort by ELO rating
    metrics_df = pd.DataFrame(final_metrics)
    metrics_df = metrics_df.sort_values('final_elo_rating', ascending=False).reset_index(drop=True)
    metrics_df.index = metrics_df.index + 1  # Start ranking at 1
    
    # Save to CSV
    metrics_df.to_csv('final_strategy_metrics.csv', index=True)
    print(f"\nFinal strategy metrics saved to final_strategy_metrics.csv")
    
    # Print summary
    print(f"\n" + "="*80)
    print("FINAL STRATEGY PERFORMANCE METRICS")
    print("="*80)
    print(metrics_df[['strategy', 'final_elo_rating', 'total_return_pct', 'sharpe_ratio', 'alpha_pct', 'win_rate_pct']].to_string(index=True))
    
    return metrics_df

def extract_daily_matches():
    """Extract daily 'who beat who' results"""
    
    print(f"\n" + "="*80)
    print("EXTRACTING DAILY MATCH RESULTS")
    print("="*80)
    
    # Load match records
    try:
        match_df = pd.read_csv('elo_match_records.csv')
        match_df['date'] = pd.to_datetime(match_df['date'])
        
        print(f"Loaded {len(match_df)} match records")
        
    except FileNotFoundError:
        print("Error: elo_match_records.csv not found. Please run elo_evolution_tracker.py first.")
        return
    
    # Create daily match results
    daily_matches = []
    
    for date in match_df['date'].unique():
        date_matches = match_df[match_df['date'] == date]
        
        for _, match in date_matches.iterrows():
            strategy_a = match['strategy_a']
            strategy_b = match['strategy_b']
            outcome = match['outcome']
            return_a = match['return_a']
            return_b = match['return_b']
            regime = match['regime']
            
            # Determine winner
            if outcome == 1.0:
                winner = strategy_a
                loser = strategy_b
                result = f"{strategy_a} beat {strategy_b}"
            elif outcome == 0.0:
                winner = strategy_b
                loser = strategy_a
                result = f"{strategy_b} beat {strategy_a}"
            else:
                winner = "Tie"
                loser = "Tie"
                result = f"{strategy_a} tied with {strategy_b}"
            
            daily_matches.append({
                'date': date,
                'strategy_a': strategy_a,
                'strategy_b': strategy_b,
                'return_a_pct': round(return_a, 4),
                'return_b_pct': round(return_b, 4),
                'outcome': outcome,
                'winner': winner,
                'loser': loser,
                'result': result,
                'regime': regime,
                'mu_change_a': round(match['mu_change_a'], 2),
                'mu_change_b': round(match['mu_change_b'], 2),
                'sigma_change_a': round(match['sigma_change_a'], 2),
                'sigma_change_b': round(match['sigma_change_b'], 2)
            })
    
    # Create DataFrame
    daily_matches_df = pd.DataFrame(daily_matches)
    daily_matches_df = daily_matches_df.sort_values(['date', 'strategy_a', 'strategy_b']).reset_index(drop=True)
    
    # Save to CSV
    daily_matches_df.to_csv('daily_match_results.csv', index=False)
    print(f"Daily match results saved to daily_match_results.csv")
    
    # Print summary statistics
    print(f"\nDaily Match Summary:")
    print(f"Total matches: {len(daily_matches_df)}")
    print(f"Date range: {daily_matches_df['date'].min().date()} to {daily_matches_df['date'].max().date()}")
    print(f"Unique strategies: {len(set(daily_matches_df['strategy_a'].unique()) | set(daily_matches_df['strategy_b'].unique()))}")
    
    # Show win counts by strategy
    print(f"\nWin counts by strategy:")
    win_counts = {}
    for _, match in daily_matches_df.iterrows():
        if match['winner'] != 'Tie':
            win_counts[match['winner']] = win_counts.get(match['winner'], 0) + 1
    
    win_counts_df = pd.DataFrame(list(win_counts.items()), columns=['strategy', 'wins'])
    win_counts_df = win_counts_df.sort_values('wins', ascending=False)
    print(win_counts_df.to_string(index=False))
    
    return daily_matches_df

def main():
    """Main function"""
    
    # Extract final metrics
    metrics_df = extract_final_metrics()
    
    # Extract daily matches
    daily_matches_df = extract_daily_matches()
    
    print(f"\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Files created:")
    print(f"  - final_strategy_metrics.csv: Final performance metrics for each strategy")
    print(f"  - daily_match_results.csv: Daily 'who beat who' results")
    
    return metrics_df, daily_matches_df

if __name__ == "__main__":
    main()










