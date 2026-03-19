"""
Plot ELO Evolution Over Time

This script creates visualizations of ELO ratings evolution over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def plot_elo_evolution():
    """Create ELO evolution plots"""
    
    # Load the daily ELO evolution data
    try:
        daily_df = pd.read_csv('elo_daily_evolution.csv')
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        print(f"Loaded {len(daily_df)} daily ELO records")
        print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        print(f"Strategies: {daily_df['strategy'].unique()}")
        
    except FileNotFoundError:
        print("Error: elo_daily_evolution.csv not found. Please run elo_evolution_tracker.py first.")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ELO Evolution Analysis (1997-2023)', fontsize=16, fontweight='bold')
    
    # 1. ELO Ratings Over Time
    ax1 = axes[0, 0]
    for strategy in daily_df['strategy'].unique():
        strategy_data = daily_df[daily_df['strategy'] == strategy]
        ax1.plot(strategy_data['date'], strategy_data['mu'], label=strategy, linewidth=1.5, alpha=0.8)
    
    ax1.set_title('ELO Ratings Over Time', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('ELO Rating (μ)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. ELO Uncertainty Over Time
    ax2 = axes[0, 1]
    for strategy in daily_df['strategy'].unique():
        strategy_data = daily_df[daily_df['strategy'] == strategy]
        ax2.plot(strategy_data['date'], strategy_data['sigma'], label=strategy, linewidth=1.5, alpha=0.8)
    
    ax2.set_title('ELO Uncertainty Over Time', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('ELO Uncertainty (σ)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative Returns Over Time
    ax3 = axes[1, 0]
    for strategy in daily_df['strategy'].unique():
        strategy_data = daily_df[daily_df['strategy'] == strategy]
        ax3.plot(strategy_data['date'], strategy_data['cumulative_return'], label=strategy, linewidth=1.5, alpha=0.8)
    
    ax3.set_title('Cumulative Returns Over Time', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulative Return (%)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Final ELO vs Performance Scatter
    ax4 = axes[1, 1]
    
    # Get final ELO ratings and performance metrics
    final_elos = daily_df.groupby('strategy')['mu'].last()
    final_returns = daily_df.groupby('strategy')['cumulative_return'].last()
    final_uncertainties = daily_df.groupby('strategy')['sigma'].last()
    
    scatter = ax4.scatter(final_elos, final_returns, 
                         s=final_uncertainties*2,  # Size based on uncertainty
                         alpha=0.7, c=range(len(final_elos)), cmap='viridis')
    
    # Add strategy labels
    for strategy, elo, ret in zip(final_elos.index, final_elos.values, final_returns.values):
        ax4.annotate(strategy, (elo, ret), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    ax4.set_title('Final ELO vs Performance', fontweight='bold')
    ax4.set_xlabel('Final ELO Rating')
    ax4.set_ylabel('Final Cumulative Return (%)')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar for uncertainty
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('ELO Uncertainty (σ)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('elo_evolution_analysis.png', dpi=300, bbox_inches='tight')
    print("ELO evolution plot saved to elo_evolution_analysis.png")
    
    # Create a separate detailed ELO evolution plot
    plt.figure(figsize=(14, 8))
    
    # Plot ELO ratings with uncertainty bands
    for strategy in daily_df['strategy'].unique():
        strategy_data = daily_df[daily_df['strategy'] == strategy]
        
        # Plot the mean ELO rating
        plt.plot(strategy_data['date'], strategy_data['mu'], label=strategy, linewidth=2)
        
        # Add uncertainty bands
        plt.fill_between(strategy_data['date'], 
                        strategy_data['mu'] - strategy_data['sigma'],
                        strategy_data['mu'] + strategy_data['sigma'],
                        alpha=0.2)
    
    plt.title('ELO Ratings Evolution with Uncertainty Bands (1997-2023)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('ELO Rating', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elo_evolution_detailed.png', dpi=300, bbox_inches='tight')
    print("Detailed ELO evolution plot saved to elo_evolution_detailed.png")
    
    # Print summary statistics
    print(f"\n" + "="*60)
    print("ELO EVOLUTION SUMMARY")
    print("="*60)
    
    print(f"\nFinal ELO Rankings:")
    final_leaderboard = daily_df.groupby('strategy')['mu'].last().sort_values(ascending=False)
    for i, (strategy, elo) in enumerate(final_leaderboard.items(), 1):
        uncertainty = daily_df[daily_df['strategy'] == strategy]['sigma'].iloc[-1]
        print(f"  {i}. {strategy}: {elo:.1f} ± {uncertainty:.1f}")
    
    print(f"\nPerformance Summary:")
    performance_summary = daily_df.groupby('strategy').agg({
        'cumulative_return': 'last',
        'mu': 'last',
        'sigma': 'last',
        'matches_played': 'last',
        'wins': 'last',
        'losses': 'last'
    }).round(2)
    
    performance_summary['win_rate'] = (performance_summary['wins'] / 
                                     (performance_summary['wins'] + performance_summary['losses']) * 100).round(1)
    
    print(performance_summary[['cumulative_return', 'mu', 'sigma', 'win_rate']].to_string())
    
    plt.show()

if __name__ == "__main__":
    plot_elo_evolution()










