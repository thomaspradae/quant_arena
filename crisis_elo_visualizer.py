"""
Crisis ELO Visualizer - Show ELO Evolution During Crisis Periods

This script creates detailed visualizations of ELO ratings and returns
during specific crisis periods, comparing best performers to Buy and Hold.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Crisis periods to analyze
CRISIS_PERIODS = {
    "GFC_2008": ("2007-10-01", "2009-03-31"),
    "COVID_2020": ("2020-02-15", "2020-04-30"),
    "Selloff_2022": ("2022-01-01", "2022-10-31"),
    "Dotcom_2000_2002": ("2000-03-01", "2002-10-01"),
    "TaperTantrum_2013": ("2013-05-01", "2013-08-31"),
}

# Best performing strategies for each crisis
BEST_STRATEGIES = {
    "GFC_2008": "FadeExtremes_63d",
    "COVID_2020": "FadeExtremes_63d", 
    "Selloff_2022": "MeanReversion_20d",
    "Dotcom_2000_2002": "FadeExtremes_63d",
    "TaperTantrum_2013": "MeanReversion_20d",
}

def load_elo_data():
    """Load ELO evolution data"""
    try:
        elo_df = pd.read_csv('elo_full_evolution_1997_2024.csv')
        elo_df['date'] = pd.to_datetime(elo_df['date'], utc=True).dt.tz_localize(None)
        return elo_df
    except FileNotFoundError:
        print("Error: elo_full_evolution_1997_2024.csv not found. Please run elo_full_evolution_tracker.py first.")
        return None

def load_crisis_data():
    """Load crisis performance data"""
    try:
        crisis_df = pd.read_csv('crisis_performance_results.csv')
        return crisis_df
    except FileNotFoundError:
        print("Error: crisis_performance_results.csv not found. Please run regime_performance_analyzer.py first.")
        return None

def create_crisis_elo_plots(elo_df, crisis_df):
    """Create detailed ELO evolution plots for crisis periods"""
    
    # Create figure with subplots for each crisis
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()
    
    for i, (crisis_name, (start_date, end_date)) in enumerate(CRISIS_PERIODS.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Filter ELO data for this crisis period
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        crisis_mask = (elo_df['date'] >= start_dt) & (elo_df['date'] <= end_dt)
        crisis_elo = elo_df[crisis_mask].copy()
        
        if crisis_elo.empty:
            ax.text(0.5, 0.5, f'No data for {crisis_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{crisis_name} - No Data')
            continue
        
        # Get best strategy for this crisis
        best_strategy = BEST_STRATEGIES[crisis_name]
        
        # Plot ELO evolution for key strategies
        strategies_to_plot = [best_strategy, 'BuyAndHold', 'MomXover_20_50', 'LowVol_20d']
        
        for strategy in strategies_to_plot:
            strategy_data = crisis_elo[crisis_elo['strategy'] == strategy]
            if not strategy_data.empty:
                ax.plot(strategy_data['date'], strategy_data['mu'], 
                       label=strategy, linewidth=2, alpha=0.8)
        
        # Add crisis period shading
        ax.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date), 
                  alpha=0.1, color='red', label='Crisis Period')
        
        # Customize plot
        ax.set_title(f'{crisis_name}\nBest: {best_strategy}', fontsize=12, fontweight='bold')
        ax.set_ylabel('ELO Rating')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        
        # Add crisis start/end markers
        ax.axvline(pd.to_datetime(start_date), color='red', linestyle='--', alpha=0.7)
        ax.axvline(pd.to_datetime(end_date), color='red', linestyle='--', alpha=0.7)
    
    # Remove empty subplot if odd number of crises
    if len(CRISIS_PERIODS) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('crisis_elo_evolution_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_crisis_comparison_plot(elo_df, crisis_df):
    """Create comparison plot showing ELO changes during crises"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data for comparison
    comparison_data = []
    
    for crisis_name, (start_date, end_date) in CRISIS_PERIODS.items():
        best_strategy = BEST_STRATEGIES[crisis_name]
        
        # Get ELO data for this crisis
        crisis_mask = (elo_df['date'] >= start_date) & (elo_df['date'] <= end_date)
        crisis_elo = elo_df[crisis_mask]
        
        if crisis_elo.empty:
            continue
        
        # Calculate ELO change for each strategy
        for strategy in ['BuyAndHold', best_strategy, 'MomXover_20_50', 'LowVol_20d']:
            strategy_data = crisis_elo[crisis_elo['strategy'] == strategy]
            if len(strategy_data) >= 2:
                elo_start = strategy_data.iloc[0]['mu']
                elo_end = strategy_data.iloc[-1]['mu']
                elo_change = elo_end - elo_start
                
                comparison_data.append({
                    'Crisis': crisis_name,
                    'Strategy': strategy,
                    'ELO_Change': elo_change,
                    'ELO_Start': elo_start,
                    'ELO_End': elo_end,
                    'Is_Best': strategy == best_strategy
                })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Plot 1: ELO Changes by Crisis
    sns.barplot(data=comp_df, x='Crisis', y='ELO_Change', hue='Strategy', ax=axes[0,0])
    axes[0,0].set_title('ELO Rating Changes During Crisis Periods')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Best Strategy vs Buy and Hold ELO Changes
    best_vs_bh = comp_df[comp_df['Strategy'].isin(['BuyAndHold']) | comp_df['Is_Best']].copy()
    best_vs_bh['Strategy_Type'] = best_vs_bh.apply(
        lambda x: 'Best Strategy' if x['Is_Best'] else 'Buy & Hold', axis=1
    )
    
    sns.barplot(data=best_vs_bh, x='Crisis', y='ELO_Change', hue='Strategy_Type', ax=axes[0,1])
    axes[0,1].set_title('Best Strategy vs Buy & Hold: ELO Changes')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Crisis Returns vs ELO Changes
    crisis_returns = []
    for crisis_name in CRISIS_PERIODS.keys():
        crisis_data = crisis_df[crisis_df['Crisis'] == crisis_name]
        if not crisis_data.empty:
            best_strategy = BEST_STRATEGIES[crisis_name]
            best_return = crisis_data[crisis_data['Strategy'] == best_strategy]['total_return'].iloc[0]
            bh_return = crisis_data[crisis_data['Strategy'] == 'BuyAndHold']['total_return'].iloc[0]
            
            crisis_returns.append({
                'Crisis': crisis_name,
                'Best_Strategy_Return': best_return,
                'BuyHold_Return': bh_return,
                'Return_Difference': best_return - bh_return
            })
    
    returns_df = pd.DataFrame(crisis_returns)
    
    # Merge with ELO data
    elo_changes = comp_df[comp_df['Is_Best']].set_index('Crisis')['ELO_Change']
    returns_df['Best_Strategy_ELO_Change'] = returns_df['Crisis'].map(elo_changes)
    
    axes[1,0].scatter(returns_df['Best_Strategy_ELO_Change'], returns_df['Return_Difference'], s=100)
    for i, row in returns_df.iterrows():
        axes[1,0].annotate(row['Crisis'], 
                          (row['Best_Strategy_ELO_Change'], row['Return_Difference']),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1,0].set_xlabel('Best Strategy ELO Change')
    axes[1,0].set_ylabel('Return Difference (Best - Buy & Hold)')
    axes[1,0].set_title('ELO Change vs Return Outperformance')
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Crisis Performance Summary
    crisis_summary = []
    for crisis_name in CRISIS_PERIODS.keys():
        crisis_data = crisis_df[crisis_df['Crisis'] == crisis_name]
        if not crisis_data.empty:
            best_strategy = BEST_STRATEGIES[crisis_name]
            best_return = crisis_data[crisis_data['Strategy'] == best_strategy]['total_return'].iloc[0]
            bh_return = crisis_data[crisis_data['Strategy'] == 'BuyAndHold']['total_return'].iloc[0]
            
            crisis_summary.append({
                'Crisis': crisis_name,
                'Best_Strategy': best_strategy,
                'Best_Return': best_return,
                'BuyHold_Return': bh_return,
                'Outperformance': best_return - bh_return
            })
    
    summary_df = pd.DataFrame(crisis_summary)
    
    x = np.arange(len(summary_df))
    width = 0.35
    
    axes[1,1].bar(x - width/2, summary_df['Best_Return'], width, label='Best Strategy', alpha=0.8)
    axes[1,1].bar(x + width/2, summary_df['BuyHold_Return'], width, label='Buy & Hold', alpha=0.8)
    axes[1,1].set_xlabel('Crisis Period')
    axes[1,1].set_ylabel('Total Return')
    axes[1,1].set_title('Crisis Performance: Best Strategy vs Buy & Hold')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(summary_df['Crisis'], rotation=45, ha='right')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('crisis_elo_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_elo_velocity_plot(elo_df):
    """Create ELO velocity (rate of change) plots during crises"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (crisis_name, (start_date, end_date)) in enumerate(CRISIS_PERIODS.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Filter ELO data for this crisis period
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        crisis_mask = (elo_df['date'] >= start_dt) & (elo_df['date'] <= end_dt)
        crisis_elo = elo_df[crisis_mask].copy()
        
        if crisis_elo.empty:
            ax.text(0.5, 0.5, f'No data for {crisis_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{crisis_name} - No Data')
            continue
        
        # Get best strategy for this crisis
        best_strategy = BEST_STRATEGIES[crisis_name]
        
        # Calculate ELO velocity (rate of change)
        strategies_to_plot = [best_strategy, 'BuyAndHold']
        
        for strategy in strategies_to_plot:
            strategy_data = crisis_elo[crisis_elo['strategy'] == strategy].copy()
            if len(strategy_data) > 1:
                strategy_data = strategy_data.sort_values('date')
                strategy_data['ELO_Velocity'] = strategy_data['mu'].diff()
                
                ax.plot(strategy_data['date'], strategy_data['ELO_Velocity'], 
                       label=f'{strategy} Velocity', linewidth=2, alpha=0.8)
        
        # Add crisis period shading
        ax.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date), 
                  alpha=0.1, color='red')
        
        # Customize plot
        ax.set_title(f'{crisis_name}\nELO Velocity (Rate of Change)', fontsize=10)
        ax.set_ylabel('ELO Velocity')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for j in range(len(CRISIS_PERIODS), len(axes)):
        axes[j].remove()
    
    plt.tight_layout()
    plt.savefig('crisis_elo_velocity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_crisis_summary_table(crisis_df):
    """Create a summary table of crisis performance"""
    
    print("\n" + "="*100)
    print("CRISIS PERFORMANCE SUMMARY")
    print("="*100)
    
    summary_data = []
    
    for crisis_name in CRISIS_PERIODS.keys():
        crisis_data = crisis_df[crisis_df['Crisis'] == crisis_name]
        if not crisis_data.empty:
            best_strategy = BEST_STRATEGIES[crisis_name]
            
            # Get performance data
            best_perf = crisis_data[crisis_data['Strategy'] == best_strategy].iloc[0]
            bh_perf = crisis_data[crisis_data['Strategy'] == 'BuyAndHold'].iloc[0]
            
            summary_data.append({
                'Crisis': crisis_name,
                'Best_Strategy': best_strategy,
                'Best_Return': f"{best_perf['total_return']:.1%}",
                'Best_Sharpe': f"{best_perf['sharpe_ratio']:.2f}",
                'Best_ELO_Score': f"{best_perf['elo_score']:.2f}",
                'BuyHold_Return': f"{bh_perf['total_return']:.1%}",
                'BuyHold_Sharpe': f"{bh_perf['sharpe_ratio']:.2f}",
                'Outperformance': f"{best_perf['total_return'] - bh_perf['total_return']:.1%}",
                'Days': int(best_perf['days'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print formatted table
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv('crisis_performance_summary.csv', index=False)
    print(f"\nSummary saved to crisis_performance_summary.csv")
    
    return summary_df

def main():
    """Main function to create all crisis visualizations"""
    
    print("="*100)
    print("CRISIS ELO VISUALIZER")
    print("="*100)
    
    # Load data
    print("\n[Step 1] Loading ELO evolution data...")
    elo_df = load_elo_data()
    if elo_df is None:
        return
    
    print("\n[Step 2] Loading crisis performance data...")
    crisis_df = load_crisis_data()
    if crisis_df is None:
        return
    
    print(f"\n[Step 3] Creating crisis ELO evolution plots...")
    create_crisis_elo_plots(elo_df, crisis_df)
    
    print(f"\n[Step 4] Creating crisis comparison analysis...")
    create_crisis_comparison_plot(elo_df, crisis_df)
    
    print(f"\n[Step 5] Creating ELO velocity analysis...")
    create_elo_velocity_plot(elo_df)
    
    print(f"\n[Step 6] Creating crisis summary table...")
    summary_df = create_crisis_summary_table(crisis_df)
    
    print(f"\n[Step 7] Analysis complete!")
    print(f"Generated files:")
    print(f"  - crisis_elo_evolution_detailed.png")
    print(f"  - crisis_elo_comparison_analysis.png") 
    print(f"  - crisis_elo_velocity_analysis.png")
    print(f"  - crisis_performance_summary.csv")

if __name__ == "__main__":
    main()
