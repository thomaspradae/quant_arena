"""
Complete Regime Detection + ELO Analysis Pipeline
Runs on S&P 500 data from 2014-2024
Generates all visualizations and results in one go

Run this as a standalone script:
    python regime_elo_complete.py

Outputs:
- Console validation reports
- CSV files with results
- PNG visualizations
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.stats import spearmanr

# Data fetching
try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    exit(1)

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("WARNING: hmmlearn not available. Install with: pip install hmmlearn")

# Setup
sns.set_style("darkgrid")
OUTPUT_DIR = "regime_elo_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKER = "SPY"
START_DATE = "2014-01-01"
END_DATE = "2024-12-31"


# ============================================================================
# DATA & FEATURE ENGINEERING
# ============================================================================

def fetch_market_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    """Download OHLCV data from Yahoo Finance."""
    print(f"Fetching {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    df.index = pd.to_datetime(df.index)
    print(f"  → Downloaded {len(df)} days")
    return df


def build_features(df):
    """
    Build comprehensive multi-scale technical features.
    Returns (feature_df, feature_names)
    """
    df = df.copy()
    
    # Returns
    df['log_ret'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    df['ret'] = df['adj_close'].pct_change()
    df['abs_ret'] = df['log_ret'].abs()
    
    # GOLDILOCKS: Mix of medium-term (capture regimes) + some short-term (capture transitions)
    # Medium-term returns (main signal)
    for window in [21, 42, 63]:
        df[f'ma_r_{window}'] = df['log_ret'].rolling(window=window, min_periods=max(5, window//3)).mean()
    
    # Volatility: short + medium + long (capture vol regime shifts)
    for window in [10, 21, 63]:
        df[f'vol_{window}'] = df['log_ret'].rolling(window=window, min_periods=max(5, window//3)).std() * np.sqrt(252)
    
    # Momentum (medium-term)
    for window in [21, 63]:
        df[f'mom_{window}'] = df['log_ret'].rolling(window=window, min_periods=max(5, window//3)).sum()
    
    # Higher moments (capture distribution changes)
    df['skew_21'] = df['log_ret'].rolling(window=21, min_periods=10).skew()
    df['kurt_21'] = df['log_ret'].rolling(window=21, min_periods=10).kurt()
    
    # ATR (medium-term)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['adj_close'].shift(1)).abs()
    low_close = (df['low'] - df['adj_close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = (tr.rolling(14, min_periods=7).mean().values / df['adj_close'].values)
    
    # Volume ratio (medium-term)
    df['vol_ratio_21'] = df['volume'] / df['volume'].rolling(21, min_periods=10).mean()
    
    # Vol regime indicator (short vs medium-term vol)
    realized_vol_10 = df['log_ret'].rolling(10, min_periods=5).std() * np.sqrt(252)
    realized_vol_63 = df['log_ret'].rolling(63, min_periods=30).std() * np.sqrt(252)
    df['vol_regime'] = realized_vol_10 / realized_vol_63
    
    # Recent trend (captures directional moves)
    df['trend_21'] = (df['adj_close'] / df['adj_close'].shift(21) - 1)
    
    # Feature list: BALANCED mix
    feature_cols = ['ma_r_21', 'ma_r_42', 'ma_r_63',
                   'vol_10', 'vol_21', 'vol_63',
                   'mom_21', 'mom_63',
                   'skew_21', 'kurt_21',
                   'atr', 'vol_ratio_21', 'vol_regime', 'trend_21']
    
    # Drop NaN
    df_clean = df.dropna(subset=feature_cols)
    
    return df_clean, feature_cols


# ============================================================================
# HMM REGIME DETECTION WITH MODEL SELECTION
# ============================================================================

def select_best_hmm(X, n_states_range=[2, 3, 4], cov_types=['full', 'diag'], 
                    n_restarts=5, n_iter=200, criterion='bic'):
    """
    Grid search HMM with BIC/AIC selection.
    Returns (results_df, best_model, best_params)
    """
    if not HMM_AVAILABLE:
        raise RuntimeError("hmmlearn not installed. Run: pip install hmmlearn")
    
    n_obs, n_feat = X.shape
    results = []
    best_score = np.inf
    best_model = None
    best_params = None
    
    print(f"Running HMM grid search: {len(n_states_range)} x {len(cov_types)} = {len(n_states_range)*len(cov_types)} configs")
    
    for n_states in n_states_range:
        for cov_type in cov_types:
            best_ll = -np.inf
            best_local = None
            
            for seed in range(n_restarts):
                try:
                    model = GaussianHMM(
                        n_components=n_states,
                        covariance_type=cov_type,
                        n_iter=n_iter,
                        random_state=42 + seed,
                        verbose=False
                    )
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X)
                    
                    ll = model.score(X)
                    
                    if ll > best_ll:
                        best_ll = ll
                        best_local = model
                
                except Exception:
                    continue
            
            if best_local is not None:
                # Count parameters
                k = (n_states * n_feat + 
                     (n_states * n_feat * (n_feat + 1) // 2 if cov_type == 'full' else n_states * n_feat) +
                     n_states * (n_states - 1) + (n_states - 1))
                
                aic = 2 * k - 2 * best_ll
                bic = k * np.log(n_obs) - 2 * best_ll
                
                score = bic if criterion == 'bic' else aic
                
                results.append({
                    'n_states': n_states,
                    'cov_type': cov_type,
                    'log_likelihood': best_ll,
                    'k_params': k,
                    'aic': aic,
                    'bic': bic
                })
                
                if score < best_score:
                    best_score = score
                    best_model = best_local
                    best_params = {'n_states': n_states, 'cov_type': cov_type}
    
    if not results:
        raise RuntimeError("No HMM models converged")
    
    df_results = pd.DataFrame(results).sort_values(criterion)
    
    return df_results, best_model, best_params


# ============================================================================
# VALIDATION METRICS
# ============================================================================

def compute_stability_metrics(regimes):
    """Compute regime persistence metrics."""
    # Duration
    runs = []
    current = regimes.iloc[0]
    current_run = 1
    
    for i in range(1, len(regimes)):
        if regimes.iloc[i] == current:
            current_run += 1
        else:
            runs.append(current_run)
            current = regimes.iloc[i]
            current_run = 1
    runs.append(current_run)
    
    mean_duration = float(np.mean(runs))
    
    # Transition rate
    transitions = (regimes.diff() != 0).sum()
    transition_rate = float(transitions / len(regimes))
    
    # Entropy
    counts = regimes.value_counts()
    probs = counts / len(regimes)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(counts))
    normalized_entropy = float(entropy / max_entropy if max_entropy > 0 else 0)
    
    return {
        'mean_duration_days': mean_duration,
        'transition_rate': transition_rate,
        'normalized_entropy': normalized_entropy
    }


def detect_crash_regime(regimes, returns):
    """Identify crash regime (lowest mean return)."""
    avg_by_regime = {}
    for r in regimes.unique():
        mask = regimes == r
        avg_by_regime[r] = returns[mask].mean()
    
    crash_regime = min(avg_by_regime, key=avg_by_regime.get)
    return crash_regime, avg_by_regime


def compute_regime_stats(regimes, returns):
    """Compute per-regime statistics."""
    stats = []
    for regime in sorted(regimes.unique()):
        mask = regimes == regime
        regime_rets = returns[mask]
        
        stats.append({
            'regime': regime,
            'n_days': len(regime_rets),
            'pct_days': len(regime_rets) / len(regimes) * 100,
            'mean_return_daily': regime_rets.mean(),
            'volatility_annual': regime_rets.std() * np.sqrt(252),
            'sharpe_annual': (regime_rets.mean() * 252) / (regime_rets.std() * np.sqrt(252)) if regime_rets.std() > 0 else 0
        })
    
    return pd.DataFrame(stats)


# ============================================================================
# SIMPLE STRATEGY SIMULATION FOR DEMO
# ============================================================================

def create_demo_strategies(df):
    """
    Create simple demo strategies to test regime-conditional performance.
    Returns dict of {strategy_name: returns_series}
    """
    strategies = {}
    
    # 1. Buy and Hold
    strategies['BuyAndHold'] = df['log_ret']
    
    # 2. SMA Crossover (20/50)
    sma_20 = df['adj_close'].rolling(20).mean()
    sma_50 = df['adj_close'].rolling(50).mean()
    signal = (sma_20 > sma_50).astype(int)
    signal = signal.shift(1).fillna(0)
    strategies['SMA_20_50'] = signal * df['log_ret']
    
    # 3. Mean Reversion (Bollinger)
    sma = df['adj_close'].rolling(20).mean()
    std = df['adj_close'].rolling(20).std()
    z_score = (df['adj_close'] - sma) / std
    signal = -np.sign(z_score).shift(1).fillna(0)  # Mean revert
    strategies['Bollinger_MR'] = signal * df['log_ret']
    
    # 4. Volatility targeting
    vol_target = 0.15
    realized_vol = df['log_ret'].rolling(21).std() * np.sqrt(252)
    leverage = (vol_target / realized_vol).clip(0, 2).shift(1).fillna(1)
    strategies['Vol_Target'] = leverage * df['log_ret']
    
    # 5. Momentum
    mom_signal = np.sign(df['adj_close'].rolling(63).mean() - df['adj_close'].shift(63)).shift(1).fillna(0)
    strategies['Momentum_63d'] = mom_signal * df['log_ret']
    
    return strategies


def compute_regime_elo_variance(regimes, strategies):
    """
    Compute how much each strategy's performance varies across regimes.
    This is the KEY METRIC for specialist vs generalist.
    """
    results = []
    
    for name, returns in strategies.items():
        regime_perf = {}
        
        for regime in sorted(regimes.unique()):
            mask = regimes == regime
            regime_returns = returns[mask]
            if len(regime_returns) > 10:
                regime_perf[regime] = regime_returns.mean() * 252  # Annualized
        
        if len(regime_perf) >= 2:
            perfs = list(regime_perf.values())
            variance = np.std(perfs)
            regime_range = max(perfs) - min(perfs)
            
            # Classification
            if variance > 0.20:
                classification = 'Specialist'
            elif variance > 0.10:
                classification = 'Moderate'
            else:
                classification = 'Generalist'
            
            results.append({
                'strategy': name,
                'regime_variance': variance,
                'regime_range': regime_range,
                'classification': classification,
                **{f'regime_{k}_return': v for k, v in regime_perf.items()}
            })
    
    return pd.DataFrame(results).sort_values('regime_variance', ascending=False)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_price_with_regimes(df, regimes, save_path):
    """Plot price colored by regime."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    
    for regime in sorted(regimes.unique()):
        mask = regimes == regime
        dates = regimes[mask].index
        prices = df.loc[dates, 'adj_close']
        ax.scatter(dates, prices, c=colors[regime], s=8, alpha=0.6, label=f'Regime {regime}')
    
    ax.plot(df.index, df['adj_close'], 'k-', linewidth=0.5, alpha=0.3, zorder=0)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('S&P 500 Price', fontsize=12)
    ax.set_title('S&P 500 Price with Detected Regimes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_regime_stats(regime_stats, save_path):
    """Plot regime statistics bars."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Days
    axes[0, 0].bar(regime_stats['regime'], regime_stats['pct_days'], color='steelblue')
    axes[0, 0].set_title('% Days in Each Regime', fontweight='bold')
    axes[0, 0].set_xlabel('Regime')
    axes[0, 0].set_ylabel('% of Days')
    
    # Mean return
    colors = ['green' if x > 0 else 'red' for x in regime_stats['mean_return_daily']]
    axes[0, 1].bar(regime_stats['regime'], regime_stats['mean_return_daily'] * 252, color=colors)
    axes[0, 1].set_title('Annualized Return by Regime', fontweight='bold')
    axes[0, 1].set_xlabel('Regime')
    axes[0, 1].set_ylabel('Return')
    axes[0, 1].axhline(0, color='black', linewidth=0.5)
    
    # Volatility
    axes[1, 0].bar(regime_stats['regime'], regime_stats['volatility_annual'], color='orange')
    axes[1, 0].set_title('Annualized Volatility by Regime', fontweight='bold')
    axes[1, 0].set_xlabel('Regime')
    axes[1, 0].set_ylabel('Volatility')
    
    # Sharpe
    axes[1, 1].bar(regime_stats['regime'], regime_stats['sharpe_annual'], color='purple')
    axes[1, 1].set_title('Sharpe Ratio by Regime', fontweight='bold')
    axes[1, 1].set_xlabel('Regime')
    axes[1, 1].set_ylabel('Sharpe')
    axes[1, 1].axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_regime_variance_heatmap(variance_df, save_path):
    """Heatmap of strategy performance by regime."""
    # Get regime columns
    regime_cols = [c for c in variance_df.columns if c.startswith('regime_') and c.endswith('_return')]
    
    if not regime_cols:
        print("  → No regime columns for heatmap")
        return
    
    data = variance_df[['strategy'] + regime_cols].set_index('strategy')
    data.columns = [c.replace('regime_', 'R').replace('_return', '') for c in data.columns]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Annualized Return'}, ax=ax)
    ax.set_title('Strategy Returns by Regime (Annualized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Regime', fontsize=12)
    ax.set_ylabel('Strategy', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_cumulative_returns(strategies, regimes, save_path):
    """Plot cumulative returns with regime backgrounds."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Regime backgrounds
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    for regime in sorted(regimes.unique()):
        mask = regimes == regime
        spans = []
        start = None
        for i, (idx, val) in enumerate(mask.items()):
            if val and start is None:
                start = idx
            elif not val and start is not None:
                spans.append((start, idx))
                start = None
        if start is not None:
            spans.append((start, mask.index[-1]))
        
        for span_start, span_end in spans:
            ax.axvspan(span_start, span_end, alpha=0.15, color=colors[regime], zorder=0)
    
    # Plot strategies
    for name, returns in strategies.items():
        cum_ret = returns.cumsum()
        ax.plot(cum_ret.index, cum_ret.values, linewidth=1.5, label=name, alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Log Return', fontsize=12)
    ax.set_title('Strategy Performance Across Regimes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*90)
    print("REGIME DETECTION + STRATEGY ANALYSIS PIPELINE")
    print("S&P 500: 2014-2024")
    print("="*90 + "\n")
    
    # 1. Fetch data
    df = fetch_market_data(TICKER, START_DATE, END_DATE)
    
    # 2. Build features
    print("\nBuilding features...")
    df_feat, feature_names = build_features(df)
    print(f"  → {len(feature_names)} features: {feature_names[:5]}...")
    print(f"  → {len(df_feat)} days after dropna")
    
    # 3. Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat[feature_names].values)
    print(f"  → Feature matrix shape: {X.shape}")
    
    # 4. HMM model selection
    print("\n" + "-"*90)
    print("HMM MODEL SELECTION")
    print("-"*90)
    
    model_comparison, best_model, best_params = select_best_hmm(
        X,
        n_states_range=[2, 3],  # Limit to 2-3 states to avoid overfitting
        cov_types=['full', 'diag'],
        n_restarts=5,
        n_iter=300,
        criterion='bic'
    )
    
    print(f"\nTop 5 models by BIC:")
    print(model_comparison[['n_states', 'cov_type', 'bic', 'aic']].head().to_string(index=False))
    
    print(f"\n✓ Best model: {best_params['n_states']} states, {best_params['cov_type']} covariance")
    print(f"  BIC: {model_comparison.iloc[0]['bic']:.2f}")
    
    # Save model comparison
    model_comparison.to_csv(os.path.join(OUTPUT_DIR, 'hmm_model_comparison.csv'), index=False)
    
    # 5. Predict regimes
    print("\nPredicting regimes...")
    regime_labels = best_model.predict(X)
    regimes = pd.Series(regime_labels, index=df_feat.index, name='regime')
    print(f"  → Detected regimes: {sorted(regimes.unique())}")
    
    # Save regimes
    regimes.to_csv(os.path.join(OUTPUT_DIR, 'detected_regimes.csv'))
    
    # 6. Stability validation
    print("\n" + "-"*90)
    print("REGIME VALIDATION")
    print("-"*90)
    
    stability = compute_stability_metrics(regimes)
    
    print(f"\nStability Metrics:")
    print(f"  Mean Duration: {stability['mean_duration_days']:.1f} days")
    if 20 < stability['mean_duration_days'] < 90:
        print("    ✓ GOOD: Persistent but not static")
    elif stability['mean_duration_days'] < 10:
        print("    ❌ WARNING: Regimes too volatile (daily flickering)")
        print("       → Consider using longer-term features or smoothing")
    else:
        print("    ⚠ Regimes may be too static")
    
    print(f"  Transition Rate: {stability['transition_rate']:.1%}")
    if 0.01 < stability['transition_rate'] < 0.10:
        print("    ✓ GOOD: Stable with occasional shifts")
    
    print(f"  Entropy: {stability['normalized_entropy']:.2f}")
    print(f"    (0=all one regime, 1=uniform)")
    
    # 7. Regime statistics
    print("\n" + "-"*90)
    print("PER-REGIME STATISTICS")
    print("-"*90)
    
    regime_stats = compute_regime_stats(regimes, df_feat['log_ret'])
    print(f"\n{regime_stats.to_string(index=False)}")
    regime_stats.to_csv(os.path.join(OUTPUT_DIR, 'regime_statistics.csv'), index=False)
    
    # Identify crash regime
    crash_regime, avg_by_regime = detect_crash_regime(regimes, df_feat['log_ret'])
    print(f"\nCrash Regime Identification:")
    print(f"  → Regime {crash_regime} (lowest mean return: {avg_by_regime[crash_regime]*252:.2%} annual)")
    
    # 8. Create demo strategies
    print("\n" + "-"*90)
    print("STRATEGY SIMULATION")
    print("-"*90)
    
    print("\nCreating demo strategies...")
    strategies = create_demo_strategies(df_feat)
    print(f"  → {len(strategies)} strategies created")
    
    # Overall performance
    print("\nOverall Strategy Performance (2014-2024):")
    overall_stats = []
    for name, returns in strategies.items():
        total_return = returns.sum()
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        overall_stats.append({
            'strategy': name,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe
        })
    
    overall_df = pd.DataFrame(overall_stats).sort_values('sharpe', ascending=False)
    print(overall_df.to_string(index=False))
    overall_df.to_csv(os.path.join(OUTPUT_DIR, 'strategy_overall_performance.csv'), index=False)
    
    # 9. Regime-conditional analysis (THE KEY FINDING)
    print("\n" + "="*90)
    print("REGIME IMPACT ANALYSIS: Specialist vs Generalist")
    print("="*90)
    
    variance_df = compute_regime_elo_variance(regimes, strategies)
    
    print(f"\n{variance_df.to_string(index=False)}")
    variance_df.to_csv(os.path.join(OUTPUT_DIR, 'regime_impact_analysis.csv'), index=False)
    
    print("\n" + "-"*90)
    print("INTERPRETATION:")
    print("-"*90)
    
    most_specialist = variance_df.iloc[0]
    most_generalist = variance_df.iloc[-1]
    
    print(f"\n✓ Most Specialist: {most_specialist['strategy']}")
    print(f"  • Variance: {most_specialist['regime_variance']:.3f}")
    print(f"  • Range: {most_specialist['regime_range']:.3f}")
    
    print(f"\n✓ Most Generalist: {most_generalist['strategy']}")
    print(f"  • Variance: {most_generalist['regime_variance']:.3f}")
    print(f"  • Range: {most_generalist['regime_range']:.3f}")
    
    mean_var = variance_df['regime_variance'].mean()
    print(f"\n✓ Mean regime variance: {mean_var:.3f}")
    print(f"  → Traditional metrics assume variance = 0")
    print(f"  → Our system reveals {mean_var:.1%} hidden context-dependence")
    
    # 10. Generate visualizations
    print("\n" + "-"*90)
    print("GENERATING VISUALIZATIONS")
    print("-"*90 + "\n")
    
    plot_price_with_regimes(
        df_feat, regimes, 
        os.path.join(OUTPUT_DIR, '1_price_with_regimes.png')
    )
    
    plot_regime_stats(
        regime_stats,
        os.path.join(OUTPUT_DIR, '2_regime_statistics.png')
    )
    
    plot_regime_variance_heatmap(
        variance_df,
        os.path.join(OUTPUT_DIR, '3_regime_variance_heatmap.png')
    )
    
    plot_cumulative_returns(
        strategies, regimes,
        os.path.join(OUTPUT_DIR, '4_cumulative_returns_by_regime.png')
    )
    
    # 11. Summary
    print("\n" + "="*90)
    print("PIPELINE COMPLETE")
    print("="*90)
    
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("  • hmm_model_comparison.csv")
    print("  • detected_regimes.csv")
    print("  • regime_statistics.csv")
    print("  • strategy_overall_performance.csv")
    print("  • regime_impact_analysis.csv")
    print("  • 4 PNG visualizations")
    
    print("\n" + "="*90)
    print("KEY FINDINGS FOR PRESENTATION:")
    print("="*90)
    print(f"1. Detected {best_params['n_states']} market regimes (BIC-selected)")
    print(f"2. Regimes persist {stability['mean_duration_days']:.1f} days on average")
    print(f"3. Crash regime (#{crash_regime}) has {avg_by_regime[crash_regime]*252:.1%} annual return")
    print(f"4. Strategy variance across regimes: {mean_var:.1%} (context-dependence)")
    print(f"5. Most specialist: {most_specialist['strategy']} (variance={most_specialist['regime_variance']:.3f})")
    print(f"6. Most generalist: {most_generalist['strategy']} (variance={most_generalist['regime_variance']:.3f})")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()