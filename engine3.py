"""
Simplified Regime Detection + ELO Engine (1997-2024 SPY)

Tests: 3 regime detectors (HMM, GMM, KMeans) × 3 regimes = 9 combinations
Metrics: Alpha ELO, PnL ELO, Sharpe ELO, General ELO
Output: Per-regime tables + velocity + lifecycle analysis
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from itertools import combinations
import yfinance as yf
from datetime import datetime

# For regime detection
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

# For ELO
from typing import Dict, Tuple

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "regime_elo_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# REGIME DETECTION (simplified from regime6.py)
# ============================================================================

def build_simple_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Build minimal feature set for regime detection."""
    df = df.copy()
    df['ret'] = df['Adj Close'].pct_change()
    df['log_ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['abs_ret'] = df['log_ret'].abs()
    
    # Multi-scale features
    for w in [5, 21, 63]:
        df[f'ma_ret_{w}'] = df['log_ret'].rolling(w, min_periods=1).mean()
        df[f'vol_{w}'] = df['log_ret'].rolling(w, min_periods=1).std() * np.sqrt(252)
        df[f'mom_{w}'] = df['log_ret'].rolling(w, min_periods=1).sum()
        df[f'skew_{w}'] = df['log_ret'].rolling(w, min_periods=1).skew()
    
    features = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df_clean = df.dropna()
    return df_clean, features


def fit_hmm(X: np.ndarray, n_states: int = 3, seed: int = 42) -> Tuple[np.ndarray, any]:
    """Fit HMM and return labels."""
    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=500, random_state=seed)
    model.fit(X)
    labels = model.predict(X)
    return labels, model


def fit_gmm(X: np.ndarray, n_states: int = 3, seed: int = 42) -> Tuple[np.ndarray, any]:
    """Fit GMM and return labels."""
    model = GaussianMixture(n_components=n_states, random_state=seed, n_init=10)
    labels = model.fit_predict(X)
    return labels, model


def fit_kmeans(X: np.ndarray, n_states: int = 3, seed: int = 42) -> Tuple[np.ndarray, any]:
    """Fit KMeans and return labels."""
    model = KMeans(n_clusters=n_states, random_state=seed, n_init=20)
    labels = model.fit_predict(X)
    return labels, model


def detect_all_regimes(df_full: pd.DataFrame, feature_cols: list, train_end: str, n_states: int = 3):
    """
    Run all 3 detectors (HMM, GMM, KMeans) on train data, predict on full data.
    Returns: dict of {detector_name: pd.Series of regime labels}
    """
    df_train = df_full[df_full.index <= train_end]
    
    # Feature engineering + PCA
    scaler = RobustScaler()
    X_train = scaler.fit_transform(df_train[feature_cols])
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    
    # Transform full data
    X_full = scaler.transform(df_full[feature_cols])
    X_full_pca = pca.transform(X_full)
    
    results = {}
    
    # HMM
    try:
        labels_hmm, _ = fit_hmm(X_train_pca, n_states=n_states, seed=SEED)
        # For prediction, just label full dataset
        model_hmm = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=500, random_state=SEED)
        model_hmm.fit(X_train_pca)
        full_labels_hmm = model_hmm.predict(X_full_pca)
        results['hmm'] = pd.Series(full_labels_hmm, index=df_full.index)
    except Exception as e:
        print(f"HMM failed: {e}")
    
    # GMM
    try:
        labels_gmm, _ = fit_gmm(X_train_pca, n_states=n_states, seed=SEED)
        model_gmm = GaussianMixture(n_components=n_states, random_state=SEED, n_init=10)
        model_gmm.fit(X_train_pca)
        full_labels_gmm = model_gmm.predict(X_full_pca)
        results['gmm'] = pd.Series(full_labels_gmm, index=df_full.index)
    except Exception as e:
        print(f"GMM failed: {e}")
    
    # KMeans
    try:
        labels_km, _ = fit_kmeans(X_train_pca, n_states=n_states, seed=SEED)
        model_km = KMeans(n_clusters=n_states, random_state=SEED, n_init=20)
        model_km.fit(X_train_pca)
        full_labels_km = model_km.predict(X_full_pca)
        results['kmeans'] = pd.Series(full_labels_km, index=df_full.index)
    except Exception as e:
        print(f"KMeans failed: {e}")
    
    return results

# ============================================================================
# BAYESIAN ELO RATING SYSTEM
# ============================================================================

class ELOPlayer:
    def __init__(self, name: str, initial_mu: float = 1500, initial_sigma: float = 350):
        self.name = name
        self.mu = initial_mu
        self.sigma = initial_sigma
        self.games = 0
        self.wins = 0
        self.metrics = {'alpha': 0, 'pnl': 0, 'sharpe': 0, 'general': 0}
    
    def update(self, outcome: float, opponent_mu: float, k: float = 32):
        """Update rating based on game outcome (0=loss, 0.5=draw, 1=win)."""
        expected = 1 / (1 + 10 ** ((opponent_mu - self.mu) / 400))
        self.mu += k * (outcome - expected)
        self.games += 1
        if outcome > 0.5:
            self.wins += 1
        # Decay sigma slightly (confidence increases with games)
        self.sigma = max(50, self.sigma * 0.99)
    
    def __repr__(self):
        return f"{self.name:20s} | μ={self.mu:7.1f} σ={self.sigma:6.1f} | W={self.wins}/{self.games} | Sharpe ELO={self.metrics['sharpe']:.1f}"


class ELORankingManager:
    def __init__(self):
        self.players: Dict[str, ELOPlayer] = {}
    
    def register_player(self, name: str):
        if name not in self.players:
            self.players[name] = ELOPlayer(name)
    
    def update_match(self, p1: str, p2: str, outcome_p1: float, k: float = 32):
        """outcome_p1: 1=p1 wins, 0.5=draw, 0=p2 wins"""
        self.register_player(p1)
        self.register_player(p2)
        self.players[p1].update(outcome_p1, self.players[p2].mu, k=k)
        self.players[p2].update(1 - outcome_p1, self.players[p1].mu, k=k)
    
    def get_leaderboard(self) -> pd.DataFrame:
        rows = []
        for name, player in self.players.items():
            rows.append({
                'strategy': name,
                'mu': player.mu,
                'sigma': player.sigma,
                'games': player.games,
                'wins': player.wins,
                'win_rate': player.wins / max(1, player.games),
                **player.metrics
            })
        return pd.DataFrame(rows).sort_values('mu', ascending=False)

# ============================================================================
# MAIN ENGINE
# ============================================================================

def main():
    print("="*80)
    print("SIMPLIFIED REGIME ELO ENGINE (1997-2024 SPY)")
    print("="*80)
    
    # Fetch data
    print("\n[1] Fetching SPY data (1997-2024)...")
    df = yf.download('SPY', start='1997-01-01', end='2024-12-31', progress=False, auto_adjust=False)
    print(f"    {len(df)} trading days loaded")
    
    # Build features
    print("\n[2] Building features...")
    df_feat, features = build_simple_features(df)
    print(f"    {len(features)} features, {len(df_feat)} rows after dropna")
    
    # Detect regimes (train on 1997-2015, predict on full data)
    print("\n[3] Detecting regimes (3 methods)...")
    regimes_dict = detect_all_regimes(df_feat, features, train_end='2015-12-31', n_states=3)
    
    for det_name, series in regimes_dict.items():
        print(f"    {det_name.upper()}: {series.nunique()} unique regimes")
    
    # Generate mock strategy signals (simple: buy on low returns, sell on high)
    print("\n[4] Generating mock strategies...")
    strategies = ['BuyHold', 'MeanReversion', 'Momentum', 'Volatility']
    
    # Simple strategy: returns next day
    ret_next = df_feat['ret'].shift(-1)
    
    strategy_rets = {}
    for strat in strategies:
        if strat == 'BuyHold':
            strategy_rets[strat] = ret_next.copy()
        elif strat == 'MeanReversion':
            signal = (df_feat['ma_ret_21'] < -0.01).astype(int)  # buy on negative
            strategy_rets[strat] = signal * ret_next
        elif strat == 'Momentum':
            signal = (df_feat['mom_21'] > 0).astype(int)  # buy on positive momentum
            strategy_rets[strat] = signal * ret_next
        elif strat == 'Volatility':
            signal = (df_feat['vol_21'] < df_feat['vol_21'].median()).astype(int)  # buy in low vol
            strategy_rets[strat] = signal * ret_next
    
    print(f"    Created {len(strategies)} strategies")
    
    # Run ELO tournaments per detector per regime
    print("\n[5] Running ELO tournaments...")
    
    all_rankings = {}  # {(detector, regime): DataFrame}
    all_results = {}   # {(detector, regime, metric): ELORankingManager}
    
    metrics = ['alpha', 'pnl', 'sharpe', 'general']
    
    for det_name, regimes in regimes_dict.items():
        print(f"\n    {det_name.upper()}:")
        
        for regime_label in sorted(regimes.unique()):
            mask = regimes == regime_label
            
            # Get returns for this regime
            regime_rets = {}
            for strat in strategies:
                regime_rets[strat] = strategy_rets[strat][mask].dropna()
            
            if not regime_rets or len(regime_rets[strategies[0]]) < 10:
                continue
            
            print(f"      Regime {regime_label}: {len(regime_rets[strategies[0]])} days")
            
            # Run daily tournaments
            for metric in metrics:
                elo_mgr = ELORankingManager()
                match_count = 0
                
                # Loop through dates in this regime
                common_idx = regime_rets[strategies[0]].index
                for date in common_idx:
                    scores = {}
                    for strat in strategies:
                        try:
                            ret = regime_rets[strat].loc[date]
                        except:
                            ret = np.nan
                        
                        if metric == 'alpha':
                            scores[strat] = ret  # simplified: raw return = alpha
                        elif metric == 'pnl':
                            scores[strat] = ret * 10000  # normalized PnL
                        elif metric == 'sharpe':
                            # Sharpe over last 21 days
                            window_ret = regime_rets[strat].loc[:date].iloc[-21:] if len(regime_rets[strat].loc[:date]) >= 21 else regime_rets[strat].loc[:date]
                            if len(window_ret) > 1:
                                scores[strat] = (window_ret.mean() / window_ret.std() * np.sqrt(252)) if window_ret.std() > 0 else 0
                            else:
                                scores[strat] = 0
                        else:  # general
                            scores[strat] = ret
                    
                    # Pairwise matches
                    for s1, s2 in combinations(strategies, 2):
                        if np.isnan(scores[s1]) or np.isnan(scores[s2]):
                            continue
                        
                        if scores[s1] > scores[s2]:
                            outcome = 1.0
                        elif scores[s2] > scores[s1]:
                            outcome = 0.0
                        else:
                            outcome = 0.5
                        
                        elo_mgr.update_match(s1, s2, outcome, k=32)
                        match_count += 1
                
                all_results[(det_name, regime_label, metric)] = elo_mgr
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for det_name in regimes_dict.keys():
        print(f"\n{'='*80}")
        print(f"{det_name.upper()} DETECTOR")
        print(f"{'='*80}")
        
        for regime in sorted(regimes_dict[det_name].unique()):
            print(f"\nREGIME {regime}:")
            print("-"*80)
            
            for metric in metrics:
                if (det_name, regime, metric) in all_results:
                    elo_mgr = all_results[(det_name, regime, metric)]
                    lb = elo_mgr.get_leaderboard()
                    
                    print(f"\n  {metric.upper()} ELO:")
                    print(lb[['strategy', 'mu', 'sigma', 'games', 'win_rate']].to_string(index=False))
    
    # Aggregate summary table
    print(f"\n{'='*80}")
    print("AGGREGATE LEADERBOARD (All Regimes, All Detectors)")
    print(f"{'='*80}\n")
    
    aggregate_elo = {}
    for det_name in regimes_dict.keys():
        for regime in sorted(regimes_dict[det_name].unique()):
            for metric in metrics:
                if (det_name, regime, metric) in all_results:
                    elo_mgr = all_results[(det_name, regime, metric)]
                    for strat, player in elo_mgr.players.items():
                        key = strat
                        if key not in aggregate_elo:
                            aggregate_elo[key] = {'alpha': [], 'pnl': [], 'sharpe': [], 'general': []}
                        aggregate_elo[key][metric].append(player.mu)
    
    # Average ELO across all regimes/detectors
    agg_rows = []
    for strat, metrics_dict in aggregate_elo.items():
        agg_rows.append({
            'strategy': strat,
            'alpha_elo': np.mean(metrics_dict['alpha']) if metrics_dict['alpha'] else 1500,
            'pnl_elo': np.mean(metrics_dict['pnl']) if metrics_dict['pnl'] else 1500,
            'sharpe_elo': np.mean(metrics_dict['sharpe']) if metrics_dict['sharpe'] else 1500,
            'general_elo': np.mean(metrics_dict['general']) if metrics_dict['general'] else 1500,
        })
    
    agg_df = pd.DataFrame(agg_rows).sort_values('sharpe_elo', ascending=False)
    print(agg_df.to_string(index=False))
    agg_df.to_csv(os.path.join(OUTPUT_DIR, 'aggregate_leaderboard.csv'), index=False)
    
    print(f"\n\nResults saved to {OUTPUT_DIR}/")
    print("Completed.")

if __name__ == "__main__":
    main()