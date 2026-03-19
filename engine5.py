"""
Full ELO Engine with Validation Tests
- Uses existing RankingManager and Matchmaker classes
- No look-ahead bias
- Regime-conditional ranking variance
- Regime ranking stability (Spearman)
- ELO momentum predictive test
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from itertools import combinations
import yfinance as yf
from typing import Dict, Tuple, List, Optional
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "regime_elo_full_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# REGIME DETECTION
# ============================================================================

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Build feature set"""
    df = df.copy()
    df['ret'] = df['Adj Close'].pct_change()
    df['log_ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['abs_ret'] = df['log_ret'].abs()
    
    for w in [5, 21, 63]:
        df[f'ma_ret_{w}'] = df['log_ret'].rolling(w, min_periods=1).mean()
        df[f'vol_{w}'] = df['log_ret'].rolling(w, min_periods=1).std() * np.sqrt(252)
        df[f'mom_{w}'] = df['log_ret'].rolling(w, min_periods=1).sum()
        df[f'skew_{w}'] = df['log_ret'].rolling(w, min_periods=1).skew()
    
    features = [c for c in df.columns if c not in 
                ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    return df.dropna(), features


def detect_regimes_ensemble(df: pd.DataFrame, features: List[str], 
                            train_end: str = '2015-12-31', n_states: int = 3):
    """Detect regimes with 3 methods and use majority voting"""
    df_train = df[df.index <= train_end]
    
    scaler = RobustScaler()
    X_train = scaler.fit_transform(df_train[features])
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    
    X_full = scaler.transform(df[features])
    X_full_pca = pca.transform(X_full)
    
    regimes_dict = {}
    
    # HMM
    try:
        model_hmm = GaussianHMM(n_components=n_states, covariance_type='diag', 
                                n_iter=500, random_state=SEED)
        model_hmm.fit(X_train_pca)
        regimes_dict['hmm'] = pd.Series(model_hmm.predict(X_full_pca), index=df.index)
    except:
        pass
    
    # GMM
    try:
        model_gmm = GaussianMixture(n_components=n_states, random_state=SEED, n_init=10)
        model_gmm.fit(X_train_pca)
        regimes_dict['gmm'] = pd.Series(model_gmm.predict(X_full_pca), index=df.index)
    except:
        pass
    
    # KMeans
    try:
        model_km = KMeans(n_clusters=n_states, random_state=SEED, n_init=20)
        model_km.fit(X_train_pca)
        regimes_dict['kmeans'] = pd.Series(model_km.predict(X_full_pca), index=df.index)
    except:
        pass
    
    # Ensemble via majority voting
    if regimes_dict:
        ensemble = pd.concat(regimes_dict.values(), axis=1).mode(axis=1)[0]
        ensemble.index = df.index
        return ensemble, regimes_dict
    else:
        return pd.Series(0, index=df.index), {}


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def compute_regime_variance(ranking_manager, all_regimes: List[int]) -> pd.DataFrame:
    """
    Test 1: Compute variance of ELO ratings across regimes.
    High variance = regime specialist. Low variance = generalist.
    """
    rows = []
    
    # Get global leaderboard to find all strategies
    global_lb = ranking_manager.get_leaderboard(metric='alpha', regime=None)
    if global_lb.empty:
        return pd.DataFrame()
    
    for strat_name in global_lb['strategy'].values:
        elos_per_regime = []
        for regime in all_regimes:
            try:
                rating = ranking_manager.rankings['alpha'][regime].get_rating(strat_name)
                if not np.isnan(rating):
                    elos_per_regime.append(rating)
            except:
                pass
        
        if elos_per_regime:
            variance = np.std(elos_per_regime)
            mean_elo = np.mean(elos_per_regime)
            rows.append({
                'strategy': strat_name,
                'variance_across_regimes': variance,
                'mean_elo': mean_elo,
                'type': 'specialist' if variance > 100 else 'generalist'
            })
    
    return pd.DataFrame(rows).sort_values('variance_across_regimes', ascending=False)


def compute_ranking_stability(ranking_manager, all_regimes: List[int]) -> pd.DataFrame:
    """
    Test 2: Spearman rank correlation between regimes.
    Low correlation = regimes have different winners.
    """
    regime_rankings = {}
    for regime in all_regimes:
        try:
            lb = ranking_manager.get_leaderboard(metric='alpha', regime=regime)
            if not lb.empty:
                ranked = lb['strategy'].tolist()
                regime_rankings[int(regime)] = ranked
        except:
            pass
    
    correlations = []
    regime_pairs = list(combinations(sorted(regime_rankings.keys()), 2))
    
    for r1, r2 in regime_pairs:
        ranks1 = regime_rankings[r1]
        ranks2 = regime_rankings[r2]
        
        common = list(set(ranks1) & set(ranks2))
        if len(common) < 2:
            continue
        
        pos1 = [ranks1.index(s) for s in common]
        pos2 = [ranks2.index(s) for s in common]
        
        rho, pval = spearmanr(pos1, pos2)
        correlations.append({
            'regime_pair': f"{int(r1)} vs {int(r2)}",
            'spearman_rho': rho,
            'p_value': pval,
            'significant': pval < 0.05
        })
    
    return pd.DataFrame(correlations)


def test_elo_momentum_predictive(ranking_manager, strategy_rets: Dict[str, pd.Series],
                                 W: int = 20, T: int = 20) -> Dict:
    """
    Test 3: Does ELO momentum predict future returns?
    Regression: future_return_t = alpha + beta * elo_momentum_t
    """
    results = {}
    
    # Get rating history from ranking manager
    global_leaderboard = ranking_manager.get_leaderboard(metric='alpha', regime=None)
    if global_leaderboard.empty:
        return results
    
    for strat_name in global_leaderboard['strategy'].values:
        if strat_name not in strategy_rets:
            continue
        
        ret_series = strategy_rets[strat_name]
        
        # Get rating history
        try:
            rating_history = ranking_manager.rankings['alpha'][None].get_rating_history(strat_name)
            if rating_history.empty or len(rating_history) < W + T:
                continue
        except:
            continue
        
        # Convert to numeric index (dates to position indices)
        rating_vals = rating_history.values
        
        # Compute ELO momentum
        elo_momentum = np.diff(rating_vals, n=W)
        elo_momentum = np.concatenate([[np.nan] * W, elo_momentum])
        
        # Match with future returns (no look-ahead)
        matched_rows = []
        for idx in range(len(elo_momentum)):
            if np.isnan(elo_momentum[idx]):
                continue
            
            # Future returns (from idx+1 to idx+T)
            future_end = min(idx + T, len(ret_series))
            if future_end <= idx + 1:
                continue
            
            fwd_ret = ret_series.iloc[idx + 1:future_end].sum()
            
            matched_rows.append({
                'elo_momentum': elo_momentum[idx],
                'fwd_return': fwd_ret
            })
        
        if len(matched_rows) < 10:
            continue
        
        reg_df = pd.DataFrame(matched_rows).dropna()
        if len(reg_df) < 10:
            continue
        
        # OLS regression
        X = reg_df['elo_momentum'].values
        y = reg_df['fwd_return'].values
        
        X = np.column_stack([np.ones(len(X)), X])
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            continue
        
        # R-squared
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # T-stat for slope
        residuals = y - y_pred
        mse = np.sum(residuals**2) / max(1, len(y) - 2)
        try:
            var_coef = mse * np.linalg.inv(X.T @ X)[1, 1]
            se = np.sqrt(var_coef) if var_coef > 0 else 1
            t_stat = beta[1] / se if se > 0 else 0
        except:
            t_stat = 0
        
        # Approximate p-value
        p_val = 2 * (1 - min(1, np.abs(t_stat) / 10))  # rough approximation
        
        results[strat_name] = {
            'n_obs': len(reg_df),
            'alpha': beta[0],
            'beta': beta[1],
            't_stat': t_stat,
            'p_value': p_val,
            'r2': r2,
            'significant': abs(t_stat) > 2.0
        }
    
    return results


# ============================================================================
# MAIN ENGINE
# ============================================================================

def main():
    print("="*100)
    print("FULL BAYESIAN ELO ENGINE WITH VALIDATION")
    print("="*100)
    
    # Import ranking and matchmaker (you already have these)
    from ranking import RankingManager, BayesianELORanking
    from matchmaker import RoundRobinMatcher
    
    # Fetch data
    print("\n[1] Fetching SPY data...")
    df = yf.download('SPY', start='1997-01-01', end='2024-12-31', 
                     progress=False, auto_adjust=False)
    print(f"    {len(df)} trading days loaded")
    
    # Build features
    print("\n[2] Building features...")
    df_feat, features = build_features(df)
    print(f"    {len(features)} features, {len(df_feat)} rows")
    
    # Detect regimes
    print("\n[3] Detecting regimes (ensemble of HMM, GMM, KMeans)...")
    regimes, per_method = detect_regimes_ensemble(df_feat, features, 
                                                  train_end='2015-12-31', n_states=3)
    unique_regimes = sorted([int(r) for r in regimes.unique()])
    print(f"    Regimes detected: {unique_regimes}")
    
    # Generate strategy returns
    print("\n[4] Generating strategy returns...")
    strategy_names = ['BuyAndHold', 'SMAXover_20_50', 'SMAXover_50_200', 
                      'BollingerMR', 'RSI_14']
    
    ret_next = df_feat['ret'].shift(-1)
    strategy_rets = {}
    
    for strat_name in strategy_names:
        if strat_name == 'BuyAndHold':
            strategy_rets[strat_name] = ret_next.copy()
        elif 'SMAXover' in strat_name:
            signal = (df_feat['mom_21'] > 0).astype(int)
            strategy_rets[strat_name] = signal * ret_next
        elif 'BollingerMR' in strat_name:
            signal = (df_feat['ma_ret_21'] < -0.01).astype(int)
            strategy_rets[strat_name] = signal * ret_next
        elif 'RSI' in strat_name:
            signal = (df_feat['mom_21'] > 0).astype(int)
            strategy_rets[strat_name] = signal * ret_next
    
    print(f"    Created {len(strategy_rets)} strategies")
    
    # Initialize ranking manager (with BayesianELORanking)
    print("\n[5] Initializing Bayesian ELO ranking manager...")
    ranking_manager = RankingManager(
        ranking_class=BayesianELORanking,
        initial_mu=1500,
        initial_sigma=350,
        min_sigma=50,
        tau=1.0,
        base_k=32
    )
    
    # Run daily ELO tournaments with NO LOOK-AHEAD
    print("\n[6] Running daily ELO tournaments (no look-ahead)...")
    
    common_idx = df_feat.index
    match_count = 0
    
    for date_idx, date in enumerate(common_idx):
        # Get regime for today
        try:
            regime_label = int(regimes.loc[date])
        except:
            regime_label = None
        
        # Get returns for today
        scores = {}
        for strat_name, ret_series in strategy_rets.items():
            try:
                score = float(ret_series.loc[date])
            except:
                score = np.nan
            scores[strat_name] = score
        
        # Pairwise matches
        for s1, s2 in combinations(strategy_names, 2):
            if np.isnan(scores[s1]) or np.isnan(scores[s2]):
                continue
            
            # Outcome: who had better return today
            if scores[s1] > scores[s2]:
                outcome = 1.0
            elif scores[s2] > scores[s1]:
                outcome = 0.0
            else:
                outcome = 0.5
            
            # Update global + regime-specific
            ranking_manager.update(
                strategy_a_name=s1,
                strategy_b_name=s2,
                outcome=outcome,
                metric='alpha',
                regime=regime_label,
                timestamp=pd.Timestamp(date)
            )
            
            # Also global (regime=None)
            ranking_manager.update(
                strategy_a_name=s1,
                strategy_b_name=s2,
                outcome=outcome,
                metric='alpha',
                regime=None,
                timestamp=pd.Timestamp(date)
            )
            
            match_count += 1
    
    print(f"    Total matches: {match_count}")
    print(f"    Regimes: {unique_regimes}")
    
    # ========================================================================
    # VALIDATION TESTS
    # ========================================================================
    
    print("\n" + "="*100)
    print("VALIDATION TESTS")
    print("="*100)
    
    # TEST 1: Regime-conditional ranking variance
    print("\n[TEST 1] REGIME-CONDITIONAL RANKING VARIANCE")
    print("-"*100)
    variance_df = compute_regime_variance(ranking_manager, unique_regimes)
    print(variance_df.to_string(index=False))
    variance_df.to_csv(os.path.join(OUTPUT_DIR, 'test1_regime_variance.csv'), index=False)
    
    # TEST 2: Ranking stability
    print("\n[TEST 2] RANKING STABILITY (Spearman Rank Correlation)")
    print("-"*100)
    stability_df = compute_ranking_stability(ranking_manager, unique_regimes)
    print(stability_df.to_string(index=False))
    stability_df.to_csv(os.path.join(OUTPUT_DIR, 'test2_ranking_stability.csv'), index=False)
    
    # TEST 3: ELO momentum predictive test
    print("\n[TEST 3] ELO MOMENTUM PREDICTIVE TEST")
    print("-"*100)
    predictive_results = test_elo_momentum_predictive(
        ranking_manager, strategy_rets, W=20, T=20
    )
    
    if predictive_results:
        pred_df = pd.DataFrame(predictive_results).T
        print(pred_df.to_string())
        pred_df.to_csv(os.path.join(OUTPUT_DIR, 'test3_elo_momentum.csv'))
        
        significant = pred_df[pred_df['significant']]
        if len(significant) > 0:
            print(f"\n✓ SIGNIFICANT PREDICTIVE POWER FOUND:")
            print(f"  Strategies: {list(significant.index)}")
            print(f"  {len(significant)}/{len(pred_df)} strategies have ELO momentum predictability")
        else:
            print(f"\n✗ No significant ELO momentum predictability found")
    else:
        print("Could not compute predictive test")
    
    # ========================================================================
    # FINAL LEADERBOARDS
    # ========================================================================
    
    print("\n" + "="*100)
    print("FINAL LEADERBOARDS")
    print("="*100)
    
    print("\n--- GLOBAL LEADERBOARD ---")
    global_lb = ranking_manager.get_leaderboard(metric='alpha', regime=None)
    print(global_lb.to_string())
    global_lb.to_csv(os.path.join(OUTPUT_DIR, 'leaderboard_global.csv'))
    
    print("\n--- PER-REGIME LEADERBOARDS ---")
    for regime in unique_regimes:
        print(f"\nRegime {regime}:")
        regime_lb = ranking_manager.get_leaderboard(metric='alpha', regime=regime)
        print(regime_lb.to_string())
        regime_lb.to_csv(os.path.join(OUTPUT_DIR, f'leaderboard_regime_{regime}.csv'))
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    print(f"\nData points: {len(df_feat)} trading days")
    print(f"Strategies: {len(strategy_names)}")
    print(f"Total matches: {match_count}")
    print(f"Regimes: {len(unique_regimes)}")
    
    if not variance_df.empty:
        print(f"\nKey findings:")
        print(f"- Max regime variance: {variance_df['variance_across_regimes'].max():.1f}")
        print(f"- Specialists: {sum(variance_df['type'] == 'specialist')}")
    
    if not stability_df.empty:
        print(f"- Ranking stability (mean |rho|): {stability_df['spearman_rho'].abs().mean():.2f}")
    
    if predictive_results:
        print(f"- ELO momentum predictive: {sum(1 for v in predictive_results.values() if v['significant'])}/{len(predictive_results)} strategies")
    
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()