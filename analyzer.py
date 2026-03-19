"""
Meta ELO Analyzer: Analyze the dynamics of ELO ratings themselves
Treats ELO trajectories as signals for regime detection and strategy meta-analysis
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from datetime import datetime


# ============================================================================
# META ELO ANALYZER
# ============================================================================

class MetaELOAnalyzer:
    """
    Analyzes ELO dynamics themselves as meta-signals.
    
    Key insights:
    - ELO velocity: Is a strategy improving or declining?
    - ELO volatility: Is performance consistent or erratic?
    - Transfer entropy: Are two strategies correlated?
    - Regime transitions: When do many ELOs change at once?
    """
    
    def __init__(self, ranking_manager):
        """
        Args:
            ranking_manager: RankingManager instance with rating history
        """
        self.ranking_manager = ranking_manager
    
    # ========================================================================
    # ELO VELOCITY
    # ========================================================================
    
    def compute_elo_velocity(
        self,
        strategy_name: str,
        window: int = 20,
        metric: str = 'sharpe',
        regime: Optional[str] = None
    ) -> pd.Series:
        """
        Compute rate of ELO change (first derivative).
        Positive velocity = improving, negative = declining.
        
        Args:
            strategy_name: Name of strategy
            window: Rolling window for velocity calculation
            metric: Which metric's ELO to analyze
            regime: Which regime (None = global)
        
        Returns:
            Series of ELO velocity over time
        """
        # Get rating history
        history = self.ranking_manager.get_rating_history(
            strategy_name, metric, regime
        )
        
        if len(history) < 2:
            return pd.Series(dtype=float)
        
        # Calculate velocity (change in rating per time unit)
        velocity = history.diff()
        
        # Smooth with rolling mean
        if window > 1:
            velocity = velocity.rolling(window=window, min_periods=1).mean()
        
        return velocity
    
    def get_velocity_ranking(
        self,
        metric: str = 'sharpe',
        regime: Optional[str] = None,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Rank strategies by current ELO velocity.
        Shows who's on the rise vs who's declining.
        """
        rankings = self.ranking_manager.get_leaderboard(metric, regime)
        
        if rankings.empty:
            return pd.DataFrame()
        
        velocities = []
        for strategy_name in rankings['strategy']:
            vel = self.compute_elo_velocity(strategy_name, window, metric, regime)
            current_vel = vel.iloc[-1] if len(vel) > 0 else 0
            velocities.append(current_vel)
        
        rankings['velocity'] = velocities
        rankings = rankings.sort_values('velocity', ascending=False)
        
        return rankings[['strategy', 'rating', 'velocity', 'games']]
    
    # ========================================================================
    # ELO VOLATILITY
    # ========================================================================
    
    def compute_elo_volatility(
        self,
        strategy_name: str,
        window: int = 20,
        metric: str = 'sharpe',
        regime: Optional[str] = None
    ) -> float:
        """
        Compute stability of ELO ratings.
        High volatility = inconsistent performance.
        Low volatility = stable, reliable performance.
        
        Returns:
            Standard deviation of ELO changes
        """
        # Get rating history
        history = self.ranking_manager.get_rating_history(
            strategy_name, metric, regime
        )
        
        if len(history) < 2:
            return 0.0
        
        # Calculate volatility of rating changes
        rating_changes = history.diff().dropna()
        
        if len(rating_changes) == 0:
            return 0.0
        
        # Use rolling window if specified
        if window < len(rating_changes):
            recent_changes = rating_changes.iloc[-window:]
            return recent_changes.std()
        else:
            return rating_changes.std()
    
    def get_volatility_ranking(
        self,
        metric: str = 'sharpe',
        regime: Optional[str] = None,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Rank strategies by ELO stability.
        Lower volatility = more consistent.
        """
        rankings = self.ranking_manager.get_leaderboard(metric, regime)
        
        if rankings.empty:
            return pd.DataFrame()
        
        volatilities = []
        for strategy_name in rankings['strategy']:
            vol = self.compute_elo_volatility(strategy_name, window, metric, regime)
            volatilities.append(vol)
        
        rankings['elo_volatility'] = volatilities
        rankings = rankings.sort_values('elo_volatility', ascending=True)
        
        return rankings[['strategy', 'rating', 'elo_volatility', 'games']]
    
    # ========================================================================
    # TRANSFER ENTROPY
    # ========================================================================
    
    def compute_transfer_entropy(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        metric: str = 'sharpe',
        regime: Optional[str] = None,
        lag: int = 1
    ) -> float:
        """
        Measure information flow from strategy A's ELO to strategy B's ELO.
        High transfer entropy = A's performance predicts B's.
        Indicates strategies are correlated or in same niche.
        
        Simplified version using correlation (true transfer entropy is complex).
        
        Returns:
            Correlation coefficient between lagged ELO series
        """
        # Get rating histories
        history_a = self.ranking_manager.get_rating_history(
            strategy_a_name, metric, regime
        )
        history_b = self.ranking_manager.get_rating_history(
            strategy_b_name, metric, regime
        )
        
        if len(history_a) < 2 or len(history_b) < 2:
            return 0.0
        
        # Align timestamps
        common_idx = history_a.index.intersection(history_b.index)
        if len(common_idx) < 2:
            return 0.0
        
        history_a = history_a.loc[common_idx]
        history_b = history_b.loc[common_idx]
        
        # Compute lagged correlation
        # Does A's rating at time t predict B's rating at time t+lag?
        if lag > 0:
            a_lagged = history_a.iloc[:-lag]
            b_future = history_b.iloc[lag:]
            
            if len(a_lagged) == 0 or len(b_future) == 0:
                return 0.0
            
            a_lagged.index = b_future.index  # Align
            corr = a_lagged.corr(b_future)
        else:
            corr = history_a.corr(history_b)
        
        return corr if not np.isnan(corr) else 0.0
    
    def compute_correlation_matrix(
        self,
        metric: str = 'sharpe',
        regime: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute pairwise correlation matrix of all strategy ELO trajectories.
        Identifies clusters of similar strategies.
        """
        rankings = self.ranking_manager.get_leaderboard(metric, regime)
        
        if rankings.empty:
            return pd.DataFrame()
        
        strategies = rankings['strategy'].tolist()
        n = len(strategies)
        
        # Build correlation matrix
        corr_matrix = np.zeros((n, n))
        
        for i, strat_a in enumerate(strategies):
            for j, strat_b in enumerate(strategies):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = self.compute_transfer_entropy(
                        strat_a, strat_b, metric, regime, lag=0
                    )
                    corr_matrix[i, j] = corr
        
        # Convert to DataFrame
        df = pd.DataFrame(
            corr_matrix,
            index=strategies,
            columns=strategies
        )
        
        return df
    
    # ========================================================================
    # REGIME TRANSITION DETECTION
    # ========================================================================
    
    def detect_regime_transitions(
        self,
        metric: str = 'sharpe',
        regime: Optional[str] = None,
        threshold_pct: float = 0.5
    ) -> List[Tuple[pd.Timestamp, int]]:
        """
        Detect when many ELOs change dramatically at once.
        This indicates a regime shift or market phase transition.
        
        Args:
            metric: Which metric to analyze
            regime: Which regime (None = global)
            threshold_pct: What % of strategies must change for it to count
        
        Returns:
            List of (timestamp, num_strategies_affected) tuples
        """
        rankings = self.ranking_manager.get_leaderboard(metric, regime)
        
        if rankings.empty:
            return []
        
        strategies = rankings['strategy'].tolist()
        
        # Get all rating histories
        all_histories = {}
        for strat in strategies:
            history = self.ranking_manager.get_rating_history(strat, metric, regime)
            if len(history) > 0:
                all_histories[strat] = history
        
        if not all_histories:
            return []
        
        # Find common timestamps
        common_idx = None
        for history in all_histories.values():
            if common_idx is None:
                common_idx = set(history.index)
            else:
                common_idx = common_idx.intersection(set(history.index))
        
        if not common_idx:
            return []
        
        common_idx = sorted(list(common_idx))
        
        # For each timestamp, count how many strategies had large rating changes
        transitions = []
        
        for i in range(1, len(common_idx)):
            timestamp = common_idx[i]
            prev_timestamp = common_idx[i-1]
            
            # Count strategies with large changes
            large_changes = 0
            
            for strat, history in all_histories.items():
                if timestamp in history.index and prev_timestamp in history.index:
                    change = abs(history[timestamp] - history[prev_timestamp])
                    
                    # Large change = more than 50 rating points
                    if change > 50:
                        large_changes += 1
            
            # If enough strategies changed, mark as transition
            pct_changed = large_changes / len(all_histories)
            if pct_changed >= threshold_pct:
                transitions.append((timestamp, large_changes))
        
        return transitions
    
    def get_transition_report(
        self,
        metric: str = 'sharpe',
        regime: Optional[str] = None
    ) -> pd.DataFrame:
        """Get detailed report of regime transitions"""
        transitions = self.detect_regime_transitions(metric, regime)
        
        if not transitions:
            return pd.DataFrame()
        
        report = []
        for timestamp, num_affected in transitions:
            report.append({
                'timestamp': timestamp,
                'strategies_affected': num_affected,
                'date': timestamp.strftime('%Y-%m-%d') if hasattr(timestamp, 'strftime') else str(timestamp)
            })
        
        return pd.DataFrame(report)
    
    # ========================================================================
    # ELO MOMENTUM (ACCELERATION)
    # ========================================================================
    
    def compute_elo_momentum(
        self,
        strategy_name: str,
        window: int = 10,
        metric: str = 'sharpe',
        regime: Optional[str] = None
    ) -> pd.Series:
        """
        Compute second derivative of ELO (acceleration).
        Positive momentum = accelerating improvement.
        Negative momentum = improvement is slowing.
        """
        velocity = self.compute_elo_velocity(
            strategy_name, window=1, metric=metric, regime=regime
        )
        
        if len(velocity) < 2:
            return pd.Series(dtype=float)
        
        # Momentum is change in velocity
        momentum = velocity.diff()
        
        # Smooth
        if window > 1:
            momentum = momentum.rolling(window=window, min_periods=1).mean()
        
        return momentum
    
    # ========================================================================
    # ELO PEAK DETECTION
    # ========================================================================
    
    def find_elo_peaks(
        self,
        strategy_name: str,
        metric: str = 'sharpe',
        regime: Optional[str] = None,
        prominence: float = 50
    ) -> List[Tuple[pd.Timestamp, float]]:
        """
        Find local peaks in ELO trajectory.
        Indicates periods when strategy performed exceptionally well.
        
        Args:
            prominence: Minimum height difference from surrounding values
        
        Returns:
            List of (timestamp, rating) at peaks
        """
        history = self.ranking_manager.get_rating_history(
            strategy_name, metric, regime
        )
        
        if len(history) < 3:
            return []
        
        # Find peaks
        peaks, properties = find_peaks(history.values, prominence=prominence)
        
        results = []
        for peak_idx in peaks:
            timestamp = history.index[peak_idx]
            rating = history.iloc[peak_idx]
            results.append((timestamp, rating))
        
        return results
    
    # ========================================================================
    # STRATEGY LIFECYCLE ANALYSIS
    # ========================================================================
    
    def analyze_lifecycle(
        self,
        strategy_name: str,
        metric: str = 'sharpe',
        regime: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Comprehensive lifecycle analysis of a strategy.
        
        Returns:
            Dictionary with lifecycle metrics
        """
        history = self.ranking_manager.get_rating_history(
            strategy_name, metric, regime
        )
        
        if len(history) < 2:
            return {}
        
        velocity = self.compute_elo_velocity(strategy_name, window=5, metric=metric, regime=regime)
        volatility = self.compute_elo_volatility(strategy_name, window=20, metric=metric, regime=regime)
        momentum = self.compute_elo_momentum(strategy_name, window=5, metric=metric, regime=regime)
        peaks = self.find_elo_peaks(strategy_name, metric=metric, regime=regime)
        
        # Current trend
        if len(velocity) > 0:
            recent_velocity = velocity.iloc[-5:].mean() if len(velocity) >= 5 else velocity.mean()
            if recent_velocity > 10:
                trend = "Rising"
            elif recent_velocity < -10:
                trend = "Declining"
            else:
                trend = "Stable"
        else:
            trend = "Unknown"
        
        # Lifecycle stage
        if len(history) < 10:
            stage = "New"
        elif len(peaks) > 0 and history.iloc[-1] < peaks[-1][1] * 0.9:
            stage = "Declining"
        elif recent_velocity > 0:
            stage = "Growing"
        else:
            stage = "Mature"
        
        return {
            'strategy': strategy_name,
            'current_rating': history.iloc[-1],
            'peak_rating': history.max(),
            'min_rating': history.min(),
            'total_change': history.iloc[-1] - history.iloc[0],
            'current_velocity': velocity.iloc[-1] if len(velocity) > 0 else 0,
            'elo_volatility': volatility,
            'trend': trend,
            'lifecycle_stage': stage,
            'num_peaks': len(peaks),
            'games_played': len(history)
        }
    
    def get_lifecycle_report(
        self,
        metric: str = 'sharpe',
        regime: Optional[str] = None
    ) -> pd.DataFrame:
        """Get lifecycle analysis for all strategies"""
        rankings = self.ranking_manager.get_leaderboard(metric, regime)
        
        if rankings.empty:
            return pd.DataFrame()
        
        reports = []
        for strategy_name in rankings['strategy']:
            report = self.analyze_lifecycle(strategy_name, metric, regime)
            if report:
                reports.append(report)
        
        return pd.DataFrame(reports)
    
    # ========================================================================
    # DIVERGENCE DETECTION
    # ========================================================================
    
    def detect_divergence(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        metric: str = 'sharpe',
        regime: Optional[str] = None,
        window: int = 20
    ) -> Dict[str, any]:
        """
        Detect when two strategies' ELOs diverge.
        Useful for understanding when strategies start behaving differently.
        """
        history_a = self.ranking_manager.get_rating_history(
            strategy_a_name, metric, regime
        )
        history_b = self.ranking_manager.get_rating_history(
            strategy_b_name, metric, regime
        )
        
        if len(history_a) < 2 or len(history_b) < 2:
            return {}
        
        # Align
        common_idx = history_a.index.intersection(history_b.index)
        if len(common_idx) < window:
            return {}
        
        history_a = history_a.loc[common_idx]
        history_b = history_b.loc[common_idx]
        
        # Calculate difference
        diff = history_a - history_b
        
        # Rolling correlation
        corr = history_a.rolling(window=window).corr(history_b)
        
        # Find divergence points (where correlation drops)
        divergence_points = []
        for i in range(window, len(corr)):
            if corr.iloc[i] < 0.3 and corr.iloc[i-1] > 0.5:  # Sharp drop
                divergence_points.append(common_idx[i])
        
        return {
            'strategy_a': strategy_a_name,
            'strategy_b': strategy_b_name,
            'current_correlation': corr.iloc[-1] if len(corr) > 0 else 0,
            'avg_correlation': corr.mean() if len(corr) > 0 else 0,
            'current_diff': diff.iloc[-1],
            'max_diff': diff.abs().max(),
            'divergence_events': len(divergence_points),
            'divergence_dates': divergence_points
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from ranking_system import RankingManager, ELORanking
    
    print("="*60)
    print("Meta ELO Analyzer - Example Usage")
    print("="*60)
    
    # Create ranking manager and simulate some data
    manager = RankingManager(ranking_class=ELORanking, k_factor=32)
    
    # Simulate matches over time
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    for i, date in enumerate(dates):
        # Strategy A starts strong, then declines
        if i < 25:
            manager.update("StrategyA", "StrategyB", 0.7, metric='sharpe', timestamp=date)
        else:
            manager.update("StrategyA", "StrategyB", 0.3, metric='sharpe', timestamp=date)
        
        # Strategy C is consistently good
        manager.update("StrategyC", "StrategyB", 0.6, metric='sharpe', timestamp=date)
    
    # Create analyzer
    analyzer = MetaELOAnalyzer(manager)
    
    # Test velocity
    print("\n" + "-"*60)
    print("1. ELO Velocity Analysis")
    print("-"*60)
    print(analyzer.get_velocity_ranking(metric='sharpe'))
    
    # Test volatility
    print("\n" + "-"*60)
    print("2. ELO Volatility Analysis")
    print("-"*60)
    print(analyzer.get_volatility_ranking(metric='sharpe'))
    
    # Test lifecycle
    print("\n" + "-"*60)
    print("3. Strategy Lifecycle Analysis")
    print("-"*60)
    print(analyzer.get_lifecycle_report(metric='sharpe'))
    
    # Test transfer entropy
    print("\n" + "-"*60)
    print("4. Transfer Entropy (Correlation)")
    print("-"*60)
    te = analyzer.compute_transfer_entropy("StrategyA", "StrategyC", metric='sharpe')
    print(f"StrategyA -> StrategyC correlation: {te:.3f}")
    
    # Test regime transitions
    print("\n" + "-"*60)
    print("5. Regime Transition Detection")
    print("-"*60)
    transitions = analyzer.get_transition_report(metric='sharpe')
    if not transitions.empty:
        print(transitions)
    else:
        print("No major regime transitions detected")
    
    # Test divergence
    print("\n" + "-"*60)
    print("6. Strategy Divergence Detection")
    print("-"*60)
    div = analyzer.detect_divergence("StrategyA", "StrategyC", metric='sharpe')
    if div:
        print(f"Current correlation: {div['current_correlation']:.3f}")
        print(f"Divergence events: {div['divergence_events']}")
    
    print("\n" + "="*60)
    print("Meta Analysis Complete!")
    print("="*60)