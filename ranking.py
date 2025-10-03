"""
Ranking System v1: ELO, TrueSkill, and Bradley-Terry rankings for strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
import numpy as np
import math

# Optional: TrueSkill library
try:
    import trueskill
    TRUESKILL_AVAILABLE = True
except ImportError:
    TRUESKILL_AVAILABLE = False


# ============================================================================
# RATING DATA STRUCTURES
# ============================================================================

@dataclass
class Rating:
    """Container for a rating with metadata"""
    value: float
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    last_updated: Optional[pd.Timestamp] = None
    
    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def __repr__(self):
        return f"Rating({self.value:.1f}, {self.games_played}g, {self.win_rate:.1%} wr)"


# ============================================================================
# BASE RANKING SYSTEM
# ============================================================================

class RankingSystem(ABC):
    """Base class for all ranking systems"""
    
    def __init__(self):
        self.rating_history: Dict[str, List[Tuple[pd.Timestamp, float]]] = defaultdict(list)
    
    @abstractmethod
    def update_ratings(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        outcome: float,  # 1.0 if A wins, 0.0 if B wins, 0.5 if tie
        regime: Optional[str] = None,
        timestamp: Optional[pd.Timestamp] = None
    ):
        """Update ratings based on match outcome"""
        pass
    
    @abstractmethod
    def get_rating(self, strategy_name: str, regime: Optional[str] = None) -> float:
        """Get current rating for a strategy"""
        pass
    
    @abstractmethod
    def get_all_ratings(self, regime: Optional[str] = None) -> Dict[str, Rating]:
        """Get all ratings as dictionary"""
        pass
    
    def get_leaderboard(self, regime: Optional[str] = None, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get sorted leaderboard"""
        ratings = self.get_all_ratings(regime)
        
        leaderboard = []
        for name, rating in ratings.items():
            leaderboard.append({
                'strategy': name,
                'rating': rating.value,
                'games': rating.games_played,
                'wins': rating.wins,
                'losses': rating.losses,
                'ties': rating.ties,
                'win_rate': rating.win_rate
            })
        
        df = pd.DataFrame(leaderboard)
        if not df.empty:
            df = df.sort_values('rating', ascending=False).reset_index(drop=True)
            df.index = df.index + 1  # Start ranking at 1
            if top_n:
                df = df.head(top_n)
        
        return df
    
    def get_rating_history(self, strategy_name: str) -> pd.Series:
        """Get historical ratings for a strategy"""
        history = self.rating_history.get(strategy_name, [])
        if not history:
            return pd.Series(dtype=float)
        
        timestamps, ratings = zip(*history)
        return pd.Series(ratings, index=timestamps)


# ============================================================================
# ELO RANKING SYSTEM
# ============================================================================

class ELORanking(RankingSystem):
    """
    Classic ELO rating system adapted for trading strategies.
    Simple, interpretable, battle-tested.
    """
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        """
        Args:
            k_factor: Maximum rating change per game (32 is standard)
            initial_rating: Starting rating for new strategies
        """
        super().__init__()
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        
        # Global ratings
        self.ratings: Dict[str, Rating] = defaultdict(
            lambda: Rating(value=initial_rating)
        )
        
        # Regime-specific ratings
        self.regime_ratings: Dict[str, Dict[str, Rating]] = defaultdict(
            lambda: defaultdict(lambda: Rating(value=initial_rating))
        )
    
    def update_ratings(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        outcome: float,
        regime: Optional[str] = None,
        timestamp: Optional[pd.Timestamp] = None
    ):
        """Update ELO ratings based on match outcome"""
        timestamp = timestamp or pd.Timestamp.now()
        
        # Update global ratings
        self._elo_update(
            strategy_a_name,
            strategy_b_name,
            outcome,
            self.ratings,
            timestamp
        )
        
        # Update regime-specific ratings
        if regime is not None:
            self._elo_update(
                strategy_a_name,
                strategy_b_name,
                outcome,
                self.regime_ratings[regime],
                timestamp
            )
    
    def _elo_update(
        self,
        name_a: str,
        name_b: str,
        outcome: float,
        ratings_dict: Dict[str, Rating],
        timestamp: pd.Timestamp
    ):
        """Core ELO update logic"""
        # Get current ratings
        rating_a = ratings_dict[name_a]
        rating_b = ratings_dict[name_b]
        
        # Calculate expected scores
        expected_a = self._expected_score(rating_a.value, rating_b.value)
        expected_b = 1 - expected_a
        
        # Update ratings
        rating_a.value += self.k_factor * (outcome - expected_a)
        rating_b.value += self.k_factor * ((1 - outcome) - expected_b)
        
        # Update game counts
        rating_a.games_played += 1
        rating_b.games_played += 1
        
        if outcome == 1.0:
            rating_a.wins += 1
            rating_b.losses += 1
        elif outcome == 0.0:
            rating_a.losses += 1
            rating_b.wins += 1
        else:
            rating_a.ties += 1
            rating_b.ties += 1
        
        # Update timestamps
        rating_a.last_updated = timestamp
        rating_b.last_updated = timestamp
        
        # Store history
        self.rating_history[name_a].append((timestamp, rating_a.value))
        self.rating_history[name_b].append((timestamp, rating_b.value))
    
    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def get_rating(self, strategy_name: str, regime: Optional[str] = None) -> float:
        """Get current rating"""
        if regime is None:
            return self.ratings[strategy_name].value
        else:
            return self.regime_ratings[regime][strategy_name].value
    
    def get_all_ratings(self, regime: Optional[str] = None) -> Dict[str, Rating]:
        """Get all ratings"""
        if regime is None:
            return dict(self.ratings)
        else:
            return dict(self.regime_ratings[regime])


# ============================================================================
# TRUESKILL RANKING SYSTEM
# ============================================================================

class TrueSkillRanking(RankingSystem):
    """
    Microsoft's TrueSkill system - more sophisticated than ELO.
    Tracks both skill (mu) and uncertainty (sigma).
    Better for handling new strategies and variable skill.
    """
    
    def __init__(
        self,
        mu: float = 25.0,
        sigma: float = 25/3,
        beta: float = 25/6,
        tau: float = 25/300,
        draw_probability: float = 0.1
    ):
        """
        Args:
            mu: Initial mean skill
            sigma: Initial skill uncertainty
            beta: Skill difference for 80% win probability
            tau: Dynamics factor (skill drift over time)
            draw_probability: Probability of draws
        """
        super().__init__()
        
        if not TRUESKILL_AVAILABLE:
            raise ImportError("TrueSkill not installed. Install with: pip install trueskill")
        
        # Configure TrueSkill environment
        self.env = trueskill.TrueSkill(
            mu=mu,
            sigma=sigma,
            beta=beta,
            tau=tau,
            draw_probability=draw_probability
        )
        
        # Global ratings
        self.ratings: Dict[str, trueskill.Rating] = {}
        
        # Regime-specific ratings
        self.regime_ratings: Dict[str, Dict[str, trueskill.Rating]] = defaultdict(dict)
    
    def update_ratings(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        outcome: float,
        regime: Optional[str] = None,
        timestamp: Optional[pd.Timestamp] = None
    ):
        """Update TrueSkill ratings"""
        timestamp = timestamp or pd.Timestamp.now()
        
        # Update global ratings
        self._trueskill_update(
            strategy_a_name,
            strategy_b_name,
            outcome,
            self.ratings,
            timestamp
        )
        
        # Update regime-specific ratings
        if regime is not None:
            self._trueskill_update(
                strategy_a_name,
                strategy_b_name,
                outcome,
                self.regime_ratings[regime],
                timestamp
            )
    
    def _trueskill_update(
        self,
        name_a: str,
        name_b: str,
        outcome: float,
        ratings_dict: Dict[str, trueskill.Rating],
        timestamp: pd.Timestamp
    ):
        """Core TrueSkill update logic"""
        # Initialize if needed
        if name_a not in ratings_dict:
            ratings_dict[name_a] = self.env.create_rating()
        if name_b not in ratings_dict:
            ratings_dict[name_b] = self.env.create_rating()
        
        # Determine ranks (lower is better)
        if outcome == 1.0:  # A wins
            ranks = [0, 1]
        elif outcome == 0.0:  # B wins
            ranks = [1, 0]
        else:  # Tie
            ranks = [0, 0]
        
        # Update ratings
        (new_a,), (new_b,) = self.env.rate(
            [(ratings_dict[name_a],), (ratings_dict[name_b],)],
            ranks=ranks
        )
        
        ratings_dict[name_a] = new_a
        ratings_dict[name_b] = new_b
        
        # Store history (using conservative rating: mu - 3*sigma)
        conservative_a = new_a.mu - 3 * new_a.sigma
        conservative_b = new_b.mu - 3 * new_b.sigma
        
        self.rating_history[name_a].append((timestamp, conservative_a))
        self.rating_history[name_b].append((timestamp, conservative_b))
    
    def get_rating(self, strategy_name: str, regime: Optional[str] = None) -> float:
        """Get conservative rating (mu - 3*sigma)"""
        if regime is None:
            rating = self.ratings.get(strategy_name)
        else:
            rating = self.regime_ratings[regime].get(strategy_name)
        
        if rating is None:
            return self.env.mu - 3 * self.env.sigma
        
        return rating.mu - 3 * rating.sigma
    
    def get_all_ratings(self, regime: Optional[str] = None) -> Dict[str, Rating]:
        """Get all ratings with uncertainty info"""
        if regime is None:
            ratings_dict = self.ratings
        else:
            ratings_dict = self.regime_ratings[regime]
        
        result = {}
        for name, ts_rating in ratings_dict.items():
            result[name] = Rating(
                value=ts_rating.mu - 3 * ts_rating.sigma,
                games_played=0  # TrueSkill doesn't track this directly
            )
        
        return result


# ============================================================================
# BRADLEY-TERRY RANKING SYSTEM
# ============================================================================

class BradleyTerryRanking(RankingSystem):
    """
    Bradley-Terry model: probabilistic paired comparison model.
    P(A beats B) = strength_A / (strength_A + strength_B)
    Fitted using maximum likelihood estimation.
    """
    
    def __init__(self, initial_strength: float = 1.0, learning_rate: float = 0.01):
        """
        Args:
            initial_strength: Starting strength for new strategies
            learning_rate: Step size for gradient updates
        """
        super().__init__()
        self.initial_strength = initial_strength
        self.learning_rate = learning_rate
        
        # Global strengths
        self.strengths: Dict[str, float] = defaultdict(lambda: initial_strength)
        self.game_counts: Dict[str, int] = defaultdict(int)
        
        # Regime-specific strengths
        self.regime_strengths: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: initial_strength)
        )
        self.regime_game_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
    
    def update_ratings(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        outcome: float,
        regime: Optional[str] = None,
        timestamp: Optional[pd.Timestamp] = None
    ):
        """Update Bradley-Terry strengths via gradient descent"""
        timestamp = timestamp or pd.Timestamp.now()
        
        # Update global strengths
        self._bt_update(
            strategy_a_name,
            strategy_b_name,
            outcome,
            self.strengths,
            self.game_counts,
            timestamp
        )
        
        # Update regime-specific strengths
        if regime is not None:
            self._bt_update(
                strategy_a_name,
                strategy_b_name,
                outcome,
                self.regime_strengths[regime],
                self.regime_game_counts[regime],
                timestamp
            )
    
    def _bt_update(
        self,
        name_a: str,
        name_b: str,
        outcome: float,
        strengths_dict: Dict[str, float],
        counts_dict: Dict[str, int],
        timestamp: pd.Timestamp
    ):
        """Bradley-Terry gradient update"""
        # Get current strengths
        s_a = strengths_dict[name_a]
        s_b = strengths_dict[name_b]
        
        # Predicted probability A wins
        p_a_wins = s_a / (s_a + s_b)
        
        # Gradient update
        error = outcome - p_a_wins
        
        strengths_dict[name_a] += self.learning_rate * error * s_b / (s_a + s_b)
        strengths_dict[name_b] -= self.learning_rate * error * s_a / (s_a + s_b)
        
        # Ensure positive
        strengths_dict[name_a] = max(0.01, strengths_dict[name_a])
        strengths_dict[name_b] = max(0.01, strengths_dict[name_b])
        
        # Update counts
        counts_dict[name_a] += 1
        counts_dict[name_b] += 1
        
        # Store history (convert to rating-like scale)
        rating_a = 1500 + 400 * np.log10(strengths_dict[name_a])
        rating_b = 1500 + 400 * np.log10(strengths_dict[name_b])
        
        self.rating_history[name_a].append((timestamp, rating_a))
        self.rating_history[name_b].append((timestamp, rating_b))
    
    def get_rating(self, strategy_name: str, regime: Optional[str] = None) -> float:
        """Get rating (converted from strength)"""
        if regime is None:
            strength = self.strengths[strategy_name]
        else:
            strength = self.regime_strengths[regime][strategy_name]
        
        # Convert to ELO-like scale
        return 1500 + 400 * np.log10(strength)
    
    def get_all_ratings(self, regime: Optional[str] = None) -> Dict[str, Rating]:
        """Get all ratings"""
        if regime is None:
            strengths_dict = self.strengths
            counts_dict = self.game_counts
        else:
            strengths_dict = self.regime_strengths[regime]
            counts_dict = self.regime_game_counts[regime]
        
        result = {}
        for name, strength in strengths_dict.items():
            rating_value = 1500 + 400 * np.log10(strength)
            result[name] = Rating(
                value=rating_value,
                games_played=counts_dict[name]
            )
        
        return result


# ============================================================================
# MULTI-METRIC RANKING MANAGER
# ============================================================================

class RankingManager:
    """
    Manages multiple ranking systems across metrics and regimes.
    This is the main interface for the engine.
    """
    
    def __init__(self, ranking_class=ELORanking, **ranking_kwargs):
        """
        Args:
            ranking_class: Which ranking system to use (ELO, TrueSkill, etc.)
            **ranking_kwargs: Arguments for the ranking system
        """
        self.ranking_class = ranking_class
        self.ranking_kwargs = ranking_kwargs
        
        # Structure: rankings[metric][regime] = RankingSystem
        self.rankings: Dict[str, Dict[Optional[str], RankingSystem]] = defaultdict(dict)
        
        # Track which metrics and regimes we've seen
        self.metrics: set = set()
        self.regimes: set = set()
    
    def update(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        outcome: float,
        metric: str = 'sharpe',
        regime: Optional[str] = None,
        timestamp: Optional[pd.Timestamp] = None
    ):
        """Update rankings for a specific metric and regime"""
        # Ensure ranking system exists
        if regime not in self.rankings[metric]:
            self.rankings[metric][regime] = self.ranking_class(**self.ranking_kwargs)
        
        # Track metrics and regimes
        self.metrics.add(metric)
        if regime is not None:
            self.regimes.add(regime)
        
        # Update
        self.rankings[metric][regime].update_ratings(
            strategy_a_name,
            strategy_b_name,
            outcome,
            regime,
            timestamp
        )
    
    def get_leaderboard(
        self,
        metric: str = 'sharpe',
        regime: Optional[str] = None,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """Get leaderboard for specific metric and regime"""
        if metric not in self.rankings or regime not in self.rankings[metric]:
            return pd.DataFrame()
        
        return self.rankings[metric][regime].get_leaderboard(regime, top_n)
    
    def get_all_leaderboards(self) -> Dict[str, Dict[Optional[str], pd.DataFrame]]:
        """Get all leaderboards across metrics and regimes"""
        result = {}
        for metric in self.metrics:
            result[metric] = {}
            for regime in [None] + list(self.regimes):
                if regime in self.rankings[metric]:
                    result[metric][regime] = self.get_leaderboard(metric, regime)
        return result
    
    def get_rating_history(
        self,
        strategy_name: str,
        metric: str = 'sharpe',
        regime: Optional[str] = None
    ) -> pd.Series:
        """Get rating history for a strategy"""
        if metric in self.rankings and regime in self.rankings[metric]:
            return self.rankings[metric][regime].get_rating_history(strategy_name)
        return pd.Series(dtype=float)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Ranking System v1 - Example Usage")
    print("="*60)
    
    # Test ELO
    print("\n" + "-"*60)
    print("1. ELO Ranking System")
    print("-"*60)
    
    elo = ELORanking(k_factor=32, initial_rating=1500)
    
    # Simulate some matches
    matches = [
        ("StrategyA", "StrategyB", 1.0, "high_vol"),
        ("StrategyA", "StrategyC", 0.5, "high_vol"),
        ("StrategyB", "StrategyC", 0.0, "high_vol"),
        ("StrategyA", "StrategyB", 0.0, "low_vol"),
        ("StrategyC", "StrategyB", 1.0, "low_vol"),
    ]
    
    for strat_a, strat_b, outcome, regime in matches:
        elo.update_ratings(strat_a, strat_b, outcome, regime)
    
    print("\nGlobal Leaderboard:")
    print(elo.get_leaderboard())
    
    print("\nHigh Vol Regime Leaderboard:")
    print(elo.get_leaderboard(regime="high_vol"))
    
    print("\nLow Vol Regime Leaderboard:")
    print(elo.get_leaderboard(regime="low_vol"))
    
    # Test Ranking Manager
    print("\n" + "-"*60)
    print("2. Ranking Manager (Multi-Metric)")
    print("-"*60)
    
    manager = RankingManager(ranking_class=ELORanking, k_factor=32)
    
    # Update different metrics
    manager.update("StratA", "StratB", 1.0, metric='sharpe', regime='high_vol')
    manager.update("StratA", "StratB", 0.0, metric='returns', regime='high_vol')
    manager.update("StratA", "StratC", 1.0, metric='sharpe', regime='low_vol')
    
    print("\nSharpe Leaderboard (High Vol):")
    print(manager.get_leaderboard('sharpe', 'high_vol'))
    
    print("\nReturns Leaderboard (High Vol):")
    print(manager.get_leaderboard('returns', 'high_vol'))
    
    print("\n" + "="*60)
    print("Ranking System Complete!")
    print("="*60)