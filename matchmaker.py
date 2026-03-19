"""
Matchmaker v1: Strategy pairing system for head-to-head comparisons
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Set
import pandas as pd
import numpy as np
from itertools import combinations
from dataclasses import dataclass
import random


# ============================================================================
# MATCHUP DATA STRUCTURE
# ============================================================================

@dataclass
class Matchup:
    """Represents a head-to-head comparison between two strategies"""
    strategy_a: any  # Strategy instance
    strategy_b: any  # Strategy instance
    regime: Optional[str] = None
    asset: Optional[str] = None
    timeframe: Optional[str] = None
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    
    def __repr__(self):
        regime_str = f"[{self.regime}]" if self.regime else ""
        return f"{self.strategy_a.name} vs {self.strategy_b.name} {regime_str}"
    
    def get_key(self) -> str:
        """Unique identifier for this matchup"""
        parts = [
            self.strategy_a.name,
            self.strategy_b.name,
            str(self.regime) if self.regime else "all",
            str(self.asset) if self.asset else "all",
            str(self.timeframe) if self.timeframe else "all"
        ]
        return "_".join(parts)


# ============================================================================
# BASE MATCHMAKER
# ============================================================================

class MatchMaker(ABC):
    """Base class for strategy matchmaking"""
    
    def __init__(self):
        self.match_history: List[Matchup] = []
    
    @abstractmethod
    def generate_matchups(
        self,
        strategies: List,
        data: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        asset: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Matchup]:
        """
        Generate list of strategy matchups.
        
        Args:
            strategies: List of Strategy instances
            data: Price/volume data
            regimes: Optional regime labels
            asset: Optional asset identifier
            timeframe: Optional timeframe identifier
        
        Returns:
            List of Matchup objects
        """
        pass
    
    def get_match_count(self) -> int:
        """Total number of matches generated"""
        return len(self.match_history)
    
    def clear_history(self):
        """Clear match history"""
        self.match_history = []


# ============================================================================
# ROUND ROBIN MATCHMAKER
# ============================================================================

class RoundRobinMatcher(MatchMaker):
    """
    Everyone vs everyone - exhaustive pairwise comparisons.
    Good for small strategy universes (<20 strategies).
    """
    
    def __init__(self, include_global: bool = True):
        """
        Args:
            include_global: If True, also create matchups with regime=None (overall)
        """
        super().__init__()
        self.include_global = include_global
    
    def generate_matchups(
        self,
        strategies: List,
        data: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        asset: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Matchup]:
        """Generate all possible pairwise matchups"""
        
        matchups = []
        
        # Get all strategy pairs
        strategy_pairs = list(combinations(strategies, 2))
        
        if regimes is not None:
            # Create matchups per regime
            unique_regimes = regimes.unique()
            
            for strat_a, strat_b in strategy_pairs:
                for regime in unique_regimes:
                    matchups.append(Matchup(
                        strategy_a=strat_a,
                        strategy_b=strat_b,
                        regime=regime,
                        asset=asset,
                        timeframe=timeframe
                    ))
            
            # Also add global matchups (across all regimes)
            if self.include_global:
                for strat_a, strat_b in strategy_pairs:
                    matchups.append(Matchup(
                        strategy_a=strat_a,
                        strategy_b=strat_b,
                        regime=None,
                        asset=asset,
                        timeframe=timeframe
                    ))
        else:
            # No regimes - just create pairs
            for strat_a, strat_b in strategy_pairs:
                matchups.append(Matchup(
                    strategy_a=strat_a,
                    strategy_b=strat_b,
                    regime=None,
                    asset=asset,
                    timeframe=timeframe
                ))
        
        self.match_history.extend(matchups)
        return matchups
    
    def estimate_match_count(self, num_strategies: int, num_regimes: int = 1) -> int:
        """Estimate total matches before generating"""
        pairs = (num_strategies * (num_strategies - 1)) // 2
        
        if self.include_global and num_regimes > 1:
            return pairs * (num_regimes + 1)
        else:
            return pairs * num_regimes


# ============================================================================
# RANDOM SAMPLING MATCHMAKER
# ============================================================================

class RandomSamplingMatcher(MatchMaker):
    """
    Randomly sample strategy pairs.
    Good when universe is large and we want a representative sample.
    """
    
    def __init__(self, num_matches: int = 100, seed: Optional[int] = None):
        """
        Args:
            num_matches: Number of random matchups to generate
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.num_matches = num_matches
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_matchups(
        self,
        strategies: List,
        data: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        asset: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Matchup]:
        """Generate random sample of matchups"""
        
        matchups = []
        
        if len(strategies) < 2:
            return matchups
        
        # Determine regimes
        if regimes is not None:
            regime_list = list(regimes.unique()) + [None]  # Include global
        else:
            regime_list = [None]
        
        # Generate random matchups
        for _ in range(self.num_matches):
            # Random strategy pair (without replacement)
            strat_a, strat_b = random.sample(strategies, 2)
            
            # Random regime
            regime = random.choice(regime_list)
            
            matchups.append(Matchup(
                strategy_a=strat_a,
                strategy_b=strat_b,
                regime=regime,
                asset=asset,
                timeframe=timeframe
            ))
        
        self.match_history.extend(matchups)
        return matchups


# ============================================================================
# TOURNAMENT MATCHMAKER
# ============================================================================

class TournamentMatcher(MatchMaker):
    """
    Single-elimination or double-elimination tournament bracket.
    Good for identifying top strategies quickly.
    """
    
    def __init__(self, style: str = 'single'):
        """
        Args:
            style: 'single' or 'double' elimination
        """
        super().__init__()
        if style not in ['single', 'double']:
            raise ValueError("style must be 'single' or 'double'")
        self.style = style
    
    def generate_matchups(
        self,
        strategies: List,
        data: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        asset: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Matchup]:
        """Generate first round of tournament matchups"""
        
        # Shuffle strategies for random seeding
        shuffled = strategies.copy()
        random.shuffle(shuffled)
        
        # Pair them up
        matchups = []
        for i in range(0, len(shuffled) - 1, 2):
            matchups.append(Matchup(
                strategy_a=shuffled[i],
                strategy_b=shuffled[i + 1],
                regime=None,  # Tournament uses overall performance
                asset=asset,
                timeframe=timeframe
            ))
        
        # If odd number, one strategy gets a bye (auto-advance)
        if len(shuffled) % 2 == 1:
            print(f"Strategy {shuffled[-1].name} gets a bye to next round")
        
        self.match_history.extend(matchups)
        return matchups
    
    def generate_next_round(
        self,
        winners: List,
        asset: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Matchup]:
        """Generate next round based on winners from previous round"""
        
        if len(winners) <= 1:
            return []  # Tournament over
        
        matchups = []
        for i in range(0, len(winners) - 1, 2):
            matchups.append(Matchup(
                strategy_a=winners[i],
                strategy_b=winners[i + 1],
                regime=None,
                asset=asset,
                timeframe=timeframe
            ))
        
        # Handle odd number
        if len(winners) % 2 == 1:
            print(f"Strategy {winners[-1].name} gets a bye to next round")
        
        self.match_history.extend(matchups)
        return matchups


# ============================================================================
# SWISS SYSTEM MATCHMAKER
# ============================================================================

class SwissSystemMatcher(MatchMaker):
    """
    Swiss tournament system - pair strategies with similar records.
    Good for finding relative rankings without full round-robin.
    """
    
    def __init__(self, num_rounds: int = 5):
        """
        Args:
            num_rounds: Number of rounds to play
        """
        super().__init__()
        self.num_rounds = num_rounds
        self.strategy_records: Dict[str, Dict] = {}  # Track wins/losses
    
    def generate_matchups(
        self,
        strategies: List,
        data: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        asset: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Matchup]:
        """Generate first round (random pairings)"""
        
        # Initialize records
        for strat in strategies:
            if strat.name not in self.strategy_records:
                self.strategy_records[strat.name] = {
                    'wins': 0,
                    'losses': 0,
                    'opponents': set()
                }
        
        # First round is random
        shuffled = strategies.copy()
        random.shuffle(shuffled)
        
        matchups = []
        for i in range(0, len(shuffled) - 1, 2):
            matchups.append(Matchup(
                strategy_a=shuffled[i],
                strategy_b=shuffled[i + 1],
                regime=None,
                asset=asset,
                timeframe=timeframe
            ))
        
        self.match_history.extend(matchups)
        return matchups
    
    def update_records(self, winner_name: str, loser_name: str):
        """Update win/loss records after a match"""
        self.strategy_records[winner_name]['wins'] += 1
        self.strategy_records[winner_name]['opponents'].add(loser_name)
        
        self.strategy_records[loser_name]['losses'] += 1
        self.strategy_records[loser_name]['opponents'].add(winner_name)
    
    def generate_next_round(
        self,
        strategies: List,
        asset: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Matchup]:
        """Generate next round by pairing similar records"""
        
        # Sort by record (wins - losses)
        sorted_strats = sorted(
            strategies,
            key=lambda s: self.strategy_records[s.name]['wins'] - 
                         self.strategy_records[s.name]['losses'],
            reverse=True
        )
        
        matchups = []
        used = set()
        
        # Pair adjacent strategies in standings who haven't played yet
        for i, strat_a in enumerate(sorted_strats):
            if strat_a.name in used:
                continue
            
            # Find best opponent (similar record, haven't played)
            for j in range(i + 1, len(sorted_strats)):
                strat_b = sorted_strats[j]
                
                if strat_b.name in used:
                    continue
                
                # Check if they've played before
                if strat_b.name not in self.strategy_records[strat_a.name]['opponents']:
                    matchups.append(Matchup(
                        strategy_a=strat_a,
                        strategy_b=strat_b,
                        regime=None,
                        asset=asset,
                        timeframe=timeframe
                    ))
                    used.add(strat_a.name)
                    used.add(strat_b.name)
                    break
        
        self.match_history.extend(matchups)
        return matchups


# ============================================================================
# ADAPTIVE MATCHMAKER
# ============================================================================

class AdaptiveMatcher(MatchMaker):
    """
    Prioritizes matchups that will be most informative.
    Focuses on close-skill matchups and under-tested pairs.
    """
    
    def __init__(self, budget: int = 100):
        """
        Args:
            budget: Maximum number of matchups to generate
        """
        super().__init__()
        self.budget = budget
        self.matchup_counts: Dict[str, int] = {}  # Track how often pairs are tested
    
    def generate_matchups(
        self,
        strategies: List,
        data: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        asset: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Matchup]:
        """Generate matchups prioritizing informative comparisons"""
        
        matchups = []
        
        # Get all possible pairs
        all_pairs = list(combinations(strategies, 2))
        
        # Score each pair by informativeness
        pair_scores = []
        for strat_a, strat_b in all_pairs:
            pair_key = f"{strat_a.name}_{strat_b.name}"
            
            # Penalize pairs we've already tested a lot
            test_count = self.matchup_counts.get(pair_key, 0)
            
            # Reward pairs with similar characteristics (harder to predict winner)
            similarity = self._calculate_similarity(strat_a, strat_b)
            
            # Score = similarity / (1 + test_count)
            score = similarity / (1 + test_count)
            pair_scores.append((score, strat_a, strat_b))
        
        # Sort by score (highest first) and take top budget
        pair_scores.sort(reverse=True, key=lambda x: x[0])
        
        regimes_to_test = [None]
        if regimes is not None:
            regimes_to_test.extend(regimes.unique())
        
        # Generate matchups from top-scored pairs
        matches_per_regime = self.budget // len(regimes_to_test)
        
        for regime in regimes_to_test:
            for i in range(min(matches_per_regime, len(pair_scores))):
                _, strat_a, strat_b = pair_scores[i]
                
                matchup = Matchup(
                    strategy_a=strat_a,
                    strategy_b=strat_b,
                    regime=regime,
                    asset=asset,
                    timeframe=timeframe
                )
                matchups.append(matchup)
                
                # Update count
                pair_key = f"{strat_a.name}_{strat_b.name}"
                self.matchup_counts[pair_key] = self.matchup_counts.get(pair_key, 0) + 1
        
        self.match_history.extend(matchups)
        return matchups
    
    def _calculate_similarity(self, strat_a, strat_b) -> float:
        """
        Calculate similarity between two strategies.
        Higher similarity = harder to predict winner = more informative.
        """
        # Simple heuristic: same type = more similar
        type_match = 1.0 if strat_a.strategy_type == strat_b.strategy_type else 0.5
        
        # Could extend with more sophisticated similarity metrics
        return type_match


# ============================================================================
# MATCHMAKER REGISTRY
# ============================================================================

class MatchMakerRegistry:
    """Central registry for matchmaker strategies"""
    
    def __init__(self):
        self.matchmakers: Dict[str, MatchMaker] = {}
    
    def register(self, name: str, matchmaker: MatchMaker):
        """Register a matchmaker"""
        self.matchmakers[name] = matchmaker
    
    def get(self, name: str) -> Optional[MatchMaker]:
        """Get matchmaker by name"""
        return self.matchmakers.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered matchmakers"""
        return list(self.matchmakers.keys())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Matchmaker v1 - Example Usage")
    print("="*60)
    
    # Create dummy strategies
    class DummyStrategy:
        def __init__(self, name, strat_type):
            self.name = name
            self.strategy_type = strat_type
    
    strategies = [
        DummyStrategy("SMA_Fast", "momentum"),
        DummyStrategy("SMA_Slow", "momentum"),
        DummyStrategy("BollingerMR", "mean_reversion"),
        DummyStrategy("RSI", "momentum"),
        DummyStrategy("BuyHold", "passive"),
    ]
    
    # Dummy data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({'close': np.random.randn(100).cumsum() + 100}, index=dates)
    regimes = pd.Series(np.random.choice([0, 1], 100), index=dates)
    
    print(f"\nTesting with {len(strategies)} strategies")
    
    # Test Round Robin
    print("\n" + "-"*60)
    print("1. Round Robin Matcher")
    print("-"*60)
    rr = RoundRobinMatcher(include_global=True)
    matchups = rr.generate_matchups(strategies, data, regimes)
    print(f"Generated {len(matchups)} matchups")
    print(f"Expected: {rr.estimate_match_count(len(strategies), 2)}")
    print(f"Sample: {matchups[0]}")
    
    # Test Random Sampling
    print("\n" + "-"*60)
    print("2. Random Sampling Matcher")
    print("-"*60)
    rs = RandomSamplingMatcher(num_matches=20, seed=42)
    matchups = rs.generate_matchups(strategies, data, regimes)
    print(f"Generated {len(matchups)} random matchups")
    print(f"Samples:")
    for m in matchups[:3]:
        print(f"  {m}")
    
    # Test Tournament
    print("\n" + "-"*60)
    print("3. Tournament Matcher")
    print("-"*60)
    tm = TournamentMatcher(style='single')
    matchups = tm.generate_matchups(strategies, data)
    print(f"Round 1: {len(matchups)} matchups")
    for m in matchups:
        print(f"  {m}")
    
    # Test Swiss System
    print("\n" + "-"*60)
    print("4. Swiss System Matcher")
    print("-"*60)
    swiss = SwissSystemMatcher(num_rounds=3)
    matchups = swiss.generate_matchups(strategies, data)
    print(f"Round 1: {len(matchups)} matchups")
    
    # Simulate some results and generate round 2
    swiss.update_records(strategies[0].name, strategies[1].name)
    swiss.update_records(strategies[2].name, strategies[3].name)
    
    matchups_r2 = swiss.generate_next_round(strategies)
    print(f"Round 2: {len(matchups_r2)} matchups")
    for m in matchups_r2:
        print(f"  {m}")
    
    # Test Adaptive
    print("\n" + "-"*60)
    print("5. Adaptive Matcher")
    print("-"*60)
    adaptive = AdaptiveMatcher(budget=10)
    matchups = adaptive.generate_matchups(strategies, data, regimes)
    print(f"Generated {len(matchups)} adaptive matchups")
    print("Top matchups:")
    for m in matchups[:5]:
        print(f"  {m}")
    
    print("\n" + "="*60)
    print("Matchmaking Complete!")
    print("="*60)