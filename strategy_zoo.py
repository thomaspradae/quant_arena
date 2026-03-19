"""
Strategy Zoo v2: TRULY DIVERSE Strategies
Each exploits a different market pattern (orthogonal alpha sources)

Key improvement: Strategies now return SIZED POSITIONS, not just binary signals
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


# ============================================================================
# BASE STRATEGY CLASS (UPDATED)
# ============================================================================

class Strategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self):
        self.name: str = self.__class__.__name__
        self.strategy_type: str = "unknown"
        self.required_features: List[str] = ['close']
        
    @abstractmethod
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        """
        Generate dollar positions (NOT just signals).
        
        Args:
            data: DataFrame with OHLCV data
            capital: Available capital for sizing
        
        Returns:
            pd.Series with dollar positions (positive=long, negative=short, 0=flat)
        """
        pass
    
    def warmup_period(self) -> int:
        """Bars needed before valid signals"""
        return 0
    
    def __repr__(self):
        return f"{self.name}(type={self.strategy_type})"


# ============================================================================
# 1. BUY AND HOLD (Benchmark)
# ============================================================================

class BuyAndHold(Strategy):
    """
    Always 100% long. The benchmark everyone must beat.
    """
    
    def __init__(self):
        super().__init__()
        self.strategy_type = "passive"
        self.name = "BuyAndHold"
        
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        # Always invest full capital
        return pd.Series(capital, index=data.index)
    
    def warmup_period(self) -> int:
        return 0


# ============================================================================
# 2. TREND FOLLOWING (Momentum)
# ============================================================================

class TrendFollowing(Strategy):
    """
    Follow the trend: Long when price > MA, flat when price < MA.
    Wins: Strong bull/bear markets
    Loses: Choppy sideways markets
    """
    
    def __init__(self, lookback: int = 50):
        super().__init__()
        self.strategy_type = "momentum"
        self.lookback = lookback
        self.name = f"TrendFollow_{lookback}d"
        
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        # Trend strength: distance from moving average
        ma = data['close'].rolling(self.lookback).mean()
        trend_strength = (data['close'] - ma) / ma
        
        # Position size scales with trend strength (0 to 100% capital)
        # Strong uptrend: full capital, weak trend: reduced exposure
        position_scalar = trend_strength.clip(0, 0.20) * 5  # Scale to [0, 1]
        
        positions = position_scalar * capital
        return positions
    
    def warmup_period(self) -> int:
        return self.lookback


# ============================================================================
# 3. MEAN REVERSION (Buy Dips)
# ============================================================================

class MeanReversion(Strategy):
    """
    Buy oversold, sell overbought. Bet on reversion to mean.
    Wins: Range-bound choppy markets
    Loses: Strong trending markets (catches falling knives)
    """
    
    def __init__(self, lookback: int = 20, entry_threshold: float = -1.5):
        super().__init__()
        self.strategy_type = "mean_reversion"
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.name = f"MeanReversion_{lookback}d"
        
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        # Z-score: how many std devs from mean
        returns = data['close'].pct_change()
        mean = returns.rolling(self.lookback).mean()
        std = returns.rolling(self.lookback).std()
        z_score = (returns - mean) / std
        
        # Buy when oversold (z < -1.5), sell when overbought (z > 1.5)
        # Position size scales with extremity
        position_scalar = (-z_score / 2).clip(0, 1)  # Inverted: buy dips
        
        positions = position_scalar * capital
        return positions
    
    def warmup_period(self) -> int:
        return self.lookback


# ============================================================================
# 4. LOW VOLATILITY (Avoid Chaos)
# ============================================================================

class LowVolatility(Strategy):
    """
    Stay invested during calm, reduce exposure during volatility.
    Wins: Calm grinding bull markets
    Loses: Misses sharp V-shaped recoveries
    """
    
    def __init__(self, lookback: int = 20, vol_threshold_percentile: float = 0.7):
        super().__init__()
        self.strategy_type = "low_volatility"
        self.lookback = lookback
        self.vol_threshold = vol_threshold_percentile
        self.name = f"LowVol_{lookback}d"
        
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        # Realized volatility
        returns = data['close'].pct_change()
        vol = returns.rolling(self.lookback).std() * np.sqrt(252)
        
        # Historical vol percentile
        vol_rank = vol.rolling(252).rank(pct=True)
        
        # Reduce exposure when vol is high
        # Low vol (rank < 0.3): full capital
        # High vol (rank > 0.7): 30% capital
        position_scalar = 1.0 - (vol_rank.clip(0.3, 1.0) - 0.3) / 0.7 * 0.7
        
        positions = position_scalar * capital
        return positions
    
    def warmup_period(self) -> int:
        return self.lookback + 252


# ============================================================================
# 5. VOLATILITY BREAKOUT (Ride Chaos)
# ============================================================================

class VolatilityBreakout(Strategy):
    """
    Bet BIG during volatility spikes, flat during calm.
    Wins: Crisis events (COVID, 2008)
    Loses: False alarms during calm markets
    """
    
    def __init__(self, lookback: int = 20, spike_threshold: float = 0.8):
        super().__init__()
        self.strategy_type = "volatility_breakout"
        self.lookback = lookback
        self.spike_threshold = spike_threshold
        self.name = f"VolBreakout_{lookback}d"
        
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        # Volatility spikes
        returns = data['close'].pct_change()
        vol = returns.rolling(self.lookback).std() * np.sqrt(252)
        vol_rank = vol.rolling(252).rank(pct=True)
        
        # Only trade during vol spikes (top 20%)
        is_spike = vol_rank > self.spike_threshold
        
        # Direction: follow momentum during spike
        momentum = returns.rolling(5).mean()
        direction = np.sign(momentum)
        
        # Position: 100% capital during spikes, 0% otherwise
        position_scalar = is_spike.astype(float) * direction
        
        positions = position_scalar * capital
        return positions
    
    def warmup_period(self) -> int:
        return self.lookback + 252


# ============================================================================
# 6. FADE EXTREMES (Contrarian)
# ============================================================================

class FadeExtremes(Strategy):
    """
    Bet against extreme moves (>3 std dev days).
    Wins: After panic sells or euphoric rallies that overshoot
    Loses: When panic continues for multiple days
    """
    
    def __init__(self, lookback: int = 63, extreme_threshold: float = 2.5):
        super().__init__()
        self.strategy_type = "contrarian"
        self.lookback = lookback
        self.extreme_threshold = extreme_threshold
        self.name = f"FadeExtremes_{lookback}d"
        
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        # Detect extreme moves
        returns = data['close'].pct_change()
        mean = returns.rolling(self.lookback).mean()
        std = returns.rolling(self.lookback).std()
        z_score = (returns - mean) / std
        
        # Fade when abs(z) > threshold
        is_extreme = z_score.abs() > self.extreme_threshold
        
        # Bet AGAINST the direction (fade)
        fade_direction = -np.sign(z_score)
        
        # Position: 100% when extreme, 0% otherwise
        position_scalar = is_extreme.astype(float) * fade_direction
        
        positions = position_scalar * capital
        return positions
    
    def warmup_period(self) -> int:
        return self.lookback


# ============================================================================
# 7. MOMENTUM CROSSOVER (Classic)
# ============================================================================

class MomentumCrossover(Strategy):
    """
    Fast MA > Slow MA = Long, else Flat.
    Different from TrendFollowing: binary signal, no gradual scaling.
    """
    
    def __init__(self, fast: int = 20, slow: int = 50):
        super().__init__()
        self.strategy_type = "momentum"
        self.fast = fast
        self.slow = slow
        self.name = f"MomXover_{fast}_{slow}"
        
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        fast_ma = data['close'].rolling(self.fast).mean()
        slow_ma = data['close'].rolling(self.slow).mean()
        
        # Binary: full capital when fast > slow, 0 otherwise
        signal = (fast_ma > slow_ma).astype(float)
        
        positions = signal * capital
        return positions
    
    def warmup_period(self) -> int:
        return self.slow


# ============================================================================
# 8. RSI MEAN REVERSION
# ============================================================================

class RSIMeanReversion(Strategy):
    """
    Buy when RSI < 30 (oversold), reduce when RSI > 70 (overbought).
    Similar to MeanReversion but uses RSI instead of z-score.
    """
    
    def __init__(self, period: int = 14):
        super().__init__()
        self.strategy_type = "mean_reversion"
        self.period = period
        self.name = f"RSI_{period}"
        
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        rsi = self.calculate_rsi(data['close'])
        
        # Scale position with RSI distance from neutral (50)
        # RSI 30: full long, RSI 70: flat, RSI 50: 50% long
        position_scalar = ((70 - rsi) / 40).clip(0, 1)
        
        positions = position_scalar * capital
        return positions
    
    def warmup_period(self) -> int:
        return self.period + 1


# ============================================================================
# 9. RANGE BREAKOUT
# ============================================================================

class RangeBreakout(Strategy):
    """
    Trade breakouts from consolidation (low ATR → high ATR).
    Wins: When consolidation leads to explosive moves
    Loses: False breakouts in choppy markets
    """
    
    def __init__(self, lookback: int = 20):
        super().__init__()
        self.strategy_type = "breakout"
        self.lookback = lookback
        self.name = f"RangeBreak_{lookback}d"
        self.required_features = ['close', 'high', 'low']
        
    def generate_positions(self, data: pd.DataFrame, capital: float = 100000) -> pd.Series:
        # ATR (volatility)
        high_low = data['high'] - data['low']
        atr = high_low.rolling(self.lookback).mean()
        atr_rank = atr.rolling(126).rank(pct=True)
        
        # Breakout: when ATR moves from low to high
        is_consolidation = atr_rank < 0.3  # Bottom 30%
        breaks_out = (atr_rank > 0.5) & is_consolidation.shift(1)
        
        # Direction: follow momentum on breakout
        momentum = data['close'].pct_change(5)
        direction = np.sign(momentum)
        
        # Position: 100% during breakout phase, decay over 10 days
        breakout_signal = breaks_out.astype(float)
        decayed_signal = breakout_signal.rolling(10, min_periods=1).max() * (1 - (np.arange(len(data)) % 10) / 10)
        
        position_scalar = decayed_signal * direction
        
        positions = position_scalar * capital
        return positions
    
    def warmup_period(self) -> int:
        return self.lookback + 126


# ============================================================================
# STRATEGY REGISTRY (UPDATED)
# ============================================================================

class StrategyRegistry:
    """Centralized registry for all strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        
    def register(self, strategy: Strategy) -> None:
        """Add strategy to registry"""
        self.strategies[strategy.name] = strategy
        
    def get(self, name: str) -> Optional[Strategy]:
        """Get strategy by name"""
        return self.strategies.get(name)
    
    def list_all(self) -> List[str]:
        """List all strategy names"""
        return list(self.strategies.keys())
    
    def get_summary(self) -> pd.DataFrame:
        """Summary table of all strategies"""
        summary = []
        for strat in self.strategies.values():
            summary.append({
                'name': strat.name,
                'type': strat.strategy_type,
                'warmup': strat.warmup_period()
            })
        return pd.DataFrame(summary)
    
    def create_default_universe(self) -> 'StrategyRegistry':
        """
        Create the default strategy universe with 9 diverse strategies.
        Call this to get a pre-configured set.
        """
        self.register(BuyAndHold())
        self.register(TrendFollowing(lookback=50))
        self.register(MeanReversion(lookback=20))
        self.register(LowVolatility(lookback=20))
        self.register(VolatilityBreakout(lookback=20))
        self.register(FadeExtremes(lookback=63))
        self.register(MomentumCrossover(fast=20, slow=50))
        self.register(RSIMeanReversion(period=14))
        self.register(RangeBreakout(lookback=20))
        return self


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create registry with default universe
    registry = StrategyRegistry().create_default_universe()
    
    print("="*80)
    print("DIVERSE STRATEGY UNIVERSE")
    print("="*80)
    print(registry.get_summary().to_string(index=False))
    print(f"\nTotal strategies: {len(registry.list_all())}")
    
    # Test with dummy data
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # Simulate different market regimes
    regime1 = np.random.randn(60) * 0.01 + 0.001  # Calm uptrend
    regime2 = np.random.randn(80) * 0.03 - 0.001  # High vol downtrend
    regime3 = np.random.randn(60) * 0.01          # Choppy sideways
    
    returns = np.concatenate([regime1, regime2, regime3])
    prices = 100 * np.exp(np.cumsum(returns))
    
    dummy_data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    print("\n" + "="*80)
    print("TESTING STRATEGIES ON SIMULATED REGIMES")
    print("="*80)
    print("Regime 1 (days 0-59): Calm uptrend")
    print("Regime 2 (days 60-139): High vol downtrend")
    print("Regime 3 (days 140-199): Choppy sideways")
    
    # Test each strategy
    capital = 100000
    for strategy in registry.strategies.values():
        positions = strategy.generate_positions(dummy_data, capital)
        
        # Compute returns
        price_returns = dummy_data['close'].pct_change()
        position_pct = positions / capital  # As fraction of capital
        strategy_returns = position_pct.shift(1) * price_returns
        
        total_return = strategy_returns.sum()
        
        print(f"\n{strategy.name:25s} | Type: {strategy.strategy_type:18s} | Return: {total_return:7.2%}")
    
    print("\n" + "="*80)
    print("NOTE: Different strategies should excel in different regimes!")
    print("="*80)