"""
Strategy Zoo: Initial Implementation with 5 Baseline Strategies
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


# ============================================================================
# BASE STRATEGY CLASS
# ============================================================================

class Strategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self):
        self.name: str = self.__class__.__name__
        self.strategy_type: str = "unknown"
        self.required_features: List[str] = ['close']
        self.forecast_horizon: int = 1  # periods ahead
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from data.
        
        Args:
            data: DataFrame with OHLCV data (datetime index)
        
        Returns:
            pd.Series with datetime index and signal values in [-1, 1]
            -1 = max short, 0 = neutral, +1 = max long
        """
        pass
    
    def warmup_period(self) -> int:
        """Number of bars needed before strategy can generate valid signals"""
        return 0
    
    def get_tags(self) -> Dict[str, Any]:
        """Metadata tags for filtering/grouping strategies"""
        return {
            'asset_class_specialization': None,
            'volatility_regime': None,
            'trend_regime': None,
            'complexity': 'simple',
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Check if data has required features"""
        missing = set(self.required_features) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return True
    
    def __repr__(self):
        return f"{self.name}(type={self.strategy_type}, horizon={self.forecast_horizon})"


# ============================================================================
# STRATEGY 1: BUY AND HOLD
# ============================================================================

class BuyAndHold(Strategy):
    """
    Baseline: Always long, never adjusts position.
    This is the benchmark everyone should beat.
    """
    
    def __init__(self):
        super().__init__()
        self.strategy_type = "passive"
        self.required_features = ['close']
        self.forecast_horizon = 1
        
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        # Always return +1 (max long)
        return pd.Series(1.0, index=data.index)
    
    def warmup_period(self) -> int:
        return 0
    
    def get_tags(self) -> Dict[str, Any]:
        return {
            'asset_class_specialization': None,
            'volatility_regime': 'any',
            'trend_regime': 'trending',
            'complexity': 'trivial',
        }


# ============================================================================
# STRATEGY 2: SIMPLE MOVING AVERAGE CROSSOVER
# ============================================================================

class SMACrossover(Strategy):
    """
    Classic trend-following: long when fast MA > slow MA, short otherwise.
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__()
        self.strategy_type = "momentum"
        self.required_features = ['close']
        self.forecast_horizon = 1
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"SMAXover_{fast_period}_{slow_period}"
        
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        
        # Calculate moving averages
        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()
        
        # Generate signal: +1 when fast > slow, -1 when fast < slow
        signal = pd.Series(0.0, index=data.index)
        signal[fast_ma > slow_ma] = 1.0
        signal[fast_ma < slow_ma] = -1.0
        
        return signal
    
    def warmup_period(self) -> int:
        return self.slow_period
    
    def get_tags(self) -> Dict[str, Any]:
        return {
            'asset_class_specialization': None,
            'volatility_regime': 'any',
            'trend_regime': 'trending',
            'complexity': 'simple',
        }


# ============================================================================
# STRATEGY 3: MEAN REVERSION (BOLLINGER BANDS)
# ============================================================================

class BollingerMeanReversion(Strategy):
    """
    Mean reversion: buy when price touches lower band, sell at upper band.
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__()
        self.strategy_type = "mean_reversion"
        self.required_features = ['close']
        self.forecast_horizon = 1
        self.period = period
        self.num_std = num_std
        self.name = f"BollingerMR_{period}_{num_std}"
        
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        
        # Calculate Bollinger Bands
        sma = data['close'].rolling(window=self.period).mean()
        std = data['close'].rolling(window=self.period).std()
        
        upper_band = sma + (self.num_std * std)
        lower_band = sma - (self.num_std * std)
        
        # Generate signal
        # Long when price is below lower band (oversold)
        # Short when price is above upper band (overbought)
        signal = pd.Series(0.0, index=data.index)
        
        # Normalize distance to bands to get continuous signal
        distance_to_middle = (data['close'] - sma) / (self.num_std * std)
        signal = -distance_to_middle.clip(-1, 1)  # Invert: buy low, sell high
        
        return signal
    
    def warmup_period(self) -> int:
        return self.period
    
    def get_tags(self) -> Dict[str, Any]:
        return {
            'asset_class_specialization': None,
            'volatility_regime': 'low_vol',
            'trend_regime': 'mean_reverting',
            'complexity': 'simple',
        }


# ============================================================================
# STRATEGY 4: MOMENTUM (RSI-BASED)
# ============================================================================

class RSIMomentum(Strategy):
    """
    Momentum based on Relative Strength Index.
    """
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__()
        self.strategy_type = "momentum"
        self.required_features = ['close']
        self.forecast_horizon = 1
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.name = f"RSI_{period}"
        
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        
        rsi = self.calculate_rsi(data['close'])
        
        # Generate signal based on RSI
        # RSI < 30: oversold (buy signal) -> +1
        # RSI > 70: overbought (sell signal) -> -1
        # Scale linearly between these zones
        signal = pd.Series(0.0, index=data.index)
        
        # Map RSI [0, 100] to signal [-1, 1]
        # RSI 50 (neutral) -> 0
        # RSI 0 (extremely oversold) -> +1
        # RSI 100 (extremely overbought) -> -1
        signal = (50 - rsi) / 50
        signal = signal.clip(-1, 1)
        
        return signal
    
    def warmup_period(self) -> int:
        return self.period + 1
    
    def get_tags(self) -> Dict[str, Any]:
        return {
            'asset_class_specialization': None,
            'volatility_regime': 'any',
            'trend_regime': 'any',
            'complexity': 'simple',
        }


# ============================================================================
# STRATEGY 5: VOLATILITY TARGETING
# ============================================================================

class VolatilityTargeting(Strategy):
    """
    Adjusts position size based on recent volatility.
    Long when vol is low (expect mean reversion), reduce when vol spikes.
    """
    
    def __init__(self, lookback: int = 20, target_vol: float = 0.15):
        super().__init__()
        self.strategy_type = "volatility"
        self.required_features = ['close']
        self.forecast_horizon = 1
        self.lookback = lookback
        self.target_vol = target_vol
        self.name = f"VolTarget_{lookback}"
        
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        
        # Calculate returns and realized volatility
        returns = data['close'].pct_change()
        realized_vol = returns.rolling(window=self.lookback).std() * np.sqrt(252)  # Annualized
        
        # Signal is inverse of volatility (scaled)
        # Lower vol -> higher position size
        # Higher vol -> lower position size
        # Always maintains long bias
        vol_ratio = self.target_vol / realized_vol.replace(0, np.nan)
        signal = vol_ratio.clip(0, 2) - 1  # Map to [-1, 1] range
        signal = signal.fillna(0)
        
        return signal
    
    def warmup_period(self) -> int:
        return self.lookback + 1
    
    def get_tags(self) -> Dict[str, Any]:
        return {
            'asset_class_specialization': None,
            'volatility_regime': 'any',
            'trend_regime': 'any',
            'complexity': 'simple',
        }


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

class StrategyRegistry:
    """
    Centralized registry for all strategies.
    Makes it easy to instantiate and manage the strategy universe.
    """
    
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        
    def register(self, strategy: Strategy) -> None:
        """Add a strategy to the registry"""
        self.strategies[strategy.name] = strategy
        
    def get(self, name: str) -> Optional[Strategy]:
        """Retrieve a strategy by name"""
        return self.strategies.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered strategy names"""
        return list(self.strategies.keys())
    
    def filter_by_type(self, strategy_type: str) -> List[Strategy]:
        """Get all strategies of a given type"""
        return [s for s in self.strategies.values() if s.strategy_type == strategy_type]
    
    def filter_by_tag(self, tag_key: str, tag_value: Any) -> List[Strategy]:
        """Filter strategies by tag"""
        return [
            s for s in self.strategies.values() 
            if s.get_tags().get(tag_key) == tag_value
        ]
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary table of all strategies"""
        summary = []
        for strat in self.strategies.values():
            tags = strat.get_tags()
            summary.append({
                'name': strat.name,
                'type': strat.strategy_type,
                'horizon': strat.forecast_horizon,
                'warmup': strat.warmup_period(),
                'complexity': tags.get('complexity'),
                'regime': tags.get('trend_regime'),
            })
        return pd.DataFrame(summary)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create registry
    registry = StrategyRegistry()
    
    # Register our 5 baseline strategies
    registry.register(BuyAndHold())
    registry.register(SMACrossover(fast_period=20, slow_period=50))
    registry.register(BollingerMeanReversion(period=20, num_std=2.0))
    registry.register(RSIMomentum(period=14))
    registry.register(VolatilityTargeting(lookback=20))
    
    # Show summary
    print("Strategy Zoo - Initial Universe:")
    print(registry.get_summary())
    print(f"\nTotal strategies: {len(registry.list_all())}")
    
    # Example: Test a strategy with dummy data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    dummy_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 2),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    print("\n" + "="*60)
    print("Testing SMA Crossover strategy on dummy data:")
    sma_strat = registry.get('SMAXover_20_50')
    signals = sma_strat.generate_signal(dummy_data)
    print(f"Generated {len(signals)} signals")
    print(f"Long signals: {(signals > 0).sum()}")
    print(f"Short signals: {(signals < 0).sum()}")
    print(f"Neutral signals: {(signals == 0).sum()}")