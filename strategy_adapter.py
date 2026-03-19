"""
Strategy Adapter - Compatibility layer between old and new strategy interfaces

The backtest engine expects strategies with generate_signal() method,
but the new strategy_zoo uses generate_positions() method.
This adapter bridges the gap.
"""

from typing import List
import pandas as pd
import numpy as np


class StrategyAdapter:
    """
    Adapter to make new strategy_zoo strategies compatible with old backtest engine.
    Converts generate_positions() calls to generate_signal() calls.
    """
    
    def __init__(self, strategy, capital: float = 100000):
        """
        Args:
            strategy: Strategy from strategy_zoo (has generate_positions method)
            capital: Capital for position sizing
        """
        self.strategy = strategy
        self.capital = capital
        
        # Copy all attributes from original strategy
        for attr in dir(strategy):
            if not attr.startswith('_') and attr != 'generate_positions':
                setattr(self, attr, getattr(strategy, attr))
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Convert generate_positions() to generate_signal() format.
        
        The backtest engine expects signals as fractions of capital (0 to 1),
        but generate_positions() returns dollar amounts.
        """
        # Get dollar positions
        positions = self.strategy.generate_positions(data, self.capital)
        
        # Convert to signal format (fraction of capital)
        signals = positions / self.capital
        
        # Clip to reasonable range
        signals = signals.clip(-1.0, 1.0)
        
        return signals
    
    def __repr__(self):
        return f"StrategyAdapter({self.strategy})"


def adapt_strategies(strategies: List, capital: float = 100000) -> List[StrategyAdapter]:
    """
    Convert a list of strategy_zoo strategies to be compatible with backtest engine.
    
    Args:
        strategies: List of strategies from strategy_zoo
        capital: Capital for position sizing
    
    Returns:
        List of adapted strategies
    """
    return [StrategyAdapter(strategy, capital) for strategy in strategies]


# Example usage
if __name__ == "__main__":
    from strategy_zoo import BuyAndHold, TrendFollowing
    
    # Create original strategies
    buy_hold = BuyAndHold()
    trend_follow = TrendFollowing(lookback=50)
    
    # Adapt them for backtest engine
    adapted_strategies = adapt_strategies([buy_hold, trend_follow], capital=100000)
    
    print("Original strategies:")
    print(f"  {buy_hold}")
    print(f"  {trend_follow}")
    
    print("\nAdapted strategies:")
    for adapted in adapted_strategies:
        print(f"  {adapted}")
        print(f"    Has generate_signal: {hasattr(adapted, 'generate_signal')}")
        print(f"    Has generate_positions: {hasattr(adapted, 'generate_positions')}")
