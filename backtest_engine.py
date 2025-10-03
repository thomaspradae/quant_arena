"""
Backtest Engine v1: Strategy testing with realistic execution simulation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
import warnings


# ============================================================================
# TRANSACTION COST MODELS
# ============================================================================

class TransactionCostModel(ABC):
    """Base class for transaction cost models"""
    
    @abstractmethod
    def calculate_cost(self, trade_value: float, data: pd.Series) -> float:
        """Calculate cost for a trade"""
        pass


class SimpleTransactionCost(TransactionCostModel):
    """Simple fixed percentage costs"""
    
    def __init__(self, commission_pct: float = 0.001, spread_pct: float = 0.0005):
        """
        Args:
            commission_pct: Commission as % of trade value (0.001 = 0.1%)
            spread_pct: Bid-ask spread as % (0.0005 = 0.05%)
        """
        self.commission_pct = commission_pct
        self.spread_pct = spread_pct
    
    def calculate_cost(self, trade_value: float, data: pd.Series) -> float:
        """Calculate total cost"""
        commission = abs(trade_value) * self.commission_pct
        spread_cost = abs(trade_value) * self.spread_pct
        return commission + spread_cost


class RealisticTransactionCost(TransactionCostModel):
    """More realistic cost model with market impact"""
    
    def __init__(
        self,
        commission_pct: float = 0.001,
        spread_pct: float = 0.0005,
        impact_coef: float = 0.1
    ):
        """
        Args:
            commission_pct: Fixed commission
            spread_pct: Base spread
            impact_coef: Market impact coefficient (higher = more impact)
        """
        self.commission_pct = commission_pct
        self.spread_pct = spread_pct
        self.impact_coef = impact_coef
    
    def calculate_cost(self, trade_value: float, data: pd.Series) -> float:
        """Calculate costs including market impact"""
        commission = abs(trade_value) * self.commission_pct
        spread_cost = abs(trade_value) * self.spread_pct
        
        # Market impact: proportional to sqrt(trade_size / volume)
        if 'volume' in data.index:
            volume = data['volume']
            price = data['close']
            trade_shares = abs(trade_value) / price
            
            if volume > 0:
                participation_rate = trade_shares / volume
                # Square root impact model
                impact = self.impact_coef * price * np.sqrt(participation_rate)
                market_impact = abs(trade_value) * (impact / price)
            else:
                market_impact = 0
        else:
            market_impact = 0
        
        return commission + spread_cost + market_impact


# ============================================================================
# POSITION SIZING METHODS
# ============================================================================

class PositionSizer(ABC):
    """Base class for position sizing"""
    
    @abstractmethod
    def size_positions(
        self,
        signals: pd.Series,
        data: pd.DataFrame,
        capital: float
    ) -> pd.Series:
        """Convert signals to position sizes"""
        pass


class FixedFractionalSizer(PositionSizer):
    """Size positions as fixed fraction of capital"""
    
    def __init__(self, fraction: float = 1.0):
        """
        Args:
            fraction: Fraction of capital to use (1.0 = 100%, 0.5 = 50%)
        """
        self.fraction = fraction
    
    def size_positions(
        self,
        signals: pd.Series,
        data: pd.DataFrame,
        capital: float
    ) -> pd.Series:
        """Convert signals to dollar positions"""
        # Signal is in [-1, 1], scale by capital and fraction
        positions = signals * capital * self.fraction
        return positions


class VolatilityTargetingSizer(PositionSizer):
    """Size positions to target specific volatility"""
    
    def __init__(self, target_vol: float = 0.15, lookback: int = 20):
        """
        Args:
            target_vol: Target annualized volatility (0.15 = 15%)
            lookback: Window for volatility estimation
        """
        self.target_vol = target_vol
        self.lookback = lookback
    
    def size_positions(
        self,
        signals: pd.Series,
        data: pd.DataFrame,
        capital: float
    ) -> pd.Series:
        """Scale positions by inverse volatility"""
        returns = data['close'].pct_change()
        realized_vol = returns.rolling(window=self.lookback).std() * np.sqrt(252)
        
        # Avoid division by zero
        realized_vol = realized_vol.replace(0, np.nan).fillna(self.target_vol)
        
        # Scale factor: target_vol / realized_vol
        vol_scalar = self.target_vol / realized_vol
        vol_scalar = vol_scalar.clip(0, 2)  # Cap at 2x leverage
        
        # Apply to signals
        positions = signals * capital * vol_scalar
        
        return positions


# ============================================================================
# SLIPPAGE MODELS
# ============================================================================

class SlippageModel(ABC):
    """Base class for slippage simulation"""
    
    @abstractmethod
    def apply_slippage(self, price: float, trade_value: float, data: pd.Series) -> float:
        """Return execution price after slippage"""
        pass


class FixedSlippage(SlippageModel):
    """Fixed basis points of slippage"""
    
    def __init__(self, slippage_bps: float = 5.0):
        """
        Args:
            slippage_bps: Slippage in basis points (5 = 0.05%)
        """
        self.slippage_bps = slippage_bps
    
    def apply_slippage(self, price: float, trade_value: float, data: pd.Series) -> float:
        """Apply fixed slippage"""
        slippage_pct = self.slippage_bps / 10000
        
        # Slippage direction depends on trade direction
        if trade_value > 0:  # Buy
            execution_price = price * (1 + slippage_pct)
        elif trade_value < 0:  # Sell
            execution_price = price * (1 - slippage_pct)
        else:
            execution_price = price
        
        return execution_price


class VolumeBasedSlippage(SlippageModel):
    """Slippage based on trade size vs volume"""
    
    def __init__(self, base_slippage_bps: float = 2.0, volume_impact: float = 0.5):
        """
        Args:
            base_slippage_bps: Base slippage
            volume_impact: Additional slippage per % of volume traded
        """
        self.base_slippage_bps = base_slippage_bps
        self.volume_impact = volume_impact
    
    def apply_slippage(self, price: float, trade_value: float, data: pd.Series) -> float:
        """Apply volume-dependent slippage"""
        base_slip = self.base_slippage_bps / 10000
        
        # Calculate additional slippage from volume impact
        if 'volume' in data.index and data['volume'] > 0:
            trade_shares = abs(trade_value) / price
            volume_pct = trade_shares / data['volume']
            additional_slip = self.volume_impact * volume_pct
        else:
            additional_slip = 0
        
        total_slip = base_slip + additional_slip
        
        # Apply based on trade direction
        if trade_value > 0:
            execution_price = price * (1 + total_slip)
        elif trade_value < 0:
            execution_price = price * (1 - total_slip)
        else:
            execution_price = price
        
        return execution_price


# ============================================================================
# BACKTEST RESULT
# ============================================================================

@dataclass
class BacktestResult:
    """Container for backtest results"""
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: Dict[str, float]
    
    def summary(self) -> pd.Series:
        """Get summary of key metrics"""
        return pd.Series(self.metrics)


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """
    Main backtesting engine.
    Simulates strategy execution with realistic costs.
    """
    
    def __init__(
        self,
        transaction_model: Optional[TransactionCostModel] = None,
        sizing_method: Optional[PositionSizer] = None,
        slippage_model: Optional[SlippageModel] = None
    ):
        """
        Args:
            transaction_model: How to calculate transaction costs
            sizing_method: How to size positions
            slippage_model: How to simulate slippage
        """
        self.transaction_model = transaction_model or SimpleTransactionCost()
        self.sizing_method = sizing_method or FixedFractionalSizer(fraction=1.0)
        self.slippage_model = slippage_model or FixedSlippage(slippage_bps=5.0)
    
    def run_backtest(
        self,
        strategy,  # Strategy instance
        data: pd.DataFrame,
        initial_capital: float = 100000,
        regime: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Run a backtest on a strategy.
        
        Args:
            strategy: Strategy instance with generate_signal method
            data: DataFrame with OHLCV data
            initial_capital: Starting capital
            regime: Optional regime labels for filtering
        
        Returns:
            BacktestResult with all backtest outputs
        """
        # Validate data
        if data.empty:
            raise ValueError("Data is empty")
        
        # Generate signals
        signals = strategy.generate_signal(data)
        
        # Respect warmup period
        warmup = strategy.warmup_period()
        if warmup > 0:
            signals.iloc[:warmup] = 0
        
        # Align signals with data
        signals = signals.reindex(data.index, fill_value=0)
        
        # Size positions
        positions = self.sizing_method.size_positions(signals, data, initial_capital)
        
        # Simulate execution
        trades, equity_curve = self._simulate_execution(
            positions, data, initial_capital
        )
        
        # Calculate returns
        returns = equity_curve.pct_change().fillna(0)
        
        # Compute metrics
        metrics = self._compute_metrics(returns, equity_curve, positions, trades)
        
        return BacktestResult(
            returns=returns,
            positions=positions,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics
        )
    
    def _simulate_execution(
        self,
        positions: pd.Series,
        data: pd.DataFrame,
        initial_capital: float
    ) -> tuple:
        """
        Simulate realistic trade execution.
        
        Returns:
            (trades_df, equity_curve)
        """
        trades_list = []
        cash = initial_capital
        shares = 0
        equity = [initial_capital]
        
        for i in range(len(data)):
            date = data.index[i]
            current_data = data.iloc[i]
            price = current_data['close']
            
            # Desired position in dollars
            target_position = positions.iloc[i] if i < len(positions) else 0
            
            # Current position value
            current_position_value = shares * price
            
            # Calculate trade needed
            trade_value = target_position - current_position_value
            
            if abs(trade_value) > 1:  # Trade threshold ($1)
                # Apply slippage
                execution_price = self.slippage_model.apply_slippage(
                    price, trade_value, current_data
                )
                
                # Calculate transaction costs
                transaction_cost = self.transaction_model.calculate_cost(
                    trade_value, current_data
                )
                
                # Execute trade
                shares_traded = trade_value / execution_price
                shares += shares_traded
                cash -= (trade_value + transaction_cost)
                
                # Record trade
                trades_list.append({
                    'date': date,
                    'price': price,
                    'execution_price': execution_price,
                    'shares': shares_traded,
                    'value': trade_value,
                    'cost': transaction_cost,
                    'position_after': shares
                })
            
            # Calculate equity
            portfolio_value = cash + (shares * price)
            equity.append(portfolio_value)
        
        # Create trades DataFrame
        if trades_list:
            trades_df = pd.DataFrame(trades_list)
            trades_df.set_index('date', inplace=True)
        else:
            trades_df = pd.DataFrame()
        
        # Create equity curve
        equity_curve = pd.Series(equity[1:], index=data.index)
        
        return trades_df, equity_curve
    
    def _compute_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        positions: pd.Series,
        trades: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute performance metrics"""
        
        # Basic returns
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Annualized return
        years = len(returns) / 252  # Assuming daily data
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = annual_return / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_days = len(returns[returns != 0])
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Number of trades
        num_trades = len(trades) if not trades.empty else 0
        
        # Total transaction costs
        total_costs = trades['cost'].sum() if not trades.empty else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'total_costs': total_costs,
            'final_equity': equity_curve.iloc[-1]
        }
    
    def compare_strategies(
        self,
        strategies: List,
        data: pd.DataFrame,
        initial_capital: float = 100000
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Returns:
            DataFrame with metrics for each strategy
        """
        results = []
        
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, data, initial_capital)
                metrics = result.metrics.copy()
                metrics['strategy'] = strategy.name
                results.append(metrics)
            except Exception as e:
                print(f"Error backtesting {strategy.name}: {e}")
        
        return pd.DataFrame(results).set_index('strategy')


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime
    
    print("="*60)
    print("Backtest Engine v1 - Example Usage")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    returns = np.random.randn(252) * 0.02
    price = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': price,
        'volume': np.random.randint(1000000, 5000000, 252)
    }, index=dates)
    
    print(f"\nGenerated {len(data)} days of synthetic data")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Create a simple buy-and-hold strategy for testing
    class SimpleStrategy:
        def __init__(self):
            self.name = "BuyAndHold"
        
        def generate_signal(self, data):
            return pd.Series(1.0, index=data.index)
        
        def warmup_period(self):
            return 0
    
    strategy = SimpleStrategy()
    
    # Test with different cost models
    print("\n" + "-"*60)
    print("Testing Simple Transaction Costs")
    print("-"*60)
    
    engine_simple = BacktestEngine(
        transaction_model=SimpleTransactionCost(commission_pct=0.001),
        sizing_method=FixedFractionalSizer(fraction=1.0),
        slippage_model=FixedSlippage(slippage_bps=5)
    )
    
    result_simple = engine_simple.run_backtest(strategy, data, initial_capital=100000)
    
    print("\nSimple Cost Model Results:")
    print(result_simple.summary())
    print(f"\nNumber of trades: {len(result_simple.trades)}")
    print(f"Total costs: ${result_simple.metrics['total_costs']:.2f}")
    
    # Test with realistic costs
    print("\n" + "-"*60)
    print("Testing Realistic Transaction Costs")
    print("-"*60)
    
    engine_realistic = BacktestEngine(
        transaction_model=RealisticTransactionCost(impact_coef=0.1),
        sizing_method=VolatilityTargetingSizer(target_vol=0.15),
        slippage_model=VolumeBasedSlippage()
    )
    
    result_realistic = engine_realistic.run_backtest(strategy, data, initial_capital=100000)
    
    print("\nRealistic Cost Model Results:")
    print(result_realistic.summary())
    print(f"\nNumber of trades: {len(result_realistic.trades)}")
    print(f"Total costs: ${result_realistic.metrics['total_costs']:.2f}")
    
    # Compare
    print("\n" + "-"*60)
    print("Cost Impact Comparison")
    print("-"*60)
    print(f"Simple model final equity: ${result_simple.metrics['final_equity']:,.2f}")
    print(f"Realistic model final equity: ${result_realistic.metrics['final_equity']:,.2f}")
    print(f"Difference: ${result_simple.metrics['final_equity'] - result_realistic.metrics['final_equity']:,.2f}")
    
    print("\n" + "="*60)
    print("Backtest Complete!")
    print("="*60)