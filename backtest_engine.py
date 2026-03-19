"""
Backtest Engine v1.1: Added Beta Neutralization and Tail Risk Metrics
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import warnings


# ============================================================================
# BETA NEUTRALIZER (NEW)
# ============================================================================

class BetaNeutralizer:
    """
    Separates alpha from beta by computing market-neutral returns.
    This is critical - without it, strategies are just leveraged market bets.
    """
    
    def __init__(self, benchmark_symbol: str = 'SPY'):
        self.benchmark_symbol = benchmark_symbol
    
    def neutralize(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Tuple[pd.Series, float]:
        """
        Compute alpha (market-neutral returns) and beta.
        
        Returns:
            (alpha_returns, beta)
        """
        # Align the series
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 10:
            # Not enough data for reliable beta estimate
            return strategy_returns, np.nan
        
        # Compute beta via regression
        cov_matrix = np.cov(aligned['strategy'], aligned['benchmark'])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
        
        # Alpha = strategy return - beta * benchmark return
        alpha = aligned['strategy'] - beta * aligned['benchmark']
        
        # Reindex to match original strategy returns
        alpha = alpha.reindex(strategy_returns.index, fill_value=0)
        
        return alpha, beta

# ============================================================================
# TRANSACTION COST MODELS (unchanged)
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
        self.commission_pct = commission_pct
        self.spread_pct = spread_pct
    
    def calculate_cost(self, trade_value: float, data: pd.Series) -> float:
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
        self.commission_pct = commission_pct
        self.spread_pct = spread_pct
        self.impact_coef = impact_coef
    
    def calculate_cost(self, trade_value: float, data: pd.Series) -> float:
        commission = abs(trade_value) * self.commission_pct
        spread_cost = abs(trade_value) * self.spread_pct
        
        if 'volume' in data.index:
            volume = data['volume']
            price = data['close']
            trade_shares = abs(trade_value) / price
            
            if volume > 0:
                participation_rate = trade_shares / volume
                impact = self.impact_coef * price * np.sqrt(participation_rate)
                market_impact = abs(trade_value) * (impact / price)
            else:
                market_impact = 0
        else:
            market_impact = 0
        
        return commission + spread_cost + market_impact


# ============================================================================
# POSITION SIZING METHODS (unchanged)
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
        pass


class FixedFractionalSizer(PositionSizer):
    """Size positions as fixed fraction of capital"""
    
    def __init__(self, fraction: float = 1.0):
        self.fraction = fraction
    
    def size_positions(
        self,
        signals: pd.Series,
        data: pd.DataFrame,
        capital: float
    ) -> pd.Series:
        positions = signals * capital * self.fraction
        return positions


class VolatilityTargetingSizer(PositionSizer):
    """Size positions to target specific volatility"""
    
    def __init__(self, target_vol: float = 0.15, lookback: int = 20):
        self.target_vol = target_vol
        self.lookback = lookback
    
    def size_positions(
        self,
        signals: pd.Series,
        data: pd.DataFrame,
        capital: float
    ) -> pd.Series:
        returns = data['close'].pct_change()
        realized_vol = returns.rolling(window=self.lookback).std() * np.sqrt(252)
        realized_vol = realized_vol.replace(0, np.nan).fillna(self.target_vol)
        
        vol_scalar = self.target_vol / realized_vol
        vol_scalar = vol_scalar.clip(0, 2)
        
        positions = signals * capital * vol_scalar
        return positions


# ============================================================================
# SLIPPAGE MODELS (unchanged)
# ============================================================================

class SlippageModel(ABC):
    """Base class for slippage simulation"""
    
    @abstractmethod
    def apply_slippage(self, price: float, trade_value: float, data: pd.Series) -> float:
        pass


class FixedSlippage(SlippageModel):
    """Fixed basis points of slippage"""
    
    def __init__(self, slippage_bps: float = 5.0):
        self.slippage_bps = slippage_bps
    
    def apply_slippage(self, price: float, trade_value: float, data: pd.Series) -> float:
        slippage_pct = self.slippage_bps / 10000
        
        if trade_value > 0:
            execution_price = price * (1 + slippage_pct)
        elif trade_value < 0:
            execution_price = price * (1 - slippage_pct)
        else:
            execution_price = price
        
        return execution_price


class VolumeBasedSlippage(SlippageModel):
    """Slippage based on trade size vs volume"""
    
    def __init__(self, base_slippage_bps: float = 2.0, volume_impact: float = 0.5):
        self.base_slippage_bps = base_slippage_bps
        self.volume_impact = volume_impact
    
    def apply_slippage(self, price: float, trade_value: float, data: pd.Series) -> float:
        base_slip = self.base_slippage_bps / 10000
        
        if 'volume' in data.index and data['volume'] > 0:
            trade_shares = abs(trade_value) / price
            volume_pct = trade_shares / data['volume']
            additional_slip = self.volume_impact * volume_pct
        else:
            additional_slip = 0
        
        total_slip = base_slip + additional_slip
        
        if trade_value > 0:
            execution_price = price * (1 + total_slip)
        elif trade_value < 0:
            execution_price = price * (1 - total_slip)
        else:
            execution_price = price
        
        return execution_price


# ============================================================================
# BACKTEST RESULT (UPDATED)
# ============================================================================

@dataclass
class BacktestResult:
    """Container for backtest results"""
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: Dict[str, float]
    
    # NEW: Alpha/Beta decomposition
    alpha_returns: Optional[pd.Series] = None
    beta: Optional[float] = None
    tail_metrics: Optional[Dict[str, float]] = None
    
    def summary(self) -> pd.Series:
        """Get summary of key metrics"""
        return pd.Series(self.metrics)


# ============================================================================
# BACKTEST ENGINE (UPDATED)
# ============================================================================

class BacktestEngine:
    """
    Main backtesting engine with beta neutralization and tail risk metrics.
    """
    
    def __init__(
        self,
        transaction_model: Optional[TransactionCostModel] = None,
        sizing_method: Optional[PositionSizer] = None,
        slippage_model: Optional[SlippageModel] = None,
        benchmark_symbol: str = 'SPY'
    ):
        self.transaction_model = transaction_model or SimpleTransactionCost()
        self.sizing_method = sizing_method or FixedFractionalSizer(fraction=1.0)
        self.slippage_model = slippage_model or FixedSlippage(slippage_bps=5.0)
        self.beta_neutralizer = BetaNeutralizer(benchmark_symbol)
    
    def run_backtest(
        self,
        strategy,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        benchmark_data: Optional[pd.DataFrame] = None,
        regime: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Run a backtest with beta neutralization and tail metrics.
        
        Args:
            strategy: Strategy instance
            data: DataFrame with OHLCV data
            initial_capital: Starting capital
            benchmark_data: Benchmark data (e.g., SPY) for beta calc
            regime: Optional regime labels
        """
        if data.empty:
            raise ValueError("Data is empty")
        
        # Generate signals
        signals = strategy.generate_signal(data)
        
        # Respect warmup period
        warmup = strategy.warmup_period()
        if warmup > 0:
            signals.iloc[:warmup] = 0
        
        signals = signals.reindex(data.index, fill_value=0)
        
        # Size positions
        positions = self.sizing_method.size_positions(signals, data, initial_capital)
        
        # Simulate execution
        trades, equity_curve = self._simulate_execution(
            positions, data, initial_capital
        )
        
        # Calculate returns
        returns = equity_curve.pct_change().fillna(0)
        
        # Compute alpha and beta if benchmark provided
        alpha_returns = None
        beta = None
        if benchmark_data is not None:
            benchmark_returns = benchmark_data['close'].pct_change().fillna(0)
            alpha_returns, beta = self.beta_neutralizer.neutralize(returns, benchmark_returns)
        
        # Compute standard metrics
        metrics = self._compute_metrics(returns, equity_curve, positions, trades)
        
        # Compute tail metrics (NEW - Taleb-style)
        tail_metrics = None
        if benchmark_data is not None:
            benchmark_returns = benchmark_data['close'].pct_change().fillna(0)
            tail_metrics = self._compute_tail_metrics(returns, benchmark_returns)
            # Add tail metrics to main metrics dict
            metrics.update({f'tail_{k}': v for k, v in tail_metrics.items()})
        
        return BacktestResult(
            returns=returns,
            positions=positions,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            alpha_returns=alpha_returns,
            beta=beta,
            tail_metrics=tail_metrics
        )
    
    def _compute_tail_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Compute Taleb-style tail risk metrics (non-variance-based).
        
        These metrics focus on survival, not optimization.
        """
        # Align returns
        aligned = pd.DataFrame({
            'strategy': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 10:
            return {
                'stress_mean': 0.0,
                'stress_positive_pct': 0.0,
                'stress_positive_days': 0,
                'worst_day': 0.0,
                'recovery_days': 0,
                'tail_ratio': 1.0
            }
        
        # Find worst 5% of benchmark days (market stress periods)
        stress_threshold = aligned['benchmark'].quantile(0.05)
        stress_mask = aligned['benchmark'] <= stress_threshold
        
        if stress_mask.sum() == 0:
            stress_mean = 0.0
            stress_positive_pct = 0.0
            stress_positive_days = 0
        else:
            stress_returns = aligned.loc[stress_mask, 'strategy']
            stress_mean = stress_returns.mean()
            stress_positive_days = int((stress_returns > 0).sum())
            stress_positive_pct = stress_positive_days / len(stress_returns)
        
        # Maximum drawdown (NEW - time to recovery)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Time to recovery from max drawdown
        max_dd_idx = drawdown.idxmin()
        if max_dd_idx is not None:
            post_dd = cumulative.loc[max_dd_idx:]
            recovery_idx = post_dd[post_dd >= running_max.loc[max_dd_idx]].first_valid_index()
            if recovery_idx is not None:
                recovery_days = (recovery_idx - max_dd_idx).days
            else:
                recovery_days = len(post_dd)  # Still underwater
        else:
            recovery_days = 0
        
        # Tail ratio (gain/loss asymmetry)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            tail_ratio = abs(positive_returns.mean()) / abs(negative_returns.mean())
        else:
            tail_ratio = 1.0
        
        return {
            'stress_mean': float(stress_mean),  # Avg return during market stress
            'stress_positive_pct': float(stress_positive_pct),  # % of stress days positive
            'stress_positive_days': stress_positive_days,  # Count of positive stress days
            'worst_day': float(returns.min()),  # Single worst day
            'recovery_days': recovery_days,  # Days to recover from max DD
            'tail_ratio': float(tail_ratio)  # Upside/downside asymmetry
        }
    
    def _simulate_execution(
        self,
        positions: pd.Series,
        data: pd.DataFrame,
        initial_capital: float
    ) -> tuple:
        """Simulate realistic trade execution (unchanged)"""
        trades_list = []
        cash = initial_capital
        shares = 0
        equity = [initial_capital]
        
        for i in range(len(data)):
            date = data.index[i]
            current_data = data.iloc[i]
            price = current_data['close']
            
            target_position = positions.iloc[i] if i < len(positions) else 0
            current_position_value = shares * price
            trade_value = target_position - current_position_value
            
            if abs(trade_value) > 1:
                execution_price = self.slippage_model.apply_slippage(
                    price, trade_value, current_data
                )
                
                transaction_cost = self.transaction_model.calculate_cost(
                    trade_value, current_data
                )
                
                shares_traded = trade_value / execution_price
                shares += shares_traded
                cash -= (trade_value + transaction_cost)
                
                trades_list.append({
                    'date': date,
                    'price': price,
                    'execution_price': execution_price,
                    'shares': shares_traded,
                    'value': trade_value,
                    'cost': transaction_cost,
                    'position_after': shares
                })
            
            portfolio_value = cash + (shares * price)
            equity.append(portfolio_value)
        
        if trades_list:
            trades_df = pd.DataFrame(trades_list)
            trades_df.set_index('date', inplace=True)
        else:
            trades_df = pd.DataFrame()
        
        equity_curve = pd.Series(equity[1:], index=data.index)
        
        return trades_df, equity_curve
    
    def _compute_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        positions: pd.Series,
        trades: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute performance metrics (unchanged)"""
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = annual_return / downside_std if downside_std > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        winning_days = (returns > 0).sum()
        total_days = len(returns[returns != 0])
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        num_trades = len(trades) if not trades.empty else 0
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