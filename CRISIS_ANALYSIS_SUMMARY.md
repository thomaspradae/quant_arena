# Crisis ELO Analysis Summary

## Key Findings from Crisis Period Analysis

### Crisis Performance Summary

| Crisis Period | Best Strategy | Best Return | Buy & Hold Return | Outperformance | Days |
|---------------|---------------|-------------|-------------------|----------------|------|
| **GFC 2008** | FadeExtremes_63d | **10.5%** | -25.6% | **+36.1%** | 378 |
| **COVID 2020** | FadeExtremes_63d | **29.6%** | -3.2% | **+32.8%** | 52 |
| **2022 Selloff** | MeanReversion_20d | **21.4%** | -4.8% | **+26.2%** | 209 |
| **Dotcom Crash** | FadeExtremes_63d | **1.4%** | -22.3% | **+23.7%** | 649 |
| **Taper Tantrum** | MeanReversion_20d | **13.0%** | 1.3% | **+11.6%** | 86 |

### Key Insights

#### 1. **FadeExtremes_63d is the Crisis Champion**
- **Won 3 out of 5 major crises** (GFC 2008, COVID 2020, Dotcom Crash)
- **Average outperformance**: +30.9% vs Buy & Hold during crises
- **Strategy**: Fades extreme market movements - works when markets are oversold/overbought

#### 2. **MeanReversion_20d Excels in Specific Stress Periods**
- **Won 2 out of 5 major crises** (2022 Selloff, Taper Tantrum)
- **Average outperformance**: +18.9% vs Buy & Hold during crises
- **Strategy**: Mean reversion - works when markets are trending but volatile

#### 3. **Buy & Hold Gets Crushed During Crises**
- **Average crisis return**: -11.1%
- **Only positive crisis**: Taper Tantrum 2013 (+1.3%)
- **Worst crisis**: GFC 2008 (-25.6%)

#### 4. **ELO Rankings Don't Always Predict Crisis Performance**
- **FadeExtremes_63d**: ELO scores 0.48-0.54 (moderate)
- **MeanReversion_20d**: ELO scores 0.59-0.67 (higher)
- **Buy & Hold**: Consistently poor crisis performance despite decent overall ELO

### The Disconnect Explained

#### Why ELO Doesn't Always Translate to Crisis Returns:

1. **ELO measures overall skill** - not crisis-specific performance
2. **Crises are rare events** - ELO averages across all market conditions
3. **Buy & Hold wins long-term** - but loses badly during crises
4. **Crisis strategies are specialized** - FadeExtremes and MeanReversion are designed for stress periods

#### The Opportunity:

**Build a crisis-aware ELO system** that:
- Tracks ELO rankings during different market regimes
- Identifies crisis-specialized strategies
- Dynamically allocates based on market stress levels
- Uses uncertainty (sigma) to detect regime shifts

### Visualizations Created

1. **crisis_elo_evolution_detailed.png** - ELO evolution during each crisis period
2. **crisis_elo_comparison_analysis.png** - Comparison of ELO changes and returns
3. **crisis_elo_velocity_analysis.png** - ELO velocity (rate of change) during crises
4. **crisis_performance_summary.csv** - Detailed performance metrics

### Next Steps

1. **Regime-aware ELO tracking** - separate ELO rankings for crisis vs normal periods
2. **Crisis detection system** - use volatility and ELO uncertainty to identify stress periods
3. **Dynamic allocation** - switch to crisis strategies when stress is detected
4. **Backtesting** - test crisis-aware allocation vs static portfolios

---

*This analysis shows that while ELO rankings reveal overall strategy skill, crisis performance requires specialized strategies that excel during market stress. The key insight is that we need to build a system that can detect crises and dynamically allocate to the best crisis strategies.*










