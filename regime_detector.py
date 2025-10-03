"""
Regime Detector v1: Market state and regime detection
Initial implementations: Volatility, HMM, Change Point, Multifractal
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import warnings

# Optional imports (install as needed)
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False


# ============================================================================
# BASE REGIME DETECTOR
# ============================================================================

class RegimeDetector(ABC):
    """
    Base class for regime detection.
    
    A regime is a persistent market state (high vol, low vol, trending, etc.)
    that affects strategy performance.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.regime_labels: Optional[pd.Series] = None
        self.regime_descriptions: Dict[int, str] = {}
    
    @abstractmethod
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect market regimes in the data.
        
        Args:
            data: DataFrame with OHLCV data (datetime index)
        
        Returns:
            pd.Series with datetime index and regime labels (0, 1, 2, ...)
        """
        pass
    
    def get_regime_stats(self, regimes: pd.Series) -> pd.DataFrame:
        """Get summary statistics for each regime."""
        unique_regimes = sorted(regimes.unique())
        stats = []
        
        for regime in unique_regimes:
            mask = regimes == regime
            stats.append({
                'regime': regime,
                'count': mask.sum(),
                'percentage': mask.sum() / len(regimes) * 100,
                'description': self.regime_descriptions.get(regime, f"Regime {regime}")
            })
        
        return pd.DataFrame(stats)
    
    def get_transitions(self, regimes: pd.Series) -> pd.DataFrame:
        """Get regime transition matrix."""
        unique_regimes = sorted(regimes.unique())
        n = len(unique_regimes)
        
        # Build transition matrix
        transitions = np.zeros((n, n))
        for i in range(len(regimes) - 1):
            from_regime = regimes.iloc[i]
            to_regime = regimes.iloc[i + 1]
            from_idx = unique_regimes.index(from_regime)
            to_idx = unique_regimes.index(to_regime)
            transitions[from_idx, to_idx] += 1
        
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transitions / row_sums
        
        return pd.DataFrame(
            transition_probs,
            index=[f"From_{r}" for r in unique_regimes],
            columns=[f"To_{r}" for r in unique_regimes]
        )
    
    def __repr__(self):
        return f"{self.name}()"


# ============================================================================
# VOLATILITY REGIME DETECTOR
# ============================================================================

class VolatilityRegimeDetector(RegimeDetector):
    """
    Simple regime detector based on rolling volatility.
    Classifies periods as high/low volatility.
    """
    
    def __init__(self, window: int = 20, num_regimes: int = 2, method: str = 'quantile'):
        """
        Args:
            window: Rolling window for volatility calculation
            num_regimes: Number of volatility regimes (2=high/low, 3=high/med/low)
            method: 'quantile' or 'kmeans' for threshold determination
        """
        super().__init__()
        self.window = window
        self.num_regimes = num_regimes
        self.method = method
        self.name = f"VolRegime_{window}_{num_regimes}states"
        
        # Set regime descriptions
        if num_regimes == 2:
            self.regime_descriptions = {0: "Low Vol", 1: "High Vol"}
        elif num_regimes == 3:
            self.regime_descriptions = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}
        else:
            self.regime_descriptions = {i: f"Vol Regime {i}" for i in range(num_regimes)}
    
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect volatility regimes."""
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Calculate rolling volatility (annualized)
        vol = returns.rolling(window=self.window).std() * np.sqrt(252)
        
        # Remove NaN values
        vol = vol.dropna()
        
        if len(vol) == 0:
            raise ValueError("Not enough data to calculate volatility")
        
        # Determine regime thresholds
        if self.method == 'quantile':
            regimes = self._quantile_method(vol)
        elif self.method == 'kmeans':
            regimes = self._kmeans_method(vol)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.regime_labels = regimes
        return regimes
    
    def _quantile_method(self, vol: pd.Series) -> pd.Series:
        """Assign regimes based on quantiles."""
        regimes = pd.Series(0, index=vol.index, dtype=int)
        
        if self.num_regimes == 2:
            # Split at median
            threshold = vol.median()
            regimes[vol >= threshold] = 1
        else:
            # Split into equal-sized buckets
            quantiles = np.linspace(0, 1, self.num_regimes + 1)[1:-1]
            thresholds = vol.quantile(quantiles)
            
            for i, threshold in enumerate(thresholds):
                regimes[vol >= threshold] = i + 1
        
        return regimes
    
    def _kmeans_method(self, vol: pd.Series) -> pd.Series:
        """Assign regimes using K-means clustering."""
        # Reshape for sklearn
        X = vol.values.reshape(-1, 1)
        
        # Fit K-means
        kmeans = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Sort clusters by volatility level (0=low, 1=med, 2=high)
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_clusters = np.argsort(cluster_centers)
        label_map = {old: new for new, old in enumerate(sorted_clusters)}
        labels = np.array([label_map[l] for l in labels])
        
        return pd.Series(labels, index=vol.index)


# ============================================================================
# HIDDEN MARKOV MODEL REGIME DETECTOR
# ============================================================================

class HMMRegimeDetector(RegimeDetector):
    """
    Hidden Markov Model for regime detection.
    Discovers latent market states from observed returns.
    """
    
    def __init__(self, num_regimes: int = 2, covariance_type: str = 'full', 
                 n_iter: int = 100):
        """
        Args:
            num_regimes: Number of hidden states
            covariance_type: 'spherical', 'diag', 'full', 'tied'
            n_iter: Maximum iterations for EM algorithm
        """
        super().__init__()
        
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn not installed. Install with: pip install hmmlearn")
        
        self.num_regimes = num_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.name = f"HMM_{num_regimes}states"
        
        self.model = None
        self.regime_descriptions = {i: f"HMM State {i}" for i in range(num_regimes)}
    
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect regimes using HMM."""
        # Calculate returns and volatility as features
        returns = data['close'].pct_change().dropna()
        
        # Create feature matrix (returns and absolute returns)
        features = pd.DataFrame({
            'returns': returns,
            'abs_returns': returns.abs(),
        }).dropna()
        
        if len(features) < 10:
            raise ValueError("Not enough data for HMM (need at least 10 samples)")
        
        X = features.values
        
        # Fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.num_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X)
        
        # Predict hidden states
        hidden_states = self.model.predict(X)
        
        # Create regime series
        regimes = pd.Series(hidden_states, index=features.index)
        
        # Characterize regimes based on mean returns and volatility
        self._characterize_regimes(features, regimes)
        
        self.regime_labels = regimes
        return regimes
    
    def _characterize_regimes(self, features: pd.DataFrame, regimes: pd.Series):
        """Assign meaningful names to HMM states."""
        regime_chars = []
        
        for regime in range(self.num_regimes):
            mask = regimes == regime
            mean_return = features.loc[mask, 'returns'].mean()
            mean_vol = features.loc[mask, 'abs_returns'].mean()
            
            regime_chars.append({
                'regime': regime,
                'mean_return': mean_return,
                'volatility': mean_vol
            })
        
        # Sort by volatility and assign names
        regime_chars.sort(key=lambda x: x['volatility'])
        
        if self.num_regimes == 2:
            names = ["Low Vol", "High Vol"]
        elif self.num_regimes == 3:
            names = ["Low Vol", "Medium Vol", "High Vol"]
        else:
            names = [f"State {i}" for i in range(self.num_regimes)]
        
        for i, char in enumerate(regime_chars):
            orig_regime = char['regime']
            self.regime_descriptions[orig_regime] = names[i]


# ============================================================================
# CHANGE POINT REGIME DETECTOR
# ============================================================================

class ChangePointRegimeDetector(RegimeDetector):
    """
    Detects structural breaks in time series.
    Each segment between breakpoints is a regime.
    """
    
    def __init__(self, method: str = 'pelt', penalty: float = 10, min_size: int = 20):
        """
        Args:
            method: 'pelt', 'binseg', or 'window' (ruptures library)
            penalty: Penalty for adding breakpoints (higher = fewer breaks)
            min_size: Minimum samples between breakpoints
        """
        super().__init__()
        
        if not RUPTURES_AVAILABLE:
            raise ImportError("ruptures not installed. Install with: pip install ruptures")
        
        self.method = method
        self.penalty = penalty
        self.min_size = min_size
        self.name = f"ChangePoint_{method}"
        
        self.breakpoints = None
    
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect regimes via change point detection."""
        # Use returns as the signal
        returns = data['close'].pct_change().dropna()
        signal = returns.values.reshape(-1, 1)
        
        if len(signal) < self.min_size * 2:
            raise ValueError(f"Not enough data. Need at least {self.min_size * 2} samples.")
        
        # Choose algorithm
        if self.method == 'pelt':
            algo = rpt.Pelt(model='rbf', min_size=self.min_size)
        elif self.method == 'binseg':
            algo = rpt.Binseg(model='l2', min_size=self.min_size)
        elif self.method == 'window':
            algo = rpt.Window(width=self.min_size, model='l2')
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Fit and predict
        algo.fit(signal)
        self.breakpoints = algo.predict(pen=self.penalty)
        
        # Remove the last breakpoint (it's always the end of the series)
        if self.breakpoints and self.breakpoints[-1] == len(signal):
            self.breakpoints = self.breakpoints[:-1]
        
        # Create regime labels
        regimes = np.zeros(len(returns), dtype=int)
        for i, bp in enumerate(self.breakpoints):
            regimes[bp:] = i + 1
        
        regimes_series = pd.Series(regimes, index=returns.index)
        
        # Describe regimes
        for i in range(regimes_series.max() + 1):
            self.regime_descriptions[i] = f"Regime {i}"
        
        self.regime_labels = regimes_series
        return regimes_series
    
    def get_breakpoint_dates(self) -> List[pd.Timestamp]:
        """Get the dates where regime changes occur."""
        if self.regime_labels is None or self.breakpoints is None:
            return []
        
        dates = []
        for bp in self.breakpoints:
            if bp < len(self.regime_labels):
                dates.append(self.regime_labels.index[bp])
        return dates


# ============================================================================
# MULTIFRACTAL REGIME DETECTOR
# ============================================================================

class MultifractalRegimeDetector(RegimeDetector):
    """
    Uses multifractal properties (Hurst exponent, tail index) to detect regimes.
    Inspired by Mandelbrot's work on market fractality.
    """
    
    def __init__(self, window: int = 60, num_regimes: int = 3):
        """
        Args:
            window: Rolling window for fractal calculations
            num_regimes: Number of regimes to detect
        """
        super().__init__()
        self.window = window
        self.num_regimes = num_regimes
        self.name = f"Multifractal_{window}_{num_regimes}states"
        
        self.regime_descriptions = {i: f"Fractal Regime {i}" for i in range(num_regimes)}
    
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect regimes using multifractal analysis."""
        returns = data['close'].pct_change().dropna()
        
        # Calculate rolling Hurst exponent
        hurst = self._rolling_hurst(returns, self.window)
        
        # Calculate rolling tail index (alpha in power law)
        tail_index = self._rolling_tail_index(returns, self.window)
        
        # Combine features
        features = pd.DataFrame({
            'hurst': hurst,
            'tail_index': tail_index
        }).dropna()
        
        if len(features) < 10:
            raise ValueError("Not enough data for multifractal analysis")
        
        # Cluster based on fractal properties
        X = features.values
        kmeans = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        regimes = pd.Series(labels, index=features.index)
        
        # Characterize regimes
        self._characterize_fractal_regimes(features, regimes)
        
        self.regime_labels = regimes
        return regimes
    
    def _rolling_hurst(self, returns: pd.Series, window: int) -> pd.Series:
        """
        Calculate rolling Hurst exponent using R/S analysis.
        H < 0.5: mean-reverting
        H = 0.5: random walk
        H > 0.5: trending/persistent
        """
        def hurst_exponent(ts):
            """Simplified Hurst calculation."""
            if len(ts) < 20:
                return np.nan
            
            lags = range(2, min(20, len(ts) // 2))
            tau = []
            
            for lag in lags:
                # Calculate standard deviation of differences at this lag
                std_diff = np.std(np.diff(ts, n=lag))
                tau.append(std_diff)
            
            if len(tau) < 2:
                return np.nan
            
            # Fit log(tau) vs log(lag)
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]  # Slope is Hurst exponent
        
        hurst = returns.rolling(window=window).apply(hurst_exponent, raw=False)
        return hurst
    
    def _rolling_tail_index(self, returns: pd.Series, window: int) -> pd.Series:
        """
        Estimate tail index (alpha) using Hill estimator.
        Lower alpha = fatter tails (more extreme events)
        """
        def hill_estimator(ts, k=None):
            """Hill estimator for tail index."""
            if len(ts) < 10:
                return np.nan
            
            abs_ts = np.abs(ts)
            k = k or max(5, len(ts) // 4)  # Use top 25% for tail
            
            sorted_ts = np.sort(abs_ts)[::-1]  # Descending
            
            if sorted_ts[k-1] <= 0:
                return np.nan
            
            # Hill estimator
            alpha = k / np.sum(np.log(sorted_ts[:k] / sorted_ts[k-1]))
            return alpha
        
        tail = returns.rolling(window=window).apply(hill_estimator, raw=False)
        return tail
    
    def _characterize_fractal_regimes(self, features: pd.DataFrame, regimes: pd.Series):
        """Assign meaningful names based on fractal properties."""
        for regime in range(self.num_regimes):
            mask = regimes == regime
            mean_hurst = features.loc[mask, 'hurst'].mean()
            mean_tail = features.loc[mask, 'tail_index'].mean()
            
            # Characterize
            if mean_hurst < 0.45:
                trend = "Mean-Reverting"
            elif mean_hurst > 0.55:
                trend = "Trending"
            else:
                trend = "Random"
            
            if mean_tail < 3:
                tail = "Fat-Tailed"
            elif mean_tail > 4:
                tail = "Thin-Tailed"
            else:
                tail = "Normal-Tailed"
            
            self.regime_descriptions[regime] = f"{trend} + {tail}"


# ============================================================================
# REGIME DETECTOR REGISTRY
# ============================================================================

class RegimeDetectorRegistry:
    """Registry for managing multiple regime detectors."""
    
    def __init__(self):
        self.detectors: Dict[str, RegimeDetector] = {}
    
    def register(self, detector: RegimeDetector):
        """Add a detector to the registry."""
        self.detectors[detector.name] = detector
    
    def get(self, name: str) -> Optional[RegimeDetector]:
        """Get detector by name."""
        return self.detectors.get(name)
    
    def list_all(self) -> List[str]:
        """List all detector names."""
        return list(self.detectors.keys())
    
    def __repr__(self):
        return f"RegimeDetectorRegistry({len(self.detectors)} detectors)"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Regime Detector v1 - Example Usage")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Simulate regime changes: low vol -> high vol -> low vol
    regime_1 = np.random.randn(200) * 0.01  # Low vol
    regime_2 = np.random.randn(150) * 0.03  # High vol
    regime_3 = np.random.randn(150) * 0.01  # Low vol again
    
    returns = np.concatenate([regime_1, regime_2, regime_3])
    price = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': price,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    print(f"\nGenerated {len(data)} days of synthetic data with regime changes")
    
    # Test Volatility Regime Detector
    print("\n" + "-"*60)
    print("1. Volatility Regime Detector")
    print("-"*60)
    vol_detector = VolatilityRegimeDetector(window=20, num_regimes=2)
    vol_regimes = vol_detector.detect_regimes(data)
    print(f"Detected regimes: {vol_regimes.unique()}")
    print("\nRegime Statistics:")
    print(vol_detector.get_regime_stats(vol_regimes))
    
    # Test HMM Regime Detector (if available)
    if HMM_AVAILABLE:
        print("\n" + "-"*60)
        print("2. HMM Regime Detector")
        print("-"*60)
        hmm_detector = HMMRegimeDetector(num_regimes=2)
        hmm_regimes = hmm_detector.detect_regimes(data)
        print(f"Detected regimes: {hmm_regimes.unique()}")
        print("\nRegime Statistics:")
        print(hmm_detector.get_regime_stats(hmm_regimes))
    
    # Test Change Point Detector (if available)
    if RUPTURES_AVAILABLE:
        print("\n" + "-"*60)
        print("3. Change Point Detector")
        print("-"*60)
        cp_detector = ChangePointRegimeDetector(penalty=5)
        cp_regimes = cp_detector.detect_regimes(data)
        print(f"Detected regimes: {cp_regimes.unique()}")
        print(f"Breakpoint dates: {cp_detector.get_breakpoint_dates()}")
        print("\nRegime Statistics:")
        print(cp_detector.get_regime_stats(cp_regimes))
    
    # Test Multifractal Detector
    print("\n" + "-"*60)
    print("4. Multifractal Detector")
    print("-"*60)
    mf_detector = MultifractalRegimeDetector(window=60, num_regimes=3)
    mf_regimes = mf_detector.detect_regimes(data)
    print(f"Detected regimes: {mf_regimes.unique()}")
    print("\nRegime Statistics:")
    print(mf_detector.get_regime_stats(mf_regimes))
    
    print("\n" + "="*60)
    print("Regime Detection Complete!")
    print("="*60)