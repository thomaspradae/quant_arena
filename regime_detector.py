"""
Regime Detector v1: Market state and regime detection
Initial implementations: Volatility, HMM, Change Point, Multifractal

This file contains:
 - RegimeDetector base class
 - VolatilityRegimeDetector
 - HMMRegimeDetector (robustified: scaling, multiple restarts, fallback)
 - ChangePointRegimeDetector
 - MultifractalRegimeDetector
 - RegimeDetectorRegistry
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import warnings
import logging
import math

# Logging setup (simple)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports (install as needed)
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except Exception:
    RUPTURES_AVAILABLE = False

# Additional sklearn utility used by HMM
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
except Exception:
    # If sklearn not available we'll handle it gracefully later
    StandardScaler = None
    ConvergenceWarning = Warning


# ============================================================================
# Helper functions: AIC/BIC and HMM params
# ============================================================================

def _compute_aic_bic(loglik: float, num_params: int, n_obs: int) -> Tuple[float, float]:
    """
    Returns (aic, bic)
    AIC = 2k - 2 ln L
    BIC = k * ln(n) - 2 ln L
    """
    aic = 2 * num_params - 2.0 * loglik
    bic = num_params * math.log(max(1, n_obs)) - 2.0 * loglik
    return aic, bic


def _hmm_num_params(n_components: int, n_features: int, cov_type: str) -> int:
    """
    Approximate number of free parameters in Gaussian HMM.
    Components:
      - Means: n_components * n_features
      - Covariances:
         * full: n_components * n_features * (n_features + 1) / 2
         * diag: n_components * n_features
         * tied: n_features * (n_features + 1) / 2
         * spherical: n_components (one variance per comp)
      - Transition matrix: n_components*(n_components-1)
      - Initial state probs: n_components - 1
    This is an approximation but reasonable for IC comparison.
    """
    means = n_components * n_features
    if cov_type == 'full':
        covs = n_components * n_features * (n_features + 1) / 2.0
    elif cov_type == 'diag':
        covs = n_components * n_features
    elif cov_type == 'tied':
        covs = n_features * (n_features + 1) / 2.0
    elif cov_type == 'spherical':
        covs = n_components
    else:
        covs = n_components * n_features

    trans = n_components * (n_components - 1)
    init = n_components - 1
    k = int(round(means + covs + trans + init))
    return k


def _changepoint_bic(returns: np.ndarray, breakpoints: List[int]) -> float:
    """
    Compute a BIC-like score for piecewise-constant mean model.
    For each segment compute RSS; then BIC = n*ln(RSS/n) + k*ln(n)
    where k = number of parameters (one mean per segment).
    """
    n = len(returns)
    seg_idxs = [0] + list(breakpoints) + [n]
    rss = 0.0
    for i in range(len(seg_idxs) - 1):
        s, e = seg_idxs[i], seg_idxs[i + 1]
        seg = returns[s:e]
        if len(seg) == 0:
            continue
        mu = np.mean(seg)
        rss += np.sum((seg - mu) ** 2)
    k = len(seg_idxs) - 1  # number of means / segments
    bic = n * np.log(max(1e-12, rss / max(1, n))) + k * np.log(max(1, n))
    return bic


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
                'count': int(mask.sum()),
                'percentage': float(mask.sum() / len(regimes) * 100) if len(regimes) > 0 else 0.0,
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
            try:
                from_idx = unique_regimes.index(from_regime)
                to_idx = unique_regimes.index(to_regime)
                transitions[from_idx, to_idx] += 1
            except ValueError:
                # skip unexpected label
                continue

        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transitions / row_sums

        return pd.DataFrame(
            transition_probs,
            index=[f"From_{r}" for r in unique_regimes],
            columns=[f"To_{r}" for r in unique_regimes]
        )

    def expand_to_full_index(self, regimes: pd.Series, full_index: pd.DatetimeIndex, method: str = 'ffill') -> pd.Series:
        """
        Expand a (possibly shorter) regimes series to a full index (market_data.index).
        Common when features use dropna() and are shorter than price series.

        Args:
            regimes: pd.Series indexed by subset of dates (e.g., returns.dropna().index)
            full_index: Desired index (e.g., market_data.index)
            method: 'ffill', 'bfill', or 'nearest' - how to fill missing values

        Returns:
            pd.Series with index=full_index containing regime labels (int)
        """
        if regimes is None or regimes.empty:
            return pd.Series(index=full_index, dtype=float)

        # Reindex then fill
        expanded = regimes.reindex(full_index)
        if method == 'ffill':
            expanded = expanded.ffill().bfill()
        elif method == 'bfill':
            expanded = expanded.ffill().bfill()
        elif method == 'nearest':
            # nearest by index: forward then backward
            expanded = expanded.ffill().bfill()
        else:
            expanded = expanded.ffill().bfill()

        # If still NaN (e.g., entire series empty), fill with 0
        expanded = expanded.fillna(0).astype(int)
        return expanded

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
# HIDDEN MARKOV MODEL REGIME DETECTOR (ROBUSTIFIED)
# ============================================================================

class HMMRegimeDetector(RegimeDetector):
    """
    Hidden Markov Model for regime detection.
    Robustified: feature scaling, multiple restarts, alternative covariance types,
    and fallback to volatility detector if HMM fails.
    """

    def __init__(self, num_regimes: int = 2, covariance_type: str = 'full',
                 n_iter: int = 100, n_init: int = 5):
        """
        Args:
            num_regimes: Number of hidden states
            covariance_type: 'spherical', 'diag', 'full', 'tied'
            n_iter: Maximum iterations for EM algorithm
            n_init: number of random restarts to try (pick best log-likelihood)
        """
        super().__init__()

        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn not installed. Install with: pip install hmmlearn")

        self.num_regimes = num_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.n_init = max(1, int(n_init))
        self.name = f"HMM_{num_regimes}states"

        self.model = None
        self.regime_descriptions = {i: f"HMM State {i}" for i in range(num_regimes)}

    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect regimes using HMM (robustified: scaling, restarts, fallback)."""
        # Calculate returns and absolute returns as features
        returns = data['close'].pct_change().dropna()

        features = pd.DataFrame({
            'returns': returns,
            'abs_returns': returns.abs(),
        }).dropna()

        if len(features) < 10:
            raise ValueError("Not enough data for HMM (need at least 10 samples)")

        X = features.values

        # If sklearn StandardScaler not available, warn and proceed without scaling
        if StandardScaler is None:
            logger.warning("sklearn StandardScaler not found; HMM may be unstable without feature scaling.")
            X_scaled = X
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

        best_model = None
        best_score = -np.inf
        best_labels = None

        # Covariance types to try: prefer configured one, then fallback options
        cov_types_to_try = [self.covariance_type] if self.covariance_type else ['full']
        # Add safer options if full fails
        for ct in ['tied', 'diag']:
            if ct not in cov_types_to_try:
                cov_types_to_try.append(ct)

        attempt_info = []
        for cov_type in cov_types_to_try:
            for seed_offset in range(self.n_init):
                seed = 42 + seed_offset
                try:
                    model = hmm.GaussianHMM(
                        n_components=self.num_regimes,
                        covariance_type=cov_type,
                        n_iter=max(self.n_iter, 100),
                        tol=1e-4,
                        random_state=seed,
                        verbose=False
                    )

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(X_scaled)

                        # capture convergence warnings if any
                        conv_warnings = [ww for ww in w if issubclass(ww.category, ConvergenceWarning)]
                        if conv_warnings:
                            logger.debug(f"HMM convergence warnings (cov={cov_type}, seed={seed}): {conv_warnings}")

                    score = model.score(X_scaled)
                    labels = model.predict(X_scaled)

                    # sanity: ensure we have all requested states present
                    unique_labels = np.unique(labels)
                    if len(unique_labels) < self.num_regimes:
                        logger.debug(f"HMM produced fewer states ({len(unique_labels)}) than requested ({self.num_regimes}) with cov={cov_type}, seed={seed}")
                        attempt_info.append((cov_type, seed, score, False))
                        continue

                    attempt_info.append((cov_type, seed, score, True))

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_labels = labels

                except Exception as ex:
                    logger.debug(f"HMM fit failed for cov={cov_type}, seed={seed}: {ex}")
                    attempt_info.append((cov_type, seed, None, False))
                    continue

            # stop early if we found a good model
            if best_model is not None:
                break

        if best_model is None or best_labels is None:
            logger.warning("HMM failed to produce a stable model — falling back to VolatilityRegimeDetector")
            # fallback
            vol_detector = VolatilityRegimeDetector(window=20, num_regimes=self.num_regimes, method='quantile')
            vol_regimes = vol_detector.detect_regimes(data)
            self.regime_labels = vol_regimes
            self.regime_descriptions = vol_detector.regime_descriptions
            return vol_regimes

        # Map labels to sorted order by volatility so labels are interpretable (0=low vol)
        label_vol = {}
        for lbl in np.unique(best_labels):
            mask = best_labels == lbl
            if mask.sum() == 0:
                label_vol[lbl] = np.inf
            else:
                label_vol[lbl] = features.loc[mask, 'abs_returns'].mean()

        sorted_labels = sorted(label_vol.items(), key=lambda x: x[1])
        label_map = {orig: new for new, (orig, _) in enumerate(sorted_labels)}
        remapped = np.array([label_map[l] for l in best_labels])

        regimes_series = pd.Series(remapped, index=features.index)

        # Characterize regimes based on features
        self._characterize_regimes(features, regimes_series)

        # Persist model and regimes
        self.model = best_model
        self.regime_labels = regimes_series

        logger.info(f"HMM fitted: best_score={best_score:.3f}, cov_type={best_model.covariance_type}")

        return regimes_series

    def select_best_hmm_by_ic(self, X: np.ndarray, states_candidates: List[int] = None,
                              cov_types: List[str] = None, ic: str = 'bic', n_init: int = 3,
                              n_iter: Optional[int] = None) -> Dict[str, Any]:
        """
        Grid search HMM over number of states and covariance types using AIC/BIC.
        Returns: dict with best_model, labels, scores (DataFrame)
        """
        if not HMM_AVAILABLE:
            raise RuntimeError("hmmlearn not available")

        if states_candidates is None:
            states_candidates = [self.num_regimes]
        if cov_types is None:
            cov_types = [self.covariance_type] if self.covariance_type else ['full', 'tied', 'diag']

        n_iter = n_iter or self.n_iter
        results = []
        best_ic_val = np.inf
        best = None

        for n_states in states_candidates:
            for cov in cov_types:
                best_local_model = None
                best_ll = -np.inf
                for seed_off in range(max(1, n_init)):
                    seed = 42 + seed_off
                    try:
                        model = hmm.GaussianHMM(n_components=n_states, covariance_type=cov,
                                                n_iter=max(50, n_iter), random_state=seed, verbose=False)
                        model.fit(X)
                        ll = model.score(X)
                        if ll > best_ll:
                            best_ll = ll
                            best_local_model = model
                    except Exception as ex:
                        logger.debug(f"HMM grid fit failed for states={n_states}, cov={cov}, seed={seed}: {ex}")
                        continue

                if best_local_model is None:
                    continue

                # compute IC
                n_obs = X.shape[0]
                n_features = X.shape[1]
                k = _hmm_num_params(n_states, n_features, cov)
                aic, bic = _compute_aic_bic(best_ll, k, n_obs)
                chosen_ic_val = bic if ic == 'bic' else aic
                results.append({
                    'n_states': n_states, 'cov': cov, 'loglik': float(best_ll),
                    'aic': float(aic), 'bic': float(bic), 'k': int(k)
                })
                # select
                if chosen_ic_val < best_ic_val:
                    best_ic_val = chosen_ic_val
                    best = (best_local_model, best_ll, n_states, cov, aic, bic, k)

        df = pd.DataFrame(results).sort_values(by=ic) if results else pd.DataFrame(results)
        if best is None:
            return {'best_model': None, 'scores': df}

        best_model = best[0]
        labels = best_model.predict(X)
        return {'best_model': best_model, 'labels': labels, 'scores': df, 'chosen_ic': ic}

    def _characterize_regimes(self, features: pd.DataFrame, regimes: pd.Series):
        """Assign meaningful names to HMM states."""
        regime_chars = []

        for regime in range(self.num_regimes):
            mask = regimes == regime
            if mask.sum() == 0:
                mean_return = 0.0
                mean_vol = np.nan
            else:
                mean_return = features.loc[mask, 'returns'].mean()
                mean_vol = features.loc[mask, 'abs_returns'].mean()

            regime_chars.append({
                'regime': regime,
                'mean_return': mean_return,
                'volatility': mean_vol
            })

        # Sort by volatility and assign names
        regime_chars.sort(key=lambda x: (np.inf if np.isnan(x['volatility']) else x['volatility']))

        if self.num_regimes == 2:
            names = ["Low Vol", "High Vol"]
        elif self.num_regimes == 3:
            names = ["Low Vol", "Medium Vol", "High Vol"]
        else:
            names = [f"State {i}" for i in range(self.num_regimes)]

        for i, char in enumerate(regime_chars):
            orig_regime = char['regime']
            # Note: mapping orig_regime->name (we keep orig_regime indices but name by sorted order)
            self.regime_descriptions[orig_regime] = names[i] if i < len(names) else f"State {i}"


# ============================================================================
# CHANGE POINT REGIME DETECTOR
# ============================================================================

class ChangePointRegimeDetector(RegimeDetector):
    """
    Detects structural breaks in time series.
    Each segment between breakpoints is a regime.
    """

    def __init__(self, method: str = 'pelt', penalty: Union[float, List[float]] = 10, min_size: int = 20):
        """
        Args:
            method: 'pelt', 'binseg', or 'window' (ruptures library)
            penalty: Penalty for adding breakpoints (higher = fewer breaks). Can be scalar or list to grid-search.
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

        # If penalty is a list, do BIC-like selection over grid
        if isinstance(self.penalty, (list, tuple, np.ndarray)):
            best_bic = np.inf
            best_bps = None
            for pen in self.penalty:
                try:
                    algo.fit(signal)
                    bps = algo.predict(pen=pen)
                    # remove terminal breakpoint
                    if bps and bps[-1] == len(signal):
                        bps = bps[:-1]
                    bic = _changepoint_bic(signal.flatten(), bps)
                    if bic < best_bic:
                        best_bic = bic
                        best_bps = bps
                except Exception as ex:
                    logger.debug(f"Changepoint fit failed for penalty={pen}: {ex}")
                    continue
            if best_bps is None:
                # fallback to single-penalty behavior if grid fails
                algo.fit(signal)
                self.breakpoints = algo.predict(pen=float(self.penalty[0]))
                if self.breakpoints and self.breakpoints[-1] == len(signal):
                    self.breakpoints = self.breakpoints[:-1]
            else:
                self.breakpoints = best_bps
        else:
            # Fit and predict with scalar penalty
            algo.fit(signal)
            self.breakpoints = algo.predict(pen=float(self.penalty))

        # Remove the last breakpoint (it's always the end of the series)
        if self.breakpoints and self.breakpoints[-1] == len(signal):
            self.breakpoints = self.breakpoints[:-1]

        # Create regime labels
        regimes = np.zeros(len(returns), dtype=int)
        for i, bp in enumerate(self.breakpoints):
            regimes[bp:] = i + 1

        regimes_series = pd.Series(regimes, index=returns.index)

        # Describe regimes
        for i in range(int(regimes_series.max()) + 1):
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

            if sorted_ts[k - 1] <= 0:
                return np.nan

            # Hill estimator
            alpha = k / np.sum(np.log(sorted_ts[:k] / sorted_ts[k - 1]))
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

    def detect_regimes_auto(
        self,
        data: pd.DataFrame,
        methods: Optional[List[str]] = None,
        hmm_states: Optional[List[int]] = None,
        hmm_cov_types: Optional[List[str]] = None,
        model_selection_ic: Optional[str] = 'bic',
        changepoint_penalty_grid: Optional[List[float]] = None,
        expand_to_index: Optional[pd.DatetimeIndex] = None,
        ensemble_method: str = 'voting'
    ) -> Dict[str, Any]:
        """
        Run multiple detectors, optionally perform model selection (AIC/BIC),
        and return per-method series plus ensemble and confidence.

        Returns dict:
          {
            'per_method': {name: pd.Series, ...},
            'ic_scores': {detector_name: DataFrame or dict,..},
            'ensemble': pd.Series,
            'confidence': pd.Series
          }
        """
        if methods is None:
            methods = list(self.detectors.keys())

        per_method: Dict[str, pd.Series] = {}
        ic_scores: Dict[str, Any] = {}

        for name in methods:
            det = self.get(name)
            if det is None:
                logger.warning(f"Detector not found in registry: {name}")
                continue

            # special-case HMM: run selection if requested
            if isinstance(det, HMMRegimeDetector) and model_selection_ic is not None:
                # build features same as HMM.detect_regimes
                returns = data['close'].pct_change().dropna()
                features = pd.DataFrame({'returns': returns, 'abs_returns': returns.abs()}).dropna()
                if len(features) < 10:
                    logger.warning(f"Not enough data for HMM selection on detector {name}")
                    try:
                        series = det.detect_regimes(data)
                    except Exception as ex:
                        logger.warning(f"HMM detect fallback failed: {ex}")
                        series = pd.Series(dtype=int)
                else:
                    X = features.values
                    hs = hmm_states if hmm_states is not None else [det.num_regimes]
                    covs = hmm_cov_types if hmm_cov_types is not None else [det.covariance_type] if det.covariance_type else ['full', 'diag', 'tied']
                    try:
                        res = det.select_best_hmm_by_ic(X, states_candidates=hs, cov_types=covs, ic=model_selection_ic, n_init=det.n_init)
                        if res.get('best_model') is None:
                            logger.info(f"HMM selection returned no model (detector={name}); running default detect_regimes()")
                            series = det.detect_regimes(data)
                        else:
                            labels = res['labels']
                            series = pd.Series(labels, index=features.index)
                            det.regime_labels = series
                            ic_scores[name] = res.get('scores', None)
                    except Exception as ex:
                        logger.warning(f"HMM selection failed for detector {name}: {ex}")
                        try:
                            series = det.detect_regimes(data)
                        except Exception as ex2:
                            logger.warning(f"HMM detect_regimes fallback failed for {name}: {ex2}")
                            series = pd.Series(dtype=int)
            elif isinstance(det, ChangePointRegimeDetector) and changepoint_penalty_grid is not None:
                # If user provided a penalty grid, set det.penalty to it and let detector handle grid search.
                det.penalty = changepoint_penalty_grid
                try:
                    series = det.detect_regimes(data)
                except Exception as ex:
                    logger.warning(f"ChangePoint detect_regimes failed for {name}: {ex}")
                    series = pd.Series(dtype=int)
            else:
                # default call
                try:
                    series = det.detect_regimes(data)
                except Exception as ex:
                    logger.warning(f"Detector {name} failed: {ex}")
                    series = pd.Series(dtype=int)

            # expand to full index if requested
            if expand_to_index is not None and not series.empty:
                try:
                    series = det.expand_to_full_index(series, expand_to_index, method='ffill')
                except Exception:
                    series = det.expand_to_full_index(series, expand_to_index, method='ffill')

            per_method[name] = series

        # Ensemble: majority voting by default
        ensemble, confidence = self._ensemble_from_methods(per_method, method=ensemble_method, full_index=expand_to_index)
        return {'per_method': per_method, 'ic_scores': ic_scores, 'ensemble': ensemble, 'confidence': confidence}

    def _ensemble_from_methods(self, per_method_dict: Dict[str, pd.Series], method: str = 'voting',
                               full_index: Optional[pd.DatetimeIndex] = None) -> Tuple[pd.Series, pd.Series]:
        """
        Build ensemble labels and confidence:
          - voting: majority vote across methods at each timestamp
          - weighted: (not implemented) would weight detectors by confidence
        Returns (ensemble_series, confidence_series)
        """
        if not per_method_dict:
            if full_index is not None:
                empty_index = full_index
            else:
                empty_index = pd.DatetimeIndex([])
            return pd.Series(index=empty_index, dtype=int), pd.Series(index=empty_index, dtype=float)

        # Build full_index if not provided as union of indexes
        if full_index is None:
            all_idx = set()
            for s in per_method_dict.values():
                all_idx.update(s.index)
            full_index = pd.DatetimeIndex(sorted(all_idx))

        # Align and forward-fill each detector series
        aligned = {}
        for name, s in per_method_dict.items():
            if s is None or s.empty:
                # create filled zero series
                aligned[name] = pd.Series(0, index=full_index, dtype=int)
            else:
                try:
                    aligned[name] = s.reindex(full_index).ffill().bfill().astype(int)
                except Exception:
                    aligned[name] = s.reindex(full_index).fillna(0).astype(int)

        df = pd.DataFrame(aligned, index=full_index)

        if df.shape[1] == 0:
            return pd.Series(index=full_index, dtype=int), pd.Series(index=full_index, dtype=float)

        # majority voting
        # mode() may return multiple columns if tie; we pick the first mode if tie
        mode_df = df.mode(axis=1)
        ensemble = mode_df.iloc[:, 0] if mode_df.shape[1] > 0 else pd.Series(0, index=full_index)
        # confidence = fraction of detectors agreeing with ensemble
        agree_counts = (df.eq(ensemble, axis=0)).sum(axis=1)
        confidence = agree_counts / max(1, df.shape[1])
        return ensemble.astype(int), confidence.astype(float)

    def __repr__(self):
        return f"RegimeDetectorRegistry({len(self.detectors)} detectors)"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Regime Detector v1 - Example Usage")
    print("=" * 60)

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
    print("\n" + "-" * 60)
    print("1. Volatility Regime Detector")
    print("-" * 60)
    vol_detector = VolatilityRegimeDetector(window=20, num_regimes=2)
    vol_regimes = vol_detector.detect_regimes(data)
    print(f"Detected regimes: {vol_regimes.unique()}")
    print("\nRegime Statistics:")
    print(vol_detector.get_regime_stats(vol_regimes))

    # Test HMM Regime Detector (if available)
    if HMM_AVAILABLE:
        print("\n" + "-" * 60)
        print("2. HMM Regime Detector")
        print("-" * 60)
        hmm_detector = HMMRegimeDetector(num_regimes=2)
        hmm_regimes = hmm_detector.detect_regimes(data)
        print(f"Detected regimes: {hmm_regimes.unique()}")
        print("\nRegime Statistics:")
        print(hmm_detector.get_regime_stats(hmm_regimes))
    else:
        print("\nHMM not available (hmmlearn missing) - skipping HMM test.")

    # Test Change Point Detector (if available)
    if RUPTURES_AVAILABLE:
        print("\n" + "-" * 60)
        print("3. Change Point Detector")
        print("-" * 60)
        cp_detector = ChangePointRegimeDetector(penalty=5)
        cp_regimes = cp_detector.detect_regimes(data)
        print(f"Detected regimes: {cp_regimes.unique()}")
        print(f"Breakpoint dates: {cp_detector.get_breakpoint_dates()}")
        print("\nRegime Statistics:")
        print(cp_detector.get_regime_stats(cp_regimes))

    # Test Multifractal Detector
    print("\n" + "-" * 60)
    print("4. Multifractal Detector")
    print("-" * 60)
    mf_detector = MultifractalRegimeDetector(window=60, num_regimes=3)
    mf_regimes = mf_detector.detect_regimes(data)
    print(f"Detected regimes: {mf_regimes.unique()}")
    print("\nRegime Statistics:")
    print(mf_detector.get_regime_stats(mf_regimes))

    # Show auto-detection registry usage
    print("\n" + "-" * 60)
    print("5. Registry Auto-detection Example (if optional libs available)")
    print("-" * 60)
    registry = RegimeDetectorRegistry()
    registry.register(vol_detector)
    registry.register(mf_detector)
    if HMM_AVAILABLE:
        registry.register(hmm_detector)
    if RUPTURES_AVAILABLE:
        registry.register(cp_detector)

    try:
        res = registry.detect_regimes_auto(
            data,
            methods=registry.list_all(),
            hmm_states=[2, 3],
            hmm_cov_types=['full', 'diag'],
            model_selection_ic='bic',
            changepoint_penalty_grid=list(np.logspace(0, 2, 8)) if RUPTURES_AVAILABLE else None,
            expand_to_index=data.index,
            ensemble_method='voting'
        )
        print("Auto detection produced per_method keys:", list(res['per_method'].keys()))
        print("Ensemble head:")
        print(res['ensemble'].head())
        print("Confidence head:")
        print(res['confidence'].head())
    except Exception as ex:
        print("Auto-detection example failed:", ex)

    print("\n" + "=" * 60)
    print("Regime Detection Complete!")
    print("=" * 60)
