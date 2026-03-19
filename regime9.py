
"""
Regime Detector v2 — Heavy-tail aware, robustified regime detection

Features:
 - Robust transforms (winsorize, rank-gauss)
 - Optional GARCH(1,1) conditional vol feature (via arch)
 - Tail-aware GMM (extra large-cov gaussian to soak tails)
 - Student-t HMM via pomegranate if available, otherwise robustified hmmlearn fallback
 - ChangePoint detector wrapper (ruptures) with BIC-like selection
 - Multifractal detector (Hurst + Hill tail index) preserved
 - collapse_states_to_3 helper
 - Registry with detect_regimes_auto preserving previous interface
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.mixture import GaussianMixture
import warnings
import logging
import math
import inspect

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Optional libs
# ----------------------------------------------------------------------
try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMMLEARN_AVAILABLE = True
except Exception:
    HMMLEARN_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except Exception:
    RUPTURES_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

# pomegranate: la API 1.x cambió; para evitar warnings/errores, la desactivamos si no está completa.
POMEGRANATE_AVAILABLE = False
try:
    import pomegranate as pg  # noqa
    try:
        from pomegranate.distributions import StudentTDistribution as _ST, ProductDistribution as _PD, MultivariateGaussianDistribution as _MVG  # noqa
        from pomegranate import HiddenMarkovModel as _HMM  # noqa
        POMEGRANATE_AVAILABLE = True
    except Exception:
        POMEGRANATE_AVAILABLE = False
except Exception:
    POMEGRANATE_AVAILABLE = False

# ======================================================================
# Utilities: transforms, garch vol, collapse helper
# ======================================================================

def robust_transform_df(df: pd.DataFrame, feature_cols: List[str],
                        method: str = 'winsorize', q: float = 0.01) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Robust transform features to mitigate fat-tail impact.
    - winsorize: clip columns to [q, 1-q] quantiles
    - rank_gauss: quantile->normal transform (QuantileTransformer output_distribution='normal')
    Returns transformed df copy and transformer (if applicable)
    """
    df2 = df.copy()
    if method == 'winsorize':
        for c in feature_cols:
            if c not in df2.columns:
                continue
            s = df2[c].dropna()
            if s.empty:
                continue
            lo = s.quantile(q)
            hi = s.quantile(1 - q)
            df2[c] = df2[c].clip(lo, hi)
        return df2, None

    elif method in ('rank_gauss', 'quantile_gauss'):
        transformer = QuantileTransformer(output_distribution='normal', copy=True, random_state=42)
        arr = transformer.fit_transform(df2[feature_cols].values)
        df2.loc[:, feature_cols] = arr
        return df2, transformer

    else:
        raise ValueError("Unknown transform method")


def add_garch_vol(df: pd.DataFrame, returns_col: str = 'log_ret', out_col: str = 'garch_sigma'):
    """
    Fit GARCH(1,1) with StudentT residuals to obtain conditional volatility as a feature.
    If arch not available, returns df unchanged.
    """
    if not ARCH_AVAILABLE:
        logger.info("arch not available; skipping GARCH vol feature")
        return df

    r = df[returns_col].dropna() * 100.0  # scale for arch
    if len(r) < 200:
        logger.info("Not enough returns for robust GARCH fit; skipping")
        return df

    try:
        am = arch_model(r, vol='Garch', p=1, q=1, dist='StudentsT')
        res = am.fit(disp='off', show_warning=False)
        cond_vol = res.conditional_volatility / 100.0
        # FIX: evitar FutureWarning
        cond_vol = cond_vol.reindex(df.index).ffill().bfill()
        df = df.copy()
        df[out_col] = cond_vol
        logger.info("Added garch_sigma feature (from arch)")
    except Exception as ex:
        logger.warning(f"GARCH fit failed: {ex}")
    return df


def collapse_states_to_3(state_series: pd.Series, price_series: pd.Series) -> pd.Series:
    """
    Collapse arbitrary states into 3 canonical regimes (0=bull,1=neutral,2=stress)
    based on per-state mean return and volatility, clustered via KMeans(3).
    """
    if state_series is None or state_series.empty:
        return pd.Series(index=price_series.index, data=0, dtype=int)

    states = sorted(state_series.dropna().unique())
    rows = []
    for s in states:
        mask = state_series == s
        idx = state_series.index[mask]
        if len(idx) == 0:
            rows.append({'orig_state': s, 'mean_ret': 0.0, 'vol': 0.0})
            continue
        rets = price_series.pct_change().reindex(idx).dropna()
        mean_ret = float(rets.mean()) if len(rets) > 0 else 0.0
        vol = float(rets.std()) if len(rets) > 1 else 0.0
        rows.append({'orig_state': s, 'mean_ret': mean_ret, 'vol': vol})

    df_states = pd.DataFrame(rows).set_index('orig_state')
    X = df_states[['mean_ret', 'vol']].fillna(0).values
    if X.shape[0] < 3:
        # simple mapping by volatility
        ordered = sorted(df_states.index, key=lambda x: df_states.loc[x, 'vol'])
        mapping = {orig: i for i, orig in enumerate(ordered)}
        return state_series.map(mapping).astype(int)

    km = KMeans(n_clusters=3, random_state=42, n_init=50).fit(X)
    labels = km.labels_
    centers = km.cluster_centers_

    # score cluster so that higher score => bull (high mean, low vol)
    cluster_scores = [c[0] - 10.0 * c[1] for c in centers]
    # order clusters by descending score → canonical mapping: 0=bull,1=neutral,2=stress
    order = np.argsort(cluster_scores)[::-1]
    cluster2canon = {cluster_idx: canon for canon, cluster_idx in enumerate(order)}
    state2canon = {orig: cluster2canon[lab] for orig, lab in zip(df_states.index.tolist(), labels)}
    collapsed = state_series.map(state2canon).astype(int)
    # expand to index if needed (ffill)
    collapsed = collapsed.reindex(price_series.index).ffill().bfill().astype(int)
    return collapsed


# ======================================================================
# Basic helpers: AIC/BIC and HMM param counting (kept from v1)
# ======================================================================

def _compute_aic_bic(loglik: float, num_params: int, n_obs: int) -> Tuple[float, float]:
    aic = 2 * num_params - 2.0 * loglik
    bic = num_params * math.log(max(1, n_obs)) - 2.0 * loglik
    return aic, bic

def _hmm_num_params(n_components: int, n_features: int, cov_type: str) -> int:
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

# ======================================================================
# Base class
# ======================================================================

class RegimeDetector(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.regime_labels: Optional[pd.Series] = None
        self.regime_descriptions: Dict[int, str] = {}

    @abstractmethod
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        pass

    def get_regime_stats(self, regimes: pd.Series) -> pd.DataFrame:
        unique_regimes = sorted(regimes.unique())
        stats = []
        for regime in unique_regimes:
            mask = regimes == regime
            stats.append({
                'regime': int(regime),
                'count': int(mask.sum()),
                'percentage': float(mask.sum() / len(regimes) * 100) if len(regimes) > 0 else 0.0,
                'description': self.regime_descriptions.get(int(regime), f"Regime {regime}")
            })
        return pd.DataFrame(stats)

    def expand_to_full_index(self, regimes: pd.Series, full_index: pd.DatetimeIndex, method: str = 'ffill') -> pd.Series:
        if regimes is None or regimes.empty:
            return pd.Series(index=full_index, dtype=int)
        expanded = regimes.reindex(full_index)
        expanded = expanded.ffill().bfill()
        expanded = expanded.fillna(0).astype(int)
        return expanded

    def __repr__(self):
        return f"{self.name}()"

# ======================================================================
# Volatility detector
# ======================================================================

class VolatilityRegimeDetector(RegimeDetector):
    def __init__(self, window: int = 20, num_regimes: int = 2, method: str = 'quantile'):
        super().__init__()
        self.window = window
        self.num_regimes = num_regimes
        self.method = method
        self.name = f"VolRegime_{window}_{num_regimes}states"
        if num_regimes == 2:
            self.regime_descriptions = {0: "Low Vol", 1: "High Vol"}
        elif num_regimes == 3:
            self.regime_descriptions = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}
        else:
            self.regime_descriptions = {i: f"Vol Regime {i}" for i in range(num_regimes)}

    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change()
        vol = returns.rolling(window=self.window, min_periods=max(3, int(self.window/2))).std() * np.sqrt(252)
        vol = vol.dropna()
        if len(vol) == 0:
            raise ValueError("Not enough data for volatility detector")
        if self.method == 'quantile':
            ql = vol.quantile(0.01)
            qh = vol.quantile(0.99)
            vol_clip = vol.clip(ql, qh)
            return self._quantile_method(vol_clip)
        else:
            return self._kmeans_method(vol)

    def _quantile_method(self, vol: pd.Series) -> pd.Series:
        regimes = pd.Series(0, index=vol.index, dtype=int)
        if self.num_regimes == 2:
            threshold = vol.median()
            regimes[vol >= threshold] = 1
        else:
            quantiles = np.linspace(0, 1, self.num_regimes + 1)[1:-1]
            thresholds = vol.quantile(quantiles)
            for i, threshold in enumerate(thresholds):
                regimes[vol >= threshold] = i + 1
        self.regime_labels = regimes
        return regimes

    def _kmeans_method(self, vol: pd.Series) -> pd.Series:
        X = vol.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_.flatten()
        sorted_clusters = np.argsort(centers)
        label_map = {old: new for new, old in enumerate(sorted_clusters)}
        labels = np.array([label_map[l] for l in labels])
        regimes = pd.Series(labels, index=vol.index)
        self.regime_labels = regimes
        return regimes

# ======================================================================
# Tail-aware GMM
# ======================================================================

class TailAwareGMM(RegimeDetector):
    """
    Fit GaussianMixture with n_components (normal) + 1 'tail' component that is initialized with large covariance.
    The extra component helps approximate heavy tails by soaking outliers into a wide Gaussian.
    """
    def __init__(self, n_components: int = 3, random_state: int = 42):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.name = f"TailAwareGMM_{n_components}normals_plus_tail"

    def detect_regimes(self, data: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.Series:
        if feature_cols is None:
            returns = data['close'].pct_change().dropna()
            X = returns.values.reshape(-1, 1)
            idx = returns.index
        else:
            X = data[feature_cols].dropna().values
            idx = data[feature_cols].dropna().index

        if X.shape[0] < 10:
            raise ValueError("Not enough data for TailAwareGMM")

        n = max(1, self.n_components)
        gmm = GaussianMixture(n_components=n + 1, random_state=self.random_state, n_init=10, covariance_type='full')
        gmm.fit(X)

        covs = gmm.covariances_
        if covs.ndim == 3:
            variances = np.array([np.linalg.det(c) if c.shape[0] > 1 else float(c) for c in covs])
        else:
            variances = np.array([float(cov) for cov in covs])

        tail_idx = int(np.argmax(variances))
        labels = gmm.predict(X)

        means = np.array([gmm.means_[i].mean() for i in range(gmm.means_.shape[0])])
        non_tail_idxs = [i for i in range(len(means)) if i != tail_idx]
        sorted_non_tail = sorted(non_tail_idxs, key=lambda i: means[i])
        mapping = {orig: new for new, orig in enumerate(sorted_non_tail)}
        mapping[tail_idx] = n  # tail al final
        mapped = np.array([mapping[label] for label in labels])
        regimes = pd.Series(mapped, index=idx).astype(int)
        self.regime_labels = regimes
        for i in range(n + 1):
            self.regime_descriptions[i] = f"GMM Regime {i}"
        return regimes

# ======================================================================
# Student-t HMM via pomegranate (si disponible) o fallback hmmlearn
# ======================================================================

class StudentTHMMRegimeDetector(RegimeDetector):
    """
    Student-t emission HMM using pomegranate when available.
    Fallback: robustified Gaussian HMM (hmmlearn) on rank-gauss transformed features.
    """
    def __init__(self, num_regimes: int = 2, n_init: int = 5, covariance_type: str = 'diag'):
        super().__init__()
        self.num_regimes = num_regimes
        self.n_init = max(1, int(n_init))
        self.covariance_type = covariance_type
        self.name = f"tHMM_{num_regimes}states"
        self.model = None
        self.regime_descriptions = {i: f"State {i}" for i in range(num_regimes)}

    def detect_regimes(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> pd.Series:
        returns = data['close'].pct_change().dropna()
        if features is None:
            feats = pd.DataFrame({'returns': returns, 'abs_returns': returns.abs()}).dropna()
        else:
            feats = data[features].dropna()
            if 'returns' not in feats.columns:
                feats['returns'] = returns.reindex(feats.index)
                feats['abs_returns'] = feats['returns'].abs()
            feats = feats.dropna()
        if len(feats) < max(10, 3 * self.num_regimes):
            raise ValueError("Not enough data for HMM")

        X = feats.values
        idx = feats.index

        # Usar pomegranate solo si está disponible y estable (evitamos errores de API)
        if POMEGRANATE_AVAILABLE:
            try:
                # Inicializar estados con KMeans y distribuciones StudentT independientes por dimensión
                km = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=10).fit(X)
                dists = []
                for s in range(self.num_regimes):
                    block = X[km.labels_ == s]
                    if len(block) < 2:
                        mu = X.mean(axis=0)
                        cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
                        d = _MVG(mu, cov)
                    else:
                        mu = block.mean(axis=0)
                        dims = []
                        for d_i in range(X.shape[1]):
                            col = block[:, d_i]
                            kurt = stats.kurtosis(col, fisher=False, bias=False) if len(col) > 3 else 3.0
                            df_est = max(3.0, min(30.0, 6.0 * (1.0 / (kurt - 3.0 + 1e-6) if kurt > 3.0 else 6.0)))
                            dims.append(_ST(df=df_est, loc=float(mu[d_i]), scale=float(np.std(col) + 1e-8)))
                        d = _PD(dims)
                    dists.append(d)
                model = _HMM.from_samples(dists[0].__class__, n_components=self.num_regimes, X=[X], n_jobs=1, max_iterations=100)
                _, path = model.viterbi(X)
                decoded = [s.name for (p, s) in path[1:]]
                numeric = []
                for nm in decoded:
                    try:
                        if nm and nm[0].upper() == "S":
                            numeric.append(int(nm[1:]))
                        else:
                            numeric.append(0)
                    except Exception:
                        numeric.append(0)
                regimes = pd.Series(numeric, index=idx).astype(int)
                self.model = model
                self.regime_labels = regimes
                self._characterize_regimes(feats, regimes)
                logger.info("StudentT HMM fitted via pomegranate")
                return regimes
            except Exception as ex:
                logger.warning(f"Pomegranate t-HMM failed, falling back to robust Gaussian HMM: {ex}")

        # Fallback limpio con hmmlearn (sin warnings por medias)
        if not HMMLEARN_AVAILABLE:
            km = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=20).fit(X)
            labels = km.labels_
            regimes = pd.Series(labels, index=idx).astype(int)
            self.regime_labels = regimes
            self._characterize_regimes(feats, regimes)
            logger.info("Fitted fallback KMeans (no HMM libs available)")
            return regimes

        # Transform rank-gauss y HMM gaussiano
        feats_df = pd.DataFrame(X, index=idx, columns=[f"f{i}" for i in range(X.shape[1])])
        feats_trans, _ = robust_transform_df(feats_df, feats_df.columns.tolist(), method='quantile_gauss', q=0.01)
        X_scaled = feats_trans.values
        best_score = -np.inf
        best_model = None
        best_labels = None
        for seed_off in range(self.n_init):
            seed = 42 + seed_off
            try:
                # NOTA: no seteamos means_ para evitar warnings; dejamos que el modelo inicialice.
                model = hmmlearn_hmm.GaussianHMM(
                    n_components=self.num_regimes,
                    covariance_type=self.covariance_type,
                    n_iter=200,
                    random_state=seed,
                    verbose=False
                )
                model.fit(X_scaled)
                score = model.score(X_scaled)
                labels = model.predict(X_scaled)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_labels = labels
            except Exception as ex:
                logger.debug(f"HMM fit failed seed {seed}: {ex}")
                continue
        if best_model is None:
            km = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=20).fit(X_scaled)
            labels = km.labels_
            regimes = pd.Series(labels, index=idx).astype(int)
            self.regime_labels = regimes
            self._characterize_regimes(feats, regimes)
            logger.info("Fallback KMeans used (HMM failed)")
            return regimes

        label_vol = {}
        for lbl in np.unique(best_labels):
            mask = best_labels == lbl
            label_vol[lbl] = feats['abs_returns'].iloc[mask].mean() if 'abs_returns' in feats.columns else np.mean(np.abs(X[mask]), axis=0).mean()

        sorted_labels = sorted(label_vol.items(), key=lambda x: x[1])
        label_map = {orig: new for new, (orig, _) in enumerate(sorted_labels)}
        remapped = np.array([label_map[l] for l in best_labels])
        regimes = pd.Series(remapped, index=idx).astype(int)
        self.model = best_model
        self.regime_labels = regimes
        self._characterize_regimes(feats, regimes)
        logger.info(f"robust Gaussian HMM fitted (score={best_score:.3f})")
        return regimes

    def select_best_hmm_by_ic(self, X: np.ndarray, states_candidates: List[int] = None, cov_types: List[str] = None, ic: str = 'bic', n_init: int = 3):
        if not HMMLEARN_AVAILABLE:
            raise RuntimeError("hmmlearn not available")
        states_candidates = states_candidates or [self.num_regimes]
        cov_types = cov_types or [self.covariance_type]
        results = []
        best_ic_val = np.inf
        best = None
        for n_states in states_candidates:
            for cov in cov_types:
                best_local = None
                best_ll = -np.inf
                for seed_off in range(max(1, n_init)):
                    seed = 42 + seed_off
                    try:
                        m = hmmlearn_hmm.GaussianHMM(n_components=n_states, covariance_type=cov, n_iter=200, random_state=seed)
                        m.fit(X)
                        ll = m.score(X)
                        if ll > best_ll:
                            best_ll = ll
                            best_local = m
                    except Exception:
                        continue
                if best_local is None:
                    continue
                n_obs = X.shape[0]; n_features = X.shape[1]
                k = _hmm_num_params(n_states, n_features, cov)
                aic, bic = _compute_aic_bic(best_ll, k, n_obs)
                chosen = bic if ic == 'bic' else aic
                results.append({'n_states': n_states, 'cov': cov, 'loglik': float(best_ll), 'aic': float(aic), 'bic': float(bic), 'k': int(k)})
                if chosen < best_ic_val:
                    best_ic_val = chosen
                    best = (best_local, best_ll, n_states, cov, aic, bic, k)
        df = pd.DataFrame(results).sort_values(by=ic) if results else pd.DataFrame(results)
        if best is None:
            return {'best_model': None, 'scores': df}
        best_model = best[0]
        labels = best_model.predict(X)
        return {'best_model': best_model, 'labels': labels, 'scores': df, 'chosen_ic': ic}

    def _characterize_regimes(self, features: pd.DataFrame, regimes: pd.Series):
        regime_chars = []
        for regime in range(self.num_regimes):
            mask = regimes == regime
            if mask.sum() == 0:
                mean_return = 0.0; mean_vol = np.nan
            else:
                if 'returns' in features.columns:
                    mean_return = features.loc[mask, 'returns'].mean()
                    mean_vol = features.loc[mask, 'abs_returns'].mean() if 'abs_returns' in features.columns else features.loc[mask].abs().mean().mean()
                else:
                    mean_return = features.loc[mask].mean().mean()
                    mean_vol = features.loc[mask].std().mean()
            regime_chars.append({'regime': regime, 'mean_return': mean_return, 'volatility': mean_vol})
        regime_chars.sort(key=lambda x: (np.inf if np.isnan(x['volatility']) else x['volatility']))
        if self.num_regimes == 2:
            names = ["Low Vol", "High Vol"]
        elif self.num_regimes == 3:
            names = ["Low Vol", "Medium Vol", "High Vol"]
        else:
            names = [f"State {i}" for i in range(self.num_regimes)]
        for i, char in enumerate(regime_chars):
            orig_regime = char['regime']
            self.regime_descriptions[orig_regime] = names[i] if i < len(names) else f"State {i}"

# ======================================================================
# ChangePoint (wrap ruptures)
# ======================================================================

class ChangePointRegimeDetector(RegimeDetector):
    def __init__(self, method: str = 'pelt', penalty: Union[float, List[float]] = 10, min_size: int = 20):
        super().__init__()
        if not RUPTURES_AVAILABLE:
            raise ImportError("ruptures not installed")
        self.method = method
        self.penalty = penalty
        self.min_size = min_size
        self.name = f"ChangePoint_{method}"
        self.breakpoints = None

    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change().dropna()
        signal = returns.values.reshape(-1, 1)
        if len(signal) < self.min_size * 2:
            raise ValueError("Not enough data for change point detection")
        if self.method == 'pelt':
            algo = rpt.Pelt(model='rbf', min_size=self.min_size)
        elif self.method == 'binseg':
            algo = rpt.Binseg(model='l2', min_size=self.min_size)
        else:
            algo = rpt.Window(width=self.min_size, model='l2')
        if isinstance(self.penalty, (list, tuple, np.ndarray)):
            best_bic = np.inf; best_bps = None
            for pen in self.penalty:
                try:
                    algo.fit(signal)
                    bps = algo.predict(pen=pen)
                    if bps and bps[-1] == len(signal):
                        bps = bps[:-1]
                    bic = _changepoint_bic(signal.flatten(), bps)
                    if bic < best_bic:
                        best_bic = bic; best_bps = bps
                except Exception:
                    continue
            self.breakpoints = best_bps or []
        else:
            algo.fit(signal)
            self.breakpoints = algo.predict(pen=float(self.penalty))
            if self.breakpoints and self.breakpoints[-1] == len(signal):
                self.breakpoints = self.breakpoints[:-1]
        regimes = np.zeros(len(returns), dtype=int)
        for i, bp in enumerate(self.breakpoints):
            regimes[bp:] = i + 1
        regimes_series = pd.Series(regimes, index=returns.index)
        for i in range(int(regimes_series.max()) + 1):
            self.regime_descriptions[i] = f"Regime {i}"
        self.regime_labels = regimes_series
        return regimes_series

    def get_breakpoint_dates(self) -> List[pd.Timestamp]:
        if self.regime_labels is None or self.breakpoints is None:
            return []
        dates = []
        for bp in self.breakpoints:
            if bp < len(self.regime_labels):
                dates.append(self.regime_labels.index[bp])
        return dates

def _changepoint_bic(series: np.ndarray, breakpoints: List[int]) -> float:
    """BIC-like penalization for number of segments."""
    if len(series) == 0:
        return np.inf
    bps = [0] + sorted([int(b) for b in breakpoints if 0 < b < len(series)]) + [len(series)]
    ll = 0.0
    k = 0
    for i in range(len(bps) - 1):
        seg = series[bps[i]:bps[i+1]]
        if len(seg) < 2:
            continue
        mu = np.mean(seg)
        var = np.var(seg) + 1e-8
        ll += -0.5 * len(seg) * (np.log(2*np.pi*var) + 1.0)
        k += 2  # mu & var por segmento
    n = len(series)
    bic = -2 * ll + k * np.log(max(1, n))
    return float(bic)

# ======================================================================
# Multifractal detector
# ======================================================================

class MultifractalRegimeDetector(RegimeDetector):
    def __init__(self, window: int = 60, num_regimes: int = 3):
        super().__init__()
        self.window = window
        self.num_regimes = num_regimes
        self.name = f"Multifractal_{window}_{num_regimes}states"
        self.regime_descriptions = {i: f"Fractal Regime {i}" for i in range(num_regimes)}

    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change().dropna()
        hurst = self._rolling_hurst(returns, self.window)
        tail_index = self._rolling_tail_index(returns, self.window)
        features = pd.DataFrame({'hurst': hurst, 'tail_index': tail_index}).dropna()
        if len(features) < 10:
            raise ValueError("Not enough data for multifractal detector")
        km = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=20).fit(features.values)
        labels = km.labels_
        regimes = pd.Series(labels, index=features.index)
        self._characterize_fractal_regimes(features, regimes)
        self.regime_labels = regimes
        return regimes

    def _rolling_hurst(self, returns: pd.Series, window: int) -> pd.Series:
        def hurst_exponent(ts):
            if len(ts) < 20: return np.nan
            lags = range(2, min(20, len(ts) // 2))
            tau = []
            for lag in lags:
                std_diff = np.std(np.diff(ts, n=lag))
                tau.append(std_diff)
            if len(tau) < 2: return np.nan
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        return returns.rolling(window=window).apply(hurst_exponent, raw=False)

    def _rolling_tail_index(self, returns: pd.Series, window: int) -> pd.Series:
        def hill_estimator(ts, k=None):
            if len(ts) < 10: return np.nan
            abs_ts = np.abs(ts)
            k = k or max(5, len(ts) // 4)
            sorted_ts = np.sort(abs_ts)[::-1]
            if sorted_ts[k - 1] <= 0: return np.nan
            alpha = k / np.sum(np.log(sorted_ts[:k] / sorted_ts[k - 1]))
            return alpha
        return returns.rolling(window=window).apply(hill_estimator, raw=False)

    def _characterize_fractal_regimes(self, features: pd.DataFrame, regimes: pd.Series):
        for regime in range(self.num_regimes):
            mask = regimes == regime
            mean_hurst = features.loc[mask, 'hurst'].mean()
            mean_tail = features.loc[mask, 'tail_index'].mean()
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

# ======================================================================
# Registry: manage detectors and ensemble
# ======================================================================

class RegimeDetectorRegistry:
    def __init__(self):
        self.detectors: Dict[str, RegimeDetector] = {}

    def register(self, detector: RegimeDetector):
        self.detectors[detector.name] = detector

    def get(self, name: str) -> Optional[RegimeDetector]:
        return self.detectors.get(name)

    def list_all(self) -> List[str]:
        return list(self.detectors.keys())

    def detect_regimes_auto(self,
                            data: pd.DataFrame,
                            methods: Optional[List[str]] = None,
                            hmm_states: Optional[List[int]] = None,
                            hmm_cov_types: Optional[List[str]] = None,
                            model_selection_ic: Optional[str] = 'bic',
                            changepoint_penalty_grid: Optional[List[float]] = None,
                            expand_to_index: Optional[pd.DatetimeIndex] = None,
                            ensemble_method: str = 'voting',
                            add_garch: bool = True,
                            garch_returns_col: str = 'log_ret'):
        if methods is None:
            methods = list(self.detectors.keys())
        per_method: Dict[str, pd.Series] = {}
        ic_scores: Dict[str, Any] = {}

        df = data.copy()
        # add common features
        if 'log_ret' not in df.columns:
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['ret'] = df['close'].pct_change()
            df['abs_ret'] = df['log_ret'].abs()

        if add_garch:
            df = add_garch_vol(df, returns_col=garch_returns_col)

        for name in methods:
            det = self.get(name)
            if det is None:
                logger.warning(f"Detector not found: {name}")
                continue

            try:
                sig = inspect.signature(det.detect_regimes)
                if 'feature_cols' in sig.parameters:
                    series = det.detect_regimes(df, feature_cols=[c for c in df.columns if c not in ('open','high','low','close','adj_close','volume')])
                else:
                    series = det.detect_regimes(df)
            except Exception as ex:
                logger.warning(f"{name} detection failed: {ex}")
                series = pd.Series(dtype=int)

            if expand_to_index is not None and not series.empty:
                try:
                    series = det.expand_to_full_index(series, expand_to_index, method='ffill')
                except Exception:
                    series = det.expand_to_full_index(series, expand_to_index, method='ffill')

            per_method[name] = series

        # FIX: NO usar "or" con DatetimeIndex
        full_index = expand_to_index if expand_to_index is not None else data.index
        ensemble, confidence = self._ensemble_from_methods(per_method, method=ensemble_method, full_index=full_index)
        return {'per_method': per_method, 'ic_scores': ic_scores, 'ensemble': ensemble, 'confidence': confidence}

    def _ensemble_from_methods(self, per_method_dict: Dict[str, pd.Series], method: str = 'voting',
                               full_index: Optional[pd.DatetimeIndex] = None) -> Tuple[pd.Series, pd.Series]:
        if not per_method_dict:
            empty_index = full_index if full_index is not None else pd.DatetimeIndex([])
            return pd.Series(index=empty_index, dtype=int), pd.Series(index=empty_index, dtype=float)
        if full_index is None:
            all_idx = set()
            for s in per_method_dict.values():
                all_idx.update(s.index)
            full_index = pd.DatetimeIndex(sorted(all_idx))
        aligned = {}
        for name, s in per_method_dict.items():
            if s is None or s.empty:
                aligned[name] = pd.Series(0, index=full_index, dtype=int)
            else:
                aligned[name] = s.reindex(full_index).ffill().bfill().astype(int)
        df = pd.DataFrame(aligned, index=full_index)
        if df.shape[1] == 0:
            return pd.Series(index=full_index, dtype=int), pd.Series(index=full_index, dtype=float)
        mode_df = df.mode(axis=1)
        ensemble = mode_df.iloc[:, 0] if mode_df.shape[1] > 0 else pd.Series(0, index=full_index)
        agree_counts = (df.eq(ensemble, axis=0)).sum(axis=1)
        confidence = agree_counts / max(1, df.shape[1])
        return ensemble.astype(int), confidence.astype(float)

    def __repr__(self):
        return f"RegimeDetectorRegistry({len(self.detectors)} detectors)"

# ======================================================================
# Quick example usage (only run when executed directly)
# ======================================================================

if __name__ == "__main__":
    import yfinance as yf
    print("Regime Detector v2 — Demo")
    df = yf.download("SPY", start="2010-01-01", end="2023-12-31", progress=False, auto_adjust=False)
    df = df.rename(columns={'Adj Close': 'adj_close'}).assign(close=lambda x: x['Adj Close'] if 'Adj Close' in x.columns else x['adj_close'])
    df.index = pd.to_datetime(df.index)

    reg = RegimeDetectorRegistry()
    reg.register(VolatilityRegimeDetector(window=21, num_regimes=3, method='quantile'))
    reg.register(TailAwareGMM(n_components=2))
    reg.register(MultifractalRegimeDetector(window=60, num_regimes=3))
    reg.register(StudentTHMMRegimeDetector(num_regimes=3, n_init=6))

    res = reg.detect_regimes_auto(df, methods=reg.list_all(), expand_to_index=df.index, add_garch=True)
    print("Per-method keys:", list(res['per_method'].keys()))
    print("Ensemble head:")
    print(res['ensemble'].head())
    print("Confidence head:")
    print(res['confidence'].head())

    # collapse to canonical 3
    collapsed = collapse_states_to_3(res['ensemble'], df['close'])
    print("Collapsed head:", collapsed.head())

    # Basic stats
    for name, series in res['per_method'].items():
        if series is None or series.empty:
            continue
        print(f"\nDetector: {name}")
        det = reg.get(name)
        print(det.get_regime_stats(series))
PY
