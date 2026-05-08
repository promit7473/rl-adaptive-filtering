"""Statistical helpers for rigorous reporting:

- bootstrap_ci: percentile bootstrap confidence interval for any statistic
- iqm: interquartile mean (Agarwal et al., NeurIPS 2021 — robust to outliers)
- stratified_bootstrap_ci: resample within strata (e.g., per seed), pool, recompute
- paired_test: Wilcoxon signed-rank for per-seed paired comparisons
- performance_profile: tau-thresholded fraction-of-runs curves
"""
from __future__ import annotations
import numpy as np
from typing import Sequence, Callable


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: average of values within [Q1, Q3]."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    lo, hi = np.percentile(x, [25, 75])
    mid = x[(x >= lo) & (x <= hi)]
    return float(np.mean(mid)) if mid.size else float(np.median(x))


def bootstrap_ci(x: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean,
                 n_boot: int = 10_000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None
                 ) -> tuple[float, float, float]:
    """Percentile bootstrap CI for `stat(x)`. Returns (point, lo, hi)."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = x.size
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boots[b] = stat(rng.choice(x, size=n, replace=True))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(stat(x)), lo, hi


def stratified_bootstrap_ci(per_stratum: Sequence[np.ndarray],
                             stat: Callable[[np.ndarray], float] = iqm,
                             n_boot: int = 10_000, alpha: float = 0.05,
                             rng: np.random.Generator | None = None
                             ) -> tuple[float, float, float]:
    """Resample WITHIN each stratum (seed/family), pool, compute `stat`."""
    rng = rng or np.random.default_rng(0)
    strata = [np.asarray(s, dtype=float) for s in per_stratum if len(s) > 0]
    if not strata:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pooled = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        boots[b] = stat(pooled)
    point = stat(np.concatenate(strata))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                   alpha: float = 0.05, rng: np.random.Generator | None = None
                   ) -> tuple[float, float, float]:
    """Bootstrap CI of mean(a - b) for paired samples (same indexing)."""
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    return bootstrap_ci(d, stat=np.mean, n_boot=n_boot, alpha=alpha, rng=rng)


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value (two-sided) for paired data."""
    try:
        from scipy.stats import wilcoxon
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if np.allclose(a, b):
            return 1.0
        return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def performance_profile(scores_per_method: dict, taus: np.ndarray,
                        lower_better: bool = True) -> dict:
    """For each method, fraction of runs with score <= tau (lower_better=True).
    Returns {method: array_over_taus}.
    """
    out = {}
    for m, s in scores_per_method.items():
        s = np.asarray(s, dtype=float)
        s = s[np.isfinite(s)]
        if lower_better:
            out[m] = np.array([(s <= t).mean() if s.size else 0.0 for t in taus])
        else:
            out[m] = np.array([(s >= t).mean() if s.size else 0.0 for t in taus])
    return out
