from .base import AdaptiveFilter, windowize
from .lms import (
    LMS, NLMS, VSSLMS, AboulnasrMayyasVSS, MathewsXieVSS,
    LMP, HeuristicMuScheduler, KalmanMuScheduler,
)
from .rls import RLS

__all__ = [
    "AdaptiveFilter", "windowize",
    "LMS", "NLMS", "VSSLMS", "AboulnasrMayyasVSS", "MathewsXieVSS",
    "LMP", "HeuristicMuScheduler", "KalmanMuScheduler",
    "RLS", "make_filter",
]


def make_filter(name: str, order: int = 16, **kwargs):
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name == "lms":
        return LMS(order=order, **kwargs)
    if name == "nlms":
        return NLMS(order=order, **kwargs)
    if name in ("vss_lms", "vsslms", "vss", "kwong_vss"):
        return VSSLMS(order=order, **kwargs)
    if name in ("aboulnasr_vss", "aboulnasr_mayyas"):
        return AboulnasrMayyasVSS(order=order, **kwargs)
    if name in ("mathews_xie", "mathews_vss"):
        return MathewsXieVSS(order=order, **kwargs)
    if name in ("lmp", "lmp_filter"):
        return LMP(order=order, **kwargs)
    if name in ("heuristic", "heuristic_scheduler"):
        return HeuristicMuScheduler(order=order, **kwargs)
    if name in ("kalman_scheduler", "kalman_mu"):
        return KalmanMuScheduler(order=order, **kwargs)
    if name == "rls":
        return RLS(order=order, **kwargs)
    raise ValueError(f"Unknown filter: {name}")
