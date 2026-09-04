"""NLMS config and thin re-exports of kernel decode/schedule."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..kernel import (
    ActionBounds,
    MU_SCHEDULE_GAIN,
    MU_SCHEDULE_REF,
    decode_action_torch,
    mu_base_schedule,
)


@dataclass
class DiffNLMSConfig:
    order: int = 16
    mu_min: float = 0.005
    mu_max: float = 2.0
    lam_min: float = 0.80
    lam_max: float = 0.999
    eps: float = 1e-6
    max_w_norm: float = 100.0


def decode_action_bptt(a: torch.Tensor, cfg: DiffNLMSConfig):
    """Decode raw actions in [-1,1] to (mu, lambda) via log-scale interp.

    Args:
        a: (..., 2) tensor of raw actions in [-1, 1]
    Returns:
        mu: (...,) tensor
        lam: (...,) tensor
    """
    bounds = ActionBounds(
        mu_min=cfg.mu_min, mu_max=cfg.mu_max,
        lam_min=cfg.lam_min, lam_max=cfg.lam_max,
    )
    return decode_action_torch(a, bounds)


__all__ = [
    "DiffNLMSConfig",
    "decode_action_bptt",
    "mu_base_schedule",
    "MU_SCHEDULE_GAIN",
    "MU_SCHEDULE_REF",
    "ActionBounds",
]
