"""Tests for the reference-based ANC + Kalman core (the sound redesign)."""
import numpy as np
import torch
import pytest

from src.filters.diff_kalman import (kalman_step, kalman_anc_numpy,
                                      decode_action_kalman, DiffKalmanConfig)
from src.filters.anc_baselines import NLMSANC, RLSANC, KalmanANC
from src.interference.families import make_interference, TRAIN_FAMILIES, OOD_FAMILIES, ALL_FAMILIES
from src.signals.generators import make_signal
from src.envs.anc_kalman_env import ANCKalmanEnv, ANCEnvConfig


FS, N, M = 360.0, 1500, 16


def _episode(fam, seed=0, snr=5.0):
    rng = np.random.default_rng(seed)
    clean = make_signal("ecg_like", n=N, fs=FS, rng=rng)
    interf, ref = make_interference(fam, clean, rng, snr_db=snr, fs=FS)
    return clean, interf, ref


def _snr(sig, resid):
    return 10 * np.log10(np.mean(sig ** 2) / (np.mean(resid ** 2) + 1e-12))


def test_torch_numpy_parity():
    clean, interf, ref = _episode("powerline")
    primary = clean + interf
    q, r = 3e-5, float(np.var(clean))
    e_np = kalman_anc_numpy(ref, primary, order=M, q=q, r=r)
    w = torch.zeros(1, M, dtype=torch.float64)
    P = torch.eye(M, dtype=torch.float64).unsqueeze(0)
    xb = torch.zeros(1, M, dtype=torch.float64)
    e_t = []
    for n in range(N):
        xb = torch.roll(xb, 1, dims=1); xb = xb.clone(); xb[:, 0] = ref[n]
        w, P, e = kalman_step(w, P, xb, torch.tensor([primary[n]], dtype=torch.float64),
                              torch.tensor([q], dtype=torch.float64),
                              torch.tensor([r], dtype=torch.float64))
        e_t.append(float(e))
    assert np.max(np.abs(np.array(e_t) - e_np)) < 1e-9


def test_gradient_flows_to_action():
    clean, interf, ref = _episode("echo")
    primary = clean + interf
    a = torch.zeros(1, 2, requires_grad=True, dtype=torch.float64)
    q, r = decode_action_kalman(a, DiffKalmanConfig())
    w = torch.zeros(1, M, dtype=torch.float64)
    P = torch.eye(M, dtype=torch.float64).unsqueeze(0)
    xb = torch.zeros(1, M, dtype=torch.float64)
    loss = 0.0
    for n in range(200):
        xb = torch.roll(xb, 1, dims=1); xb = xb.clone(); xb[:, 0] = float(ref[n])
        w, P, e = kalman_step(w, P, xb, torch.tensor([primary[n]], dtype=torch.float64), q, r)
        loss = loss + (e - clean[n]) ** 2
    loss.backward()
    assert a.grad is not None and a.grad.abs().sum() > 0


def test_reference_independent_of_clean():
    for fam in ALL_FAMILIES:
        clean, interf, ref = _episode(fam)
        # reference must be (near) uncorrelated with the clean signal
        assert abs(np.corrcoef(ref, clean)[0, 1]) < 0.3, fam


@pytest.mark.parametrize("fam", ["powerline", "narrowband_chirp"])
def test_powerline_cancellable(fam):
    clean, interf, ref = _episode(fam)
    primary = clean + interf
    e = KalmanANC(order=M, q=1e-6, r=1.0).run(ref, primary)
    gain = _snr(clean, e - clean) - _snr(clean, interf)
    assert gain > 5.0, f"{fam} gain {gain:.1f} dB"


def test_kalman_bounded_no_windup():
    """Prop.1: Kalman stays bounded even at max Q; low-forgetting RLS need not."""
    clean, interf, ref = _episode("regime_switch")
    primary = clean + interf
    e_k = kalman_anc_numpy(ref, primary, order=M, q=1e-3, r=1e-3)  # aggressive Q
    assert np.isfinite(e_k).all()
    assert _snr(clean, e_k - clean) > -10.0  # not diverged into nonsense


def test_env_features_finite_and_de_nonzero():
    cfg = ANCEnvConfig(episode_len=200)
    env = ANCKalmanEnv(cfg, fixed_family="powerline", fixed_signal="ecg_like",
                       fixed_snr_db=5.0, seed=0)
    obs, _ = env.reset(seed=0)
    a = np.array([0.2, 0.0], np.float32)
    des = []
    for _ in range(200):
        obs, rew, term, trunc, info = env.step(a)
        assert np.all(np.isfinite(obs)) and obs.shape == (11,)
        des.append(obs[2])  # de feature
        if term:
            break
    assert np.any(np.abs(des) > 1e-6)  # de is not identically zero (ordering bug guard)
    assert "episode_ss_mse" in info


def test_ood_families_held_out():
    assert set(OOD_FAMILIES).isdisjoint(set(TRAIN_FAMILIES))
    assert set(ALL_FAMILIES) == set(TRAIN_FAMILIES) | set(OOD_FAMILIES)
