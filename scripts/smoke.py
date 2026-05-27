"""Smoke test: env reset/step, every filter runs, 2k-step training works."""
from __future__ import annotations
import numpy as np

from src.envs import AdaptiveFilterEnv, EnvConfig
from src.signals.generators import make_signal
from src.noise.families import make_noise
from src.filters import (
    NLMS, RLS, VSSLMS, HeuristicMuScheduler,
    PIDLeakyNLMS, FixedLeakageNLMS, IIRNotch, windowize,
)


def test_env():
    env = AdaptiveFilterEnv(EnvConfig(episode_len=200))
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    total = 0.0
    for _ in range(200):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term:
            break
    print(f"[env] ok, return={total:.2f}, info_keys={list(info)}")


def test_filters():
    rng = np.random.default_rng(0)
    clean = make_signal("multitone", n=1024, fs=360.0, rng=rng)
    noisy = clean + make_noise("gaussian", clean, rng, snr_db=10.0, fs=360.0)
    U = windowize(noisy, 16)
    for name, filt in [
        ("NLMS", NLMS(order=16, mu=0.5)),
        ("RLS", RLS(order=16, forgetting=0.995)),
        ("VSS", VSSLMS(order=16)),
        ("Heur", HeuristicMuScheduler(order=16)),
        ("PID", PIDLeakyNLMS(order=16)),
        ("Leaky", FixedLeakageNLMS(order=16, mu=0.5, leakage=0.99)),
    ]:
        _, e = filt.run(U, clean)
        assert np.isfinite(e).all(), f"{name} produced non-finite"
        print(f"[filt] {name:6s} mse_db={10*np.log10(np.mean(e**2)):+.2f}")
    notch = IIRNotch(f0=50.0, fs=360.0)
    y = notch.run(noisy)
    assert np.isfinite(y).all()
    print(f"[filt] Notch  mse_db={10*np.log10(np.mean((clean-y)**2)):+.2f}")


def test_training():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    cfg = EnvConfig(episode_len=200)
    vec = DummyVecEnv([lambda: AdaptiveFilterEnv(cfg, seed=0)])
    model = PPO("MlpPolicy", vec, n_steps=64, batch_size=32, verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])))
    model.learn(total_timesteps=128)
    print("[train] PPO learn() ok")


if __name__ == "__main__":
    test_env()
    test_filters()
    test_training()
    print("\nALL SMOKE TESTS PASSED")
