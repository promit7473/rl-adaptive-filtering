"""Smoke tests for the v2 Gymnasium environment."""
import numpy as np
from src.envs import AdaptiveFilterEnvV2, EnvConfigV2
from src.envs.adaptive_filter_env_v2 import FEAT_DIM_V2


def test_env_random_rollout():
    cfg = EnvConfigV2(episode_len=500)
    env = AdaptiveFilterEnvV2(cfg, seed=0)
    obs, info = env.reset(seed=0)
    expected_dim = FEAT_DIM_V2 * cfg.state_window
    assert obs.shape == (expected_dim,), f"Expected ({expected_dim},), got {obs.shape}"
    assert obs.shape == env.observation_space.shape
    rng = np.random.default_rng(0)
    total = 0.0
    info = {}
    for _ in range(cfg.episode_len):
        a = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    assert np.isfinite(total), f"Non-finite return: {total}"
    assert "episode_ss_mse" in info


def test_env_fixed_family():
    cfg = EnvConfigV2(episode_len=400)
    env = AdaptiveFilterEnvV2(cfg, fixed_family="gaussian", fixed_snr_db=10.0, seed=1)
    env.reset(seed=1)
    a = np.array([0.0, 0.0], dtype=np.float32)
    last = {}
    for _ in range(cfg.episode_len):
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r), f"Non-finite reward: {r}"
        last = info
        if term or trunc:
            break
    assert "episode_ss_mse" in last


def test_env_all_families():
    from src.noise.families import TRAIN_FAMILIES, OOD_FAMILIES
    for fam in list(TRAIN_FAMILIES) + list(OOD_FAMILIES):
        env = AdaptiveFilterEnvV2(EnvConfigV2(episode_len=200),
                                  fixed_family=fam, fixed_snr_db=10.0, seed=0)
        obs, _ = env.reset(seed=0)
        obs, r, term, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert np.isfinite(r), f"Non-finite reward for {fam}: {r}"


if __name__ == "__main__":
    test_env_random_rollout()
    test_env_fixed_family()
    test_env_all_families()
    print("All environment tests passed.")
