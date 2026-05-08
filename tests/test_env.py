"""Smoke tests for the Gymnasium environment."""
import numpy as np
from src.envs import AdaptiveFilterEnv, EnvConfig
from src.envs.adaptive_filter_env import FEAT_DIM


def test_env_random_rollout():
    env = AdaptiveFilterEnv(EnvConfig(episode_len=500), seed=0)
    obs, info = env.reset(seed=0)
    expected_dim = FEAT_DIM * EnvConfig.state_window
    assert obs.shape == (expected_dim,), f"Expected ({expected_dim},), got {obs.shape}"
    assert obs.shape == env.observation_space.shape
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(500):
        a = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term:
            break
    assert np.isfinite(total), f"Non-finite reward: {total}"
    assert "episode_ss_mse" in info


def test_env_constant_action():
    env = AdaptiveFilterEnv(EnvConfig(episode_len=400),
                            fixed_family="gaussian", fixed_snr_db=10.0, seed=1)
    env.reset(seed=1)
    a = np.array([-1.0, 1.0], dtype=np.float32)
    last = None
    for _ in range(400):
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r), f"Non-finite reward: {r}"
        last = info
        if term:
            break
    assert "episode_ss_mse" in last


def test_env_reward_bounded():
    env = AdaptiveFilterEnv(EnvConfig(episode_len=300),
                            fixed_family="impulsive", fixed_snr_db=5.0, seed=2)
    env.reset(seed=2)
    rng = np.random.default_rng(2)
    for _ in range(300):
        a = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        assert r <= 0.0, f"Positive reward: {r}"
        assert r > -10.5, f"Reward too negative: {r}"
        if term:
            break


def test_env_all_families():
    from src.noise.families import TRAIN_FAMILIES, OOD_FAMILIES
    for fam in list(TRAIN_FAMILIES) + list(OOD_FAMILIES):
        env = AdaptiveFilterEnv(EnvConfig(episode_len=200),
                                fixed_family=fam, fixed_snr_db=10.0, seed=0)
        obs, _ = env.reset(seed=0)
        a = np.array([0.0, 0.0], dtype=np.float32)
        obs, r, term, _, _ = env.step(a)
        assert np.isfinite(r), f"Non-finite reward for {fam}: {r}"


if __name__ == "__main__":
    test_env_random_rollout()
    test_env_constant_action()
    test_env_reward_bounded()
    test_env_all_families()
    print("All environment tests passed.")
