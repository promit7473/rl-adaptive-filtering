#!/usr/bin/env python3
"""Comprehensive smoke test for GPU, environment, and training pipeline."""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def test_gpu_availability():
    """Check CUDA/GPU availability and specs."""
    print("=" * 70)
    print("GPU AVAILABILITY TEST")
    print("=" * 70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  - Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  - Multi-processors: {props.multi_processor_count}")
            print(f"  - Compute capability: {props.major}.{props.minor}")
        
        # Test GPU tensor operations
        print("\nTesting GPU tensor operations...")
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        start = time.time()
        for _ in range(100):
            z = torch.mm(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  100 matrix multiplications (1000x1000): {elapsed:.4f}s")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        return True
    else:
        print("WARNING: CUDA not available! Training will use CPU only.")
        return False


def test_numpy_operations():
    """Test NumPy and signal processing."""
    print("\n" + "=" * 70)
    print("NUMPY/SCIPY TEST")
    print("=" * 70)
    
    from src.signals.generators import make_signal
    from src.noise.families import make_noise, TRAIN_FAMILIES, OOD_FAMILIES
    
    rng = np.random.default_rng(42)
    
    # Test signal generation
    print("Testing signal generation...")
    for kind in ["sine", "multitone", "am", "chirp"]:
        x = make_signal(kind, n=2000, fs=8000.0, rng=rng)
        assert x.shape == (2000,), f"Shape mismatch for {kind}"
        assert np.isfinite(x).all(), f"Non-finite values in {kind}"
        print(f"  ✓ {kind:12s} shape={x.shape}")
    
    # Test noise generation
    print("\nTesting noise families...")
    clean = make_signal("multitone", n=4000, fs=8000.0, rng=rng)
    for fam in TRAIN_FAMILIES + OOD_FAMILIES:
        kwargs = {"fs": 8000.0} if fam == "chirp_interferer" else {}
        noise = make_noise(fam, clean, rng, snr_db=10.0, **kwargs)
        assert noise.shape == clean.shape, f"Shape mismatch for {fam}"
        assert np.isfinite(noise).all(), f"Non-finite values in {fam}"
        print(f"  ✓ {fam:20s} shape={noise.shape}")
    
    print("\n✓ All NumPy/Scipy operations passed")
    return True


def test_filters():
    """Test classical adaptive filters."""
    print("\n" + "=" * 70)
    print("CLASSICAL FILTERS TEST")
    print("=" * 70)
    
    from src.signals.generators import make_signal
    from src.noise.families import make_noise
    from src.filters import make_filter, windowize
    
    rng = np.random.default_rng(42)
    n = 4000
    
    clean = make_signal("multitone", n=n, fs=8000.0, rng=rng)
    noise = make_noise("gaussian", clean, rng, snr_db=10.0)
    noisy = clean + noise
    
    U = windowize(noisy, order=16)
    d = clean
    
    raw_mse = float(np.mean((noisy - clean) ** 2))
    print(f"Raw MSE (no filter): {raw_mse:.6f}")
    
    for name, kw in [
        ("lms", dict(mu=0.01)),
        ("nlms", dict(mu=0.5)),
        ("vss_lms", dict(mu_max=0.05)),
        ("rls", dict(forgetting=0.995)),
    ]:
        f = make_filter(name, order=16, **kw)
        y, e = f.run(U, d)
        ss_mse = float(np.mean(e[-1000:] ** 2))
        improvement = (raw_mse - ss_mse) / raw_mse * 100
        print(f"  {name:8s}  SS-MSE={ss_mse:.6f}  improvement={improvement:+.1f}%")
        assert ss_mse < raw_mse, f"{name} did not improve"
    
    print("\n✓ All filters passed")
    return True


def test_environment():
    """Test Gymnasium environment."""
    print("\n" + "=" * 70)
    print("ENVIRONMENT TEST")
    print("=" * 70)
    
    from src.envs import AdaptiveFilterEnv, EnvConfig
    
    env = AdaptiveFilterEnv(EnvConfig(episode_len=500), seed=42)
    obs, info = env.reset(seed=42)
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Initial obs shape: {obs.shape}")
    
    rng = np.random.default_rng(42)
    total_reward = 0.0
    steps = 0
    
    for _ in range(500):
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated:
            break
    
    print(f"\nRandom rollout: {steps} steps, total reward={total_reward:.2f}")
    print(f"Episode info: {info}")
    
    assert np.isfinite(total_reward), "Non-finite reward"
    assert "episode_ss_mse" in info, "Missing episode_ss_mse in info"
    
    print("\n✓ Environment passed")
    return True


def test_cnn_baseline(device: str = "cpu"):
    """Test CNN denoiser on CPU/GPU."""
    print("\n" + "=" * 70)
    print(f"CNN BASELINE TEST (device={device})")
    print("=" * 70)
    
    from src.signals.generators import make_signal
    from src.noise.families import make_noise
    from src.supervised.cnn import train_cnn, cnn_predict
    
    rng = np.random.default_rng(42)
    n_train = 2000
    n_test = 500
    
    # Generate training data
    clean_train = make_signal("multitone", n=n_train, fs=8000.0, rng=rng)
    noise_train = make_noise("gaussian", clean_train, rng, snr_db=10.0)
    noisy_train = clean_train + noise_train
    
    print(f"Training CNN on {device}...")
    start = time.time()
    model = train_cnn(noisy_train, clean_train, window=32, epochs=2, 
                      batch_size=128, device=device, verbose=True)
    train_time = time.time() - start
    print(f"Training time: {train_time:.2f}s")
    
    # Test prediction
    clean_test = make_signal("sine", n=n_test, fs=8000.0, rng=rng)
    noise_test = make_noise("gaussian", clean_test, rng, snr_db=10.0)
    noisy_test = clean_test + noise_test
    
    pred = cnn_predict(model, noisy_test, window=32, device=device)
    mse = float(np.mean((pred - clean_test) ** 2))
    raw_mse = float(np.mean((noisy_test - clean_test) ** 2))
    
    print(f"\nTest MSE (raw): {raw_mse:.6f}")
    print(f"Test MSE (CNN): {mse:.6f}")
    print(f"Improvement: {(raw_mse - mse) / raw_mse * 100:+.1f}%")
    
    print("\n✓ CNN baseline passed")
    return True


def test_ppo_training(device: str = "cpu", n_steps: int = 1000):
    """Test PPO training on CPU/GPU."""
    print("\n" + "=" * 70)
    print(f"PPO TRAINING TEST (device={device}, steps={n_steps})")
    print("=" * 70)
    
    from src.agents.train_ppo import train
    
    print(f"Training RecurrentPPO for {n_steps} steps...")
    start = time.time()
    
    model, records, path = train(
        total_timesteps=n_steps,
        n_envs=2,
        episode_len=200,
        hidden_size=32,
        n_steps=16,
        batch_size=16,
        seed=42,
        device=device,
        out_dir="/tmp/smoke_test_ppo",
        tag="smoke",
    )
    
    elapsed = time.time() - start
    print(f"\nTraining time: {elapsed:.2f}s")
    print(f"Steps per second: {n_steps / elapsed:.1f}")
    print(f"Model saved to: {path}")
    
    if records:
        last_record = records[-1]
        print(f"Last episode SS-MSE: {last_record['ss_mse']:.6f}")
        print(f"Last episode SS-MSE (dB): {last_record['ss_mse_db']:+.2f}")
    
    print("\n✓ PPO training passed")
    return True


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("RL ADAPTIVE FILTERING - SMOKE TEST SUITE")
    print("=" * 70)
    
    # Determine best device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    results = {}
    
    # Run tests
    try:
        results["GPU"] = test_gpu_availability()
    except Exception as e:
        print(f"FAILED: {e}")
        results["GPU"] = False
    
    try:
        results["NumPy"] = test_numpy_operations()
    except Exception as e:
        print(f"FAILED: {e}")
        results["NumPy"] = False
    
    try:
        results["Filters"] = test_filters()
    except Exception as e:
        print(f"FAILED: {e}")
        results["Filters"] = False
    
    try:
        results["Environment"] = test_environment()
    except Exception as e:
        print(f"FAILED: {e}")
        results["Environment"] = False
    
    try:
        results["CNN_CPU"] = test_cnn_baseline(device="cpu")
    except Exception as e:
        print(f"FAILED: {e}")
        results["CNN_CPU"] = False
    
    if device == "cuda":
        try:
            results["CNN_GPU"] = test_cnn_baseline(device="cuda")
        except Exception as e:
            print(f"FAILED: {e}")
            results["CNN_GPU"] = False
    
    try:
        results["PPO_CPU"] = test_ppo_training(device="cpu", n_steps=500)
    except Exception as e:
        print(f"FAILED: {e}")
        results["PPO_CPU"] = False
    
    if device == "cuda":
        try:
            results["PPO_GPU"] = test_ppo_training(device="cuda", n_steps=500)
        except Exception as e:
            print(f"FAILED: {e}")
            results["PPO_GPU"] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("Your system is ready for training!")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
