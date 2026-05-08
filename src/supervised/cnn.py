"""Competitive 1D CNN denoiser baseline.

Deeper architecture with residual connections, proper training schedule,
and multi-family training. This serves as a realistic supervised upper bound.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class Conv1DDenoiser(nn.Module):
    def __init__(self, window: int = 64, channels: int = 64, n_res_blocks: int = 4):
        super().__init__()
        self.window = window
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[
            ResBlock1D(channels, kernel_size=5) for _ in range(n_res_blocks)
        ])
        self.decoder = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, 1, kernel_size=7, padding=3),
        )
        self.head = nn.Linear(window, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)
        h = self.encoder(h)
        h = self.res_blocks(h)
        h = self.decoder(h)
        return self.head(h.squeeze(1).flatten(1)).squeeze(-1)


def make_windows(noisy: np.ndarray, clean: np.ndarray, window: int):
    N = len(noisy)
    pad = np.concatenate([np.zeros(window - 1), noisy])
    X = np.lib.stride_tricks.sliding_window_view(pad, window)[:N]
    return np.ascontiguousarray(X, dtype=np.float32), clean.astype(np.float32)


def train_cnn(noisy_train: np.ndarray, clean_train: np.ndarray,
              window: int = 64, epochs: int = 50, batch_size: int = 256,
              lr: float = 1e-3, device: str = "cpu", verbose: bool = False,
              val_split: float = 0.1) -> Conv1DDenoiser:
    X, y = make_windows(noisy_train, clean_train, window)
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    n_val = int(len(X_t) * val_split)
    if n_val > 0:
        perm = torch.randperm(len(X_t))
        X_val, y_val = X_t[perm[:n_val]], y_t[perm[:n_val]]
        X_t, y_t = X_t[perm[n_val:]], y_t[perm[n_val:]]
    else:
        X_val, y_val = X_t[:1], y_t[:1]

    model = Conv1DDenoiser(window=window).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )
    loss_fn = nn.MSELoss()

    N = X_t.shape[0]
    best_val = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        total = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * xb.shape[0]

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val), y_val))

        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{epochs}  train_mse={total/N:.6f}  val_mse={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def cnn_predict(model: Conv1DDenoiser, noisy: np.ndarray, window: int,
                device: str = "cpu") -> np.ndarray:
    X, _ = make_windows(noisy, noisy, window)
    X_t = torch.from_numpy(X).to(device)
    pred = model(X_t).cpu().numpy()
    return pred
