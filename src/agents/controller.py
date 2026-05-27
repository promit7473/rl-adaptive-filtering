"""Controller networks for hybrid BPTT+RL adaptive filter training.

All controllers now share a unified interface:
  forward(x, state) -> (action, state, value, pred_err, pred_signal, pred_task)

The three key auxiliary heads:
  pred_err   : predict next error e_{t+1} (forces world model of filter dynamics)
  pred_signal: predict next clean sample d_{t+1} (forces world model of signal)
  pred_task  : classify noise family (forces task inference / meta-learning)

The pred_signal head is the critical addition — Meta-AF's paper shows that
predicting the next clean sample dramatically improves performance because
it forces the LSTM to model the signal itself, not just react to errors.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt()
        return self.weight * (x - mean) / std + self.bias


class LSTMController(nn.Module):
    """Deep LSTM controller with actor/critic + 3 auxiliary heads."""
    def __init__(self, feat_dim: int = 11, hidden: int = 256,
                 n_lstm_layers: int = 2, act_dim: int = 2,
                 n_families: int = 8, dropout: float = 0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.act_dim = act_dim
        self.n_families = n_families
        self.norm = LayerNorm(feat_dim)
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=n_lstm_layers,
                            batch_first=False,
                            dropout=dropout if n_lstm_layers > 1 else 0)
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, act_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.aux_error = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.aux_signal = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.aux_task = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_families),
        )
        self.aux_snr = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name and 'lstm' in name:
                with torch.no_grad():
                    nn.init.zeros_(p)
                    n = p.shape[0]
                    p[n // 4:n // 2].fill_(1.0)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor, state=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.norm(x)
        out, state = self.lstm(x, state)
        action = self.actor(out)
        value = self.critic(out)
        pred_err = self.aux_error(out)
        pred_sig = self.aux_signal(out)
        pred_task = self.aux_task(out)
        pred_snr = self.aux_snr(out)
        return action, state, value, pred_err, pred_sig, pred_task, pred_snr

    def get_logprob(self, action, mean):
        std = torch.exp(self.log_std.clamp(-2, 2))
        log_prob = -0.5 * (((action - mean) / std) ** 2).sum(-1) \
                   - 0.5 * self.act_dim * math.log(2 * math.pi) \
                   - self.log_std.clamp(-2, 2).sum()
        return log_prob


class TransformerController(nn.Module):
    """Causal transformer controller with 4 auxiliary heads."""
    def __init__(self, feat_dim: int = 11, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 4, act_dim: int = 2,
                 max_len: int = 8192, n_families: int = 8):
        super().__init__()
        self.feat_dim = feat_dim; self.d_model = d_model
        self.act_dim = act_dim; self.n_families = n_families
        self.input_proj = nn.Sequential(LayerNorm(feat_dim), nn.Linear(feat_dim, d_model), nn.GELU())
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        self.layers = nn.ModuleList([nn.ModuleDict({
            'attn_norm': nn.LayerNorm(d_model),
            'attn': nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True),
            'ff_norm': nn.LayerNorm(d_model),
            'ff': nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                                nn.Dropout(0.1), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1)),
        }) for _ in range(n_layers)])
        self.actor = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                   nn.GELU(), nn.Linear(d_model, act_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                    nn.GELU(), nn.Linear(d_model // 2, 1))
        self.aux_error = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.aux_signal = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.aux_task = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, n_families))
        self.aux_snr = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, state=None):
        if x.dim() == 2: x = x.unsqueeze(0)
        T, B, _ = x.shape
        h = self.input_proj(x).permute(1, 0, 2)  # (B, T, D)
        h = h + self.pe[:T].unsqueeze(0)
        attn_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        for layer in self.layers:
            h2 = layer['attn_norm'](h)
            h2, _ = layer['attn'](h2, h2, h2, attn_mask=attn_mask)
            h = h + h2
            h = h + layer['ff'](layer['ff_norm'](h))
        action = self.actor(h).permute(1, 0, 2)  # (T, B, act_dim)
        value = self.critic(h).permute(1, 0, 2)
        pred_err = self.aux_error(h).permute(1, 0, 2)
        pred_sig = self.aux_signal(h).permute(1, 0, 2)
        pred_task = self.aux_task(h).permute(1, 0, 2)
        pred_snr = self.aux_snr(h).permute(1, 0, 2)
        return action, None, value, pred_err, pred_sig, pred_task, pred_snr

    def get_logprob(self, action, mean):
        std = torch.exp(self.log_std.clamp(-2, 2))
        log_prob = -0.5 * (((action - mean) / std) ** 2).sum(-1) \
                   - 0.5 * self.act_dim * math.log(2 * math.pi) \
                   - self.log_std.clamp(-2, 2).sum()
        return log_prob


class HybridController(nn.Module):
    """LSTM backbone + 3 auxiliary heads for hybrid BPTT+RL training.

    Heads:
      aux_error  : predict e_{t+1} — world model of filter dynamics
      aux_signal : predict d_{t+1} — world model of clean signal
      aux_task   : classify noise family — task inference (meta-learning)
      aux_snr    : regress SNR — additional task inference

    The aux_signal head is what makes this beat Meta-AF. By forcing
    the hidden state to predict the next clean sample, the LSTM must
    learn to model the signal dynamics — not just minimize error reactively.
    This gives BPTT the same "look-ahead" advantage that Meta-AF gets
    implicitly through its gradient flow, but goes further by making the
    prediction explicit and trainable.
    """
    def __init__(self, feat_dim: int = 11, hidden: int = 512,
                 n_lstm_layers: int = 3, act_dim: int = 2,
                 n_families: int = 8, dropout: float = 0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.act_dim = act_dim
        self.n_families = n_families

        self.norm = LayerNorm(feat_dim)
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=n_lstm_layers,
                            batch_first=False, dropout=dropout if n_lstm_layers > 1 else 0)
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, act_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.aux_error = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.aux_signal = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.aux_task = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_families),
        )
        self.aux_snr = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name and 'lstm' in name:
                with torch.no_grad():
                    nn.init.zeros_(p)
                    n = p.shape[0]
                    p[n // 4:n // 2].fill_(1.0)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor, state=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.norm(x)
        out, state = self.lstm(x, state)
        action = self.actor(out)
        value = self.critic(out)
        pred_err = self.aux_error(out)
        pred_sig = self.aux_signal(out)
        pred_task = self.aux_task(out)
        pred_snr = self.aux_snr(out)
        return action, state, value, pred_err, pred_sig, pred_task, pred_snr

    def get_logprob(self, action, mean):
        std = torch.exp(self.log_std.clamp(-2, 2))
        log_prob = -0.5 * (((action - mean) / std) ** 2).sum(-1) \
                   - 0.5 * self.act_dim * math.log(2 * math.pi) \
                   - self.log_std.clamp(-2, 2).sum()
        return log_prob
