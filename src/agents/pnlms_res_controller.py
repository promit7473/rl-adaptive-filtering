"""PNLMS+Residual Controller: LSTM with per-tap mu + lambda + delta output.

Action space: M+2 continuous values
  - mu_t[0..M-1]: per-tap step sizes (proportionate NLMS)
  - lam_t: global leakage factor
  - delta_t: learned residual correction (post-filter)

Feature space: 12 + M (11 base + y_filter + weight snippet)
The y_filter feature lets the controller know the current filter output,
which is critical for learning a good residual correction.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from .controller import LayerNorm


class PNLMSResController(nn.Module):
    def __init__(self, feat_dim: int = 28, hidden: int = 256,
                 n_lstm_layers: int = 2, filter_order: int = 16,
                 n_families: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.filter_order = filter_order
        self.act_dim = filter_order + 2
        self.n_families = n_families

        self.norm = LayerNorm(feat_dim)
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=n_lstm_layers,
                            batch_first=False)

        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.act_dim),
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
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))
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
        return action, state, value, pred_err, pred_sig, pred_task

    def get_logprob(self, action, mean):
        std = torch.exp(self.log_std.clamp(-2, 2))
        log_prob = -0.5 * (((action - mean) / std) ** 2).sum(-1) \
                   - 0.5 * self.act_dim * math.log(2 * math.pi) \
                   - self.log_std.clamp(-2, 2).sum()
        return log_prob
