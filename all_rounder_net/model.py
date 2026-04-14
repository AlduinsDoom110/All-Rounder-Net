from __future__ import annotations

import torch
from torch import nn


class MultiGameNet(nn.Module):
    def __init__(self, max_obs_size: int, max_action_size: int, game_count: int, hidden: int = 256) -> None:
        super().__init__()
        self.game_embedding = nn.Embedding(game_count, 32)
        self.encoder = nn.Sequential(
            nn.Linear(max_obs_size + 32, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, max_action_size)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, game_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        game_vec = self.game_embedding(game_ids)
        x = torch.cat([obs, game_vec], dim=-1)
        hidden = self.encoder(x)
        policy_logits = self.policy_head(hidden)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        return policy_logits, value
