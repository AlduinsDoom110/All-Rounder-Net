from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from all_rounder_net.games import BaseGame, build_games
from all_rounder_net.model import MultiGameNet


@dataclass
class Transition:
    game_id: int
    obs: np.ndarray
    action: int
    reward: float


class MultiGameTrainer:
    def __init__(self, games: dict[str, BaseGame], device: str = "cpu") -> None:
        self.games = games
        self.game_names = list(games.keys())
        self.game_to_id = {name: idx for idx, name in enumerate(self.game_names)}
        self.max_obs = max(g.obs_size for g in games.values())
        self.max_action = max(g.action_size for g in games.values())
        self.device = device
        self.net = MultiGameNet(self.max_obs, self.max_action, len(self.games)).to(device)

    def pad_obs(self, obs: np.ndarray) -> np.ndarray:
        out = np.zeros(self.max_obs, dtype=np.float32)
        out[: obs.shape[0]] = obs
        return out

    def sample_episode(self) -> list[Transition]:
        game_name = random.choice(self.game_names)
        game = self.games[game_name]
        game_id = self.game_to_id[game_name]

        params = {}
        if game_name == "nim":
            params = {"piles": [random.randint(3, 10) for _ in range(random.randint(3, 5))], "max_remove": random.randint(2, 4)}
        elif game_name == "tictactoe":
            params = {"size": random.choice([3, 4, 5])}
        elif game_name == "dots_and_boxes":
            params = {"width": random.choice([2, 3]), "height": random.choice([2, 3])}

        state = game.initial_state(**params)
        trajectory: list[tuple[int, np.ndarray, int, int]] = []

        while not state.done:
            obs = self.pad_obs(game.encode_state(state))
            legal = game.legal_actions(state)
            if not legal:
                break
            action = random.choice(legal) if random.random() < 0.15 else self.select_action(obs, game_id, legal)
            trajectory.append((state.current_player, obs, action, game_id))
            state = game.apply_action(state, action)

        final_winner = 0 if state.winner is None else state.winner
        transitions: list[Transition] = []
        for player, obs, action, gid in trajectory:
            reward = 0.0 if final_winner == 0 else (1.0 if final_winner == player else -1.0)
            transitions.append(Transition(gid, obs, action, reward))
        return transitions

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, game_id: int, legal: list[int]) -> int:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        gid_t = torch.tensor([game_id], dtype=torch.long, device=self.device)
        logits, _ = self.net(obs_t, gid_t)
        logits = logits[0].cpu().numpy()
        mask = np.full_like(logits, -1e9)
        mask[legal] = 0.0
        probs = torch.softmax(torch.tensor(logits + mask), dim=0).numpy()
        return int(np.random.choice(len(probs), p=probs))

    def train(self, steps: int = 3000, batch_size: int = 128, lr: float = 3e-4) -> None:
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        replay: list[Transition] = []
        for step in range(1, steps + 1):
            replay.extend(self.sample_episode())
            if len(replay) < batch_size:
                continue
            batch = random.sample(replay, batch_size)
            obs = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=self.device)
            gids = torch.tensor([t.game_id for t in batch], dtype=torch.long, device=self.device)
            actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
            rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)

            logits, values = self.net(obs, gids)
            policy_loss = F.cross_entropy(logits, actions)
            value_loss = F.mse_loss(values, rewards)
            loss = policy_loss + value_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 200 == 0:
                print(f"step={step} loss={loss.item():.4f} replay={len(replay)}")

    def save(self, path: str | Path) -> None:
        data = {
            "state_dict": self.net.state_dict(),
            "game_names": self.game_names,
            "max_obs": self.max_obs,
            "max_action": self.max_action,
        }
        torch.save(data, path)


def main() -> None:
    games = build_games()
    trainer = MultiGameTrainer(games)
    trainer.train()
    out = Path("checkpoints")
    out.mkdir(exist_ok=True)
    trainer.save(out / "multigame.pt")


if __name__ == "__main__":
    main()
