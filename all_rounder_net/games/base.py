from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GameState:
    board: Any
    current_player: int = 1
    done: bool = False
    winner: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BaseGame:
    name: str = "base"
    action_size: int = 0
    obs_size: int = 0

    def initial_state(self, **kwargs: Any) -> GameState:
        raise NotImplementedError

    def legal_actions(self, state: GameState) -> list[int]:
        raise NotImplementedError

    def apply_action(self, state: GameState, action: int) -> GameState:
        raise NotImplementedError

    def encode_state(self, state: GameState) -> np.ndarray:
        raise NotImplementedError

    def pretty(self, state: GameState) -> str:
        return str(state.board)
