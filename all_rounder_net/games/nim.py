from __future__ import annotations

from dataclasses import replace

import numpy as np

from .base import BaseGame, GameState


class NimGame(BaseGame):
    name = "nim"
    max_piles = 8
    max_remove = 8
    action_size = max_piles * max_remove
    obs_size = 32

    def initial_state(self, piles: list[int] | None = None, max_remove: int = 3) -> GameState:
        piles = piles or [7, 7, 7]
        if len(piles) > self.max_piles:
            raise ValueError(f"Nim supports at most {self.max_piles} piles")
        max_remove = min(max_remove, self.max_remove)
        return GameState(board=list(piles), meta={"max_remove": max_remove, "pile_count": len(piles)})

    def legal_actions(self, state: GameState) -> list[int]:
        max_remove = state.meta["max_remove"]
        actions: list[int] = []
        for pile_idx, stones in enumerate(state.board):
            for remove in range(1, min(stones, max_remove) + 1):
                actions.append(pile_idx * self.max_remove + (remove - 1))
        return actions

    def apply_action(self, state: GameState, action: int) -> GameState:
        if state.done:
            return state
        pile_idx = action // self.max_remove
        remove = (action % self.max_remove) + 1
        board = list(state.board)
        if pile_idx >= len(board) or remove > board[pile_idx] or remove > state.meta["max_remove"]:
            raise ValueError("Illegal Nim move")
        board[pile_idx] -= remove
        done = sum(board) == 0
        winner = state.current_player if done else None
        return replace(state, board=board, current_player=-state.current_player, done=done, winner=winner)

    def encode_state(self, state: GameState) -> np.ndarray:
        vec = np.zeros(self.obs_size, dtype=np.float32)
        for i, stones in enumerate(state.board[: self.max_piles]):
            vec[i] = stones / 20.0
        vec[self.max_piles] = state.meta["max_remove"] / self.max_remove
        vec[-1] = float(state.current_player)
        return vec

    def pretty(self, state: GameState) -> str:
        return " | ".join(f"Pile {idx}: {stones}" for idx, stones in enumerate(state.board))
