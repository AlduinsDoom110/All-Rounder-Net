from __future__ import annotations

from dataclasses import replace

import numpy as np

from .base import BaseGame, GameState


class TicTacToeGame(BaseGame):
    name = "tictactoe"
    action_size = 25  # supports up to 5x5
    obs_size = 64

    def initial_state(self, size: int = 3) -> GameState:
        if size < 3 or size > 5:
            raise ValueError("Tic Tac Toe size must be between 3 and 5")
        board = np.zeros((size, size), dtype=np.int8)
        return GameState(board=board, meta={"size": size})

    def legal_actions(self, state: GameState) -> list[int]:
        size = state.meta["size"]
        return [r * size + c for r in range(size) for c in range(size) if state.board[r, c] == 0]

    def apply_action(self, state: GameState, action: int) -> GameState:
        if state.done:
            return state
        size = state.meta["size"]
        r, c = divmod(action, size)
        if r >= size or c >= size or state.board[r, c] != 0:
            raise ValueError("Illegal Tic Tac Toe move")
        board = state.board.copy()
        board[r, c] = state.current_player
        winner = self._winner(board)
        done = winner is not None or np.all(board != 0)
        return replace(state, board=board, current_player=-state.current_player, done=done, winner=winner)

    def _winner(self, board: np.ndarray) -> int | None:
        n = board.shape[0]
        lines = [board[i, :] for i in range(n)] + [board[:, i] for i in range(n)]
        lines.append(np.diag(board))
        lines.append(np.diag(np.fliplr(board)))
        for line in lines:
            s = int(np.sum(line))
            if s == n:
                return 1
            if s == -n:
                return -1
        return None

    def encode_state(self, state: GameState) -> np.ndarray:
        size = state.meta["size"]
        vec = np.zeros(self.obs_size, dtype=np.float32)
        flat = state.board.flatten()
        vec[: flat.shape[0]] = flat
        vec[30] = size / 5.0
        vec[-1] = float(state.current_player)
        return vec

    def pretty(self, state: GameState) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        return "\n".join(" ".join(symbols[int(v)] for v in row) for row in state.board)
