from __future__ import annotations

from dataclasses import replace

import numpy as np

from .base import BaseGame, GameState


# Piece encoding: positive = player 1, negative = player -1
# 1 pawn, 2 knight, 3 bishop, 4 rook, 5 queen, 6 king


class GardnerChessGame(BaseGame):
    name = "gardner_chess"
    board_size = 5
    action_size = board_size * board_size * board_size * board_size
    obs_size = 256

    def initial_state(self) -> GameState:
        board = np.zeros((5, 5), dtype=np.int8)
        board[0] = np.array([-4, -2, -3, -5, -6], dtype=np.int8)
        board[1] = -1
        board[3] = 1
        board[4] = np.array([4, 2, 3, 5, 6], dtype=np.int8)
        return GameState(board=board)

    def legal_actions(self, state: GameState) -> list[int]:
        legal: list[int] = []
        for r in range(5):
            for c in range(5):
                piece = int(state.board[r, c])
                if piece == 0 or np.sign(piece) != state.current_player:
                    continue
                for rr, cc in self._pseudo_moves(state.board, r, c):
                    if self._would_be_legal(state, r, c, rr, cc):
                        legal.append(self._encode(r, c, rr, cc))
        return legal

    def apply_action(self, state: GameState, action: int) -> GameState:
        if state.done:
            return state
        r, c, rr, cc = self._decode(action)
        if action not in self.legal_actions(state):
            raise ValueError("Illegal Gardner chess move")
        board = state.board.copy()
        piece = board[r, c]
        board[r, c] = 0
        if abs(piece) == 1 and (rr == 0 or rr == 4):
            piece = 5 * int(np.sign(piece))
        board[rr, cc] = piece

        enemy_king_alive = np.any(board == -6 * state.current_player)
        no_reply = len(self.legal_actions(replace(state, board=board, current_player=-state.current_player))) == 0
        done = not enemy_king_alive or no_reply
        winner = state.current_player if done else None
        return replace(state, board=board, current_player=-state.current_player, done=done, winner=winner)

    def _encode(self, r: int, c: int, rr: int, cc: int) -> int:
        return ((r * 5 + c) * 25) + (rr * 5 + cc)

    def _decode(self, action: int) -> tuple[int, int, int, int]:
        src, dst = divmod(action, 25)
        r, c = divmod(src, 5)
        rr, cc = divmod(dst, 5)
        return r, c, rr, cc

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < 5 and 0 <= c < 5

    def _pseudo_moves(self, board: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
        piece = int(board[r, c])
        color = int(np.sign(piece))
        piece = abs(piece)
        moves: list[tuple[int, int]] = []

        def slide(dirs: list[tuple[int, int]]) -> None:
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                while self._in_bounds(rr, cc):
                    if board[rr, cc] == 0:
                        moves.append((rr, cc))
                    else:
                        if np.sign(board[rr, cc]) != color:
                            moves.append((rr, cc))
                        break
                    rr += dr
                    cc += dc

        if piece == 1:
            step = -1 if color == 1 else 1
            if self._in_bounds(r + step, c) and board[r + step, c] == 0:
                moves.append((r + step, c))
            for dc in (-1, 1):
                rr, cc = r + step, c + dc
                if self._in_bounds(rr, cc) and board[rr, cc] != 0 and np.sign(board[rr, cc]) != color:
                    moves.append((rr, cc))
        elif piece == 2:
            for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                rr, cc = r + dr, c + dc
                if self._in_bounds(rr, cc) and np.sign(board[rr, cc]) != color:
                    moves.append((rr, cc))
        elif piece == 3:
            slide([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        elif piece == 4:
            slide([(-1, 0), (1, 0), (0, -1), (0, 1)])
        elif piece == 5:
            slide([(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)])
        elif piece == 6:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if self._in_bounds(rr, cc) and np.sign(board[rr, cc]) != color:
                        moves.append((rr, cc))
        return moves

    def _would_be_legal(self, state: GameState, r: int, c: int, rr: int, cc: int) -> bool:
        board = state.board.copy()
        piece = board[r, c]
        board[r, c] = 0
        board[rr, cc] = piece
        king = 6 * state.current_player
        king_pos = np.argwhere(board == king)
        if king_pos.size == 0:
            return False
        kr, kc = king_pos[0]
        enemy = -state.current_player
        for er in range(5):
            for ec in range(5):
                if np.sign(board[er, ec]) == enemy:
                    for mr, mc in self._pseudo_moves(board, er, ec):
                        if mr == kr and mc == kc:
                            return False
        return True

    def encode_state(self, state: GameState) -> np.ndarray:
        vec = np.zeros(self.obs_size, dtype=np.float32)
        flat = state.board.flatten().astype(np.float32) / 6.0
        vec[: flat.shape[0]] = flat
        vec[-1] = float(state.current_player)
        return vec

    def pretty(self, state: GameState) -> str:
        symbol = {
            0: ".",
            1: "P",
            2: "N",
            3: "B",
            4: "R",
            5: "Q",
            6: "K",
            -1: "p",
            -2: "n",
            -3: "b",
            -4: "r",
            -5: "q",
            -6: "k",
        }
        return "\n".join(" ".join(symbol[int(v)] for v in row) for row in state.board)
