from __future__ import annotations

from dataclasses import replace

import numpy as np

from .base import BaseGame, GameState


class DotsAndBoxesGame(BaseGame):
    name = "dots_and_boxes"
    action_size = 128
    obs_size = 256

    def initial_state(self, width: int = 2, height: int = 2) -> GameState:
        if width < 1 or height < 1 or width > 6 or height > 6:
            raise ValueError("Dots and Boxes width/height must be in [1, 6]")
        h_edges = np.zeros((height + 1, width), dtype=np.int8)
        v_edges = np.zeros((height, width + 1), dtype=np.int8)
        owners = np.zeros((height, width), dtype=np.int8)
        return GameState(board={"h": h_edges, "v": v_edges, "o": owners}, meta={"width": width, "height": height})

    def legal_actions(self, state: GameState) -> list[int]:
        width = state.meta["width"]
        height = state.meta["height"]
        actions: list[int] = []
        idx = 0
        for r in range(height + 1):
            for c in range(width):
                if state.board["h"][r, c] == 0:
                    actions.append(idx)
                idx += 1
        for r in range(height):
            for c in range(width + 1):
                if state.board["v"][r, c] == 0:
                    actions.append(idx)
                idx += 1
        return actions

    def apply_action(self, state: GameState, action: int) -> GameState:
        if state.done:
            return state
        width = state.meta["width"]
        height = state.meta["height"]
        h = state.board["h"].copy()
        v = state.board["v"].copy()
        o = state.board["o"].copy()

        h_count = (height + 1) * width
        if action < h_count:
            r, c = divmod(action, width)
            if h[r, c] != 0:
                raise ValueError("Illegal edge move")
            h[r, c] = 1
        else:
            rel = action - h_count
            r, c = divmod(rel, width + 1)
            if r >= height or v[r, c] != 0:
                raise ValueError("Illegal edge move")
            v[r, c] = 1

        gained = 0
        for r in range(height):
            for c in range(width):
                if o[r, c] == 0 and h[r, c] and h[r + 1, c] and v[r, c] and v[r, c + 1]:
                    o[r, c] = state.current_player
                    gained += 1

        remaining = np.any(h == 0) or np.any(v == 0)
        done = not remaining
        winner = None
        if done:
            score = int(np.sum(o))
            winner = 1 if score > 0 else -1 if score < 0 else 0

        next_player = state.current_player if gained > 0 and not done else -state.current_player
        return replace(state, board={"h": h, "v": v, "o": o}, current_player=next_player, done=done, winner=winner)

    def encode_state(self, state: GameState) -> np.ndarray:
        vec = np.zeros(self.obs_size, dtype=np.float32)
        h = state.board["h"].flatten()
        v = state.board["v"].flatten()
        o = state.board["o"].flatten()
        i = 0
        vec[i : i + len(h)] = h
        i += len(h)
        vec[i : i + len(v)] = v
        i += len(v)
        vec[i : i + len(o)] = o
        vec[-2] = state.meta["width"] / 6.0
        vec[-3] = state.meta["height"] / 6.0
        vec[-1] = float(state.current_player)
        return vec

    def pretty(self, state: GameState) -> str:
        return f"Boxes claimed (P1=1,P2=-1):\n{state.board['o']}"
