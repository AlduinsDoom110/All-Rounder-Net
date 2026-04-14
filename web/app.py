from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from all_rounder_net.games import BaseGame, GameState, build_games
from all_rounder_net.model import MultiGameNet


class NewGameRequest(BaseModel):
    game: str
    params: dict[str, Any] = Field(default_factory=dict)
    human_player: int = 1


class MoveRequest(BaseModel):
    session_id: str
    action: int


app = FastAPI(title="All-Rounder-Net")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

GAMES: dict[str, BaseGame] = build_games()
GAME_NAMES = list(GAMES.keys())
GAME_TO_ID = {g: i for i, g in enumerate(GAME_NAMES)}
MAX_OBS = max(g.obs_size for g in GAMES.values())
MAX_ACTION = max(g.action_size for g in GAMES.values())
MODEL = MultiGameNet(MAX_OBS, MAX_ACTION, len(GAMES))
SESSIONS: dict[str, dict[str, Any]] = {}

ckpt_path = Path("checkpoints/multigame.pt")
if ckpt_path.exists():
    data = torch.load(ckpt_path, map_location="cpu")
    MODEL.load_state_dict(data["state_dict"])
MODEL.eval()


def _pad_obs(obs: np.ndarray) -> np.ndarray:
    out = np.zeros(MAX_OBS, dtype=np.float32)
    out[: obs.shape[0]] = obs
    return out


@torch.no_grad()
def _ai_action(game_name: str, state: GameState) -> int:
    game = GAMES[game_name]
    legal = game.legal_actions(state)
    if not legal:
        raise HTTPException(status_code=400, detail="No legal actions")
    obs = _pad_obs(game.encode_state(state))
    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
    gid_t = torch.tensor([GAME_TO_ID[game_name]], dtype=torch.long)
    logits, _ = MODEL(obs_t, gid_t)
    logits = logits[0].numpy()
    mask = np.full_like(logits, -1e9)
    mask[legal] = 0.0
    choice = int(np.argmax(logits + mask))
    return choice


def _serialize(game_name: str, state: GameState) -> dict[str, Any]:
    game = GAMES[game_name]
    board = state.board
    if isinstance(board, np.ndarray):
        board = board.tolist()
    elif isinstance(board, dict):
        board = {k: v.tolist() for k, v in board.items()}
    return {
        "game": game_name,
        "board": board,
        "current_player": state.current_player,
        "done": state.done,
        "winner": state.winner,
        "legal_actions": game.legal_actions(state),
        "pretty": game.pretty(state),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = Path("templates/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/games")
def games() -> dict[str, list[str]]:
    return {"games": GAME_NAMES}


@app.post("/api/new_game")
def new_game(req: NewGameRequest) -> dict[str, Any]:
    if req.game not in GAMES:
        raise HTTPException(status_code=404, detail=f"Unknown game: {req.game}")
    state = GAMES[req.game].initial_state(**req.params)
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"game": req.game, "state": state, "human_player": req.human_player}
    payload = _serialize(req.game, state)
    payload["session_id"] = sid

    if state.current_player != req.human_player:
        ai = _ai_action(req.game, state)
        state = GAMES[req.game].apply_action(state, ai)
        SESSIONS[sid]["state"] = state
        payload = _serialize(req.game, state)
        payload["session_id"] = sid
        payload["ai_action"] = ai
    return payload


@app.post("/api/move")
def move(req: MoveRequest) -> dict[str, Any]:
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session")
    session = SESSIONS[req.session_id]
    game_name = session["game"]
    game = GAMES[game_name]
    state: GameState = session["state"]
    if state.done:
        return {"session_id": req.session_id, **_serialize(game_name, state)}

    if req.action not in game.legal_actions(state):
        raise HTTPException(status_code=400, detail="Illegal move")
    state = game.apply_action(state, req.action)

    ai_action = None
    if not state.done and state.current_player != session["human_player"]:
        ai_action = _ai_action(game_name, state)
        state = game.apply_action(state, ai_action)

    session["state"] = state
    payload = _serialize(game_name, state)
    payload["session_id"] = req.session_id
    payload["ai_action"] = ai_action
    return payload


@app.get("/api/state/{session_id}")
def get_state(session_id: str) -> dict[str, Any]:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session")
    session = SESSIONS[session_id]
    payload = _serialize(session["game"], session["state"])
    payload["session_id"] = session_id
    return payload
