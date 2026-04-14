# All-Rounder-Net

A Python multi-game neural network project that can train one model across several games and serve it in a browser-based HTTPS UI.

## Implemented games

- Nim (configurable piles and remove limit)
- Dots and Boxes (configurable board size up to 6x6 boxes)
- Tic Tac Toe (3x3 up to 5x5)
- Gardner Chess (5x5 mini-chess variant with core chess rules)

## Project layout

- `all_rounder_net/games/`: game definitions and state encoders
- `all_rounder_net/model.py`: shared policy/value neural network
- `all_rounder_net/trainer.py`: mixed-game self-play trainer
- `web/app.py`: FastAPI backend API + static UI host
- `templates/index.html` + `static/*`: browser UI
- `scripts/train.py`: training entrypoint
- `scripts/run_https.py`: HTTPS app launcher

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Train a shared model

```bash
python scripts/train.py
```

This writes a checkpoint to:

- `checkpoints/multigame.pt`

## Run web UI over HTTPS

Create a local self-signed certificate:

```bash
./scripts/generate_certs.sh
```

Start server with TLS:

```bash
python scripts/run_https.py --host 0.0.0.0 --port 8443 --certfile certs/localhost.pem --keyfile certs/localhost-key.pem
```

Then open:

- `https://localhost:8443`

(Your browser will warn about the self-signed certificate; that is expected for local development.)

## API overview

- `GET /api/games`: list available games
- `POST /api/new_game`: create game session
- `POST /api/move`: submit human move and run AI response
- `GET /api/state/{session_id}`: get latest state

## Notes

- One neural net handles all games using a game-id embedding.
- Each game provides its own state encoder and legal action mapping.
- The model is lightweight and intended as a baseline foundation you can scale.
