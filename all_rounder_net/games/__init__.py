from .base import BaseGame, GameState
from .nim import NimGame
from .tictactoe import TicTacToeGame
from .dots_and_boxes import DotsAndBoxesGame
from .gardner_chess import GardnerChessGame


def build_games() -> dict[str, BaseGame]:
    return {
        "nim": NimGame(),
        "tictactoe": TicTacToeGame(),
        "dots_and_boxes": DotsAndBoxesGame(),
        "gardner_chess": GardnerChessGame(),
    }
