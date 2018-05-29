import numpy as np
import copy
import itertools


def horiz_next(loc):
    if loc[1] + 1 < WIDTH:
        return loc[0], loc[1] + 1
    else:
        return None


def horiz_prev(loc):
    if loc[1] - 1 >= 0:
        return loc[0], loc[1] - 1
    else:
        return None


def vert_next(loc):
    if loc[0] + 1 < HEIGHT:
        return loc[0] + 1, loc[1]
    else:
        return None


def vert_prev(loc):
    if loc[0] - 1 >= 0:
        return loc[0] - 1, loc[1]
    else:
        return None


def diag1_next(loc):
    if (loc[0] + 1 < WIDTH) and (loc[1] + 1 < HEIGHT):
        return loc[0] + 1, loc[1] + 1
    else:
        return None


def diag1_prev(loc):
    if (loc[0] - 1 >= 0) and (loc[1] - 1 >= 0):
        return loc[0] - 1, loc[1] - 1
    else:
        return None


def diag2_next(loc):
    if (loc[0] + 1 < WIDTH) and (loc[1] - 1 >= 0):
        return loc[0] + 1, loc[1] - 1
    else:
        return None


def diag2_prev(loc):
    if (loc[0] - 1 >= 0) and (loc[1] + 1 < HEIGHT):
        return loc[0] - 1, loc[1] + 1
    else:
        return None


WIDTH = 3
HEIGHT = 2
move_funcs = [
    horiz_next,
    horiz_prev,
    vert_next,
    vert_prev,
    diag1_next,
    diag1_prev,
    diag2_next,
    diag2_prev
]


def liberties_in_dir(loc, move_fun, board):
    current = loc
    path = list()
    while (current is not None) and board[current]:
        current = move_fun(current)
        path.append(current)
    return path[:-1]


class GameState:

    def __init__(self):
        self.open_cells = np.array([[True] * WIDTH] * HEIGHT)
        self.open_cells[HEIGHT - 1, WIDTH - 1] = False
        self.current_player = 0
        self.player_locations = [None, None]

    def actions(self):
        """ Return a list of legal actions for the active player

        You are free to choose any convention to represent actions,
        but one option is to represent actions by the (row, column)
        of the endpoint for the token. For example, if your token is
        in (0, 0), and your opponent is in (1, 0) then the legal
        actions could be encoded as (0, 1) and (0, 2).
        """
        return [(x, y) for x in range(HEIGHT) for y in range(WIDTH)
                if self.open_cells[x, y]]

    def player(self):
        """ Return the id of the active player

        Hint: return 0 for the first player, and 1 for the second player
        """
        return self.current_player

    def result(self, action):
        """ Return a new state that results from applying the given
        action in the current state

        Hint: Check out the deepcopy module--do NOT modify the
        objects internal state in place
        """
        assert action in self.actions()  # Check if the action is valid
        new_state = copy.deepcopy(self)
        new_state.open_cells[action] = False
        new_state.player_locations[self.player()] = action
        new_state.current_player = 0 if self.player() == 1 else 1
        return new_state

    def terminal_test(self):
        """ return True if the current state is terminal,
        and False otherwise

        Hint: an Isolation state is terminal if _either_
        player has no remaining liberties (even if the
        player is not active in the current state)
        """
        terminal = len(self.liberties(self.player_locations[0])) == 0
        terminal = terminal or len(self.liberties(self.player_locations[1])) == 0
        return terminal

    def liberties(self, loc):
        """ Return a list of all open cells in the
        neighborhood of the specified location.  The list
        should include all open spaces in a straight line
        along any row, column or diagonal from the current
        position. (Tokens CANNOT move through obstacles
        or blocked squares in queens Isolation.)

        Note: if loc is None, then return all empty cells
        on the board
        """
        if loc is None:
            return self.actions()
        liberties = [liberties_in_dir(loc, move_fun, self.open_cells)
                     for move_fun in move_funcs]
        return list(itertools.chain(*liberties))
