""" Module for generating the openings book. """
from isolation.isolation import Isolation
import datetime as dt
import pickle
import re
import os
from functools import reduce
import isolation.isolation as iso
import subprocess

W, H = 11, 9

cardinal_sym_h = {
    'N': 'N',
    'S': 'S',
    'E': 'W',
    'W': 'E'
}

cardinal_sym_v = {
    'N': 'S',
    'S': 'N',
    'E': 'E',
    'W': 'W'
}

cardinal_sym_c = {
    'N': 'S',
    'S': 'N',
    'E': 'W',
    'W': 'E'
}

actions_dict = {
    'N': iso.N,
    'S': iso.S,
    'E': iso.E,
    'W': iso.W,
}


def get_full_states(state, depth, s_set=None):
    if s_set is None:
        s_set = set()
    else:
        s_set.add(state)
    for action in state.actions():
        if depth > 0:
            s_set |= get_full_states(state.result(action), depth - 1, s_set)
    return s_set


def get_full_tree(state, depth, s_a_set=None):
    if s_a_set is None:
        s_a_set = set()
    for action in state.actions():
        s_a_set.add((state, action))
        if depth > 1:
            s_a_set |= get_full_tree(state.result(action), depth - 1, s_a_set)
    return s_a_set


def get_empty_book(state, depth):
    tree = get_full_tree(state, depth)
    return {key: [0, 0] for key in tree}


def process_game_history(state,
                         game_history,
                         book,
                         winner_id,
                         active_player=0,
                         depth=4):
    """ Given an initial state, and a list of actions, this function iterates
    through the resulting states of the actions and updates count of wins in
    the state/action book"""
    opening_moves = 2
    game_value = 2 * (active_player == winner_id) - 1
    curr_state = state  # It is a named tuple, so I think it is immutable. No need to copy.
    for num_action, action in enumerate(game_history):
        if (curr_state, action) in book.keys():
            book[(curr_state, action)][0] += game_value
            book[(curr_state, action)][1] += 1
            if curr_state.ply_count <= opening_moves:
                sym_pair = sym_sa((curr_state, action),
                                  loc_sym=h_symmetry,
                                  cardinal_sym=cardinal_sym_h)
                if sym_pair[1] in sym_pair[0].actions():
                    book[sym_pair][0] += game_value
                    book[sym_pair][1] += 1
                sym_pair = sym_sa((curr_state, action),
                                  loc_sym=v_symmetry,
                                  cardinal_sym=cardinal_sym_v)
                if sym_pair[1] in sym_pair[0].actions():
                    book[sym_pair][0] += game_value
                    book[sym_pair][1] += 1
                sym_pair = sym_sa((curr_state, action),
                                  loc_sym=c_symmetry,
                                  cardinal_sym=cardinal_sym_c)
                if sym_pair[1] in sym_pair[0].actions():
                    book[sym_pair][0] += game_value
                    book[sym_pair][1] += 1
        curr_state = curr_state.result(action)
        active_player = 1 - active_player
        game_value = 2 * (active_player == winner_id) - 1
        # Break on depth equal to book
        if num_action >= depth - 1:
            break


def save_book(book):
    timestamp = dt.datetime.now()
    filename = 'book' + timestamp.__str__().replace(' ', '_').replace(':','S') + '.pkl'
    print('Saving: {}'.format(filename))
    with open(filename, 'wb') as file:
        pickle.dump(book, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_book_list():
    pattern = 'book(.*)\.pkl'
    book_list = [
        (re.match(pattern, filename).group(0),
         dt.datetime.strptime(re.match(pattern, filename).group(1),
                              '%Y-%m-%d_%HS%MS%S.%f'))
        for filename in os.listdir('.')
        if re.match(pattern, filename) is not None
    ]
    return book_list


def get_latest_book_name(book_list):
    if len(book_list) == 0:
        return None
    return max(book_list, key=lambda x: x[1])[0]


def load_latest_book(depth=4):
    filename = get_latest_book_name(get_book_list())
    if filename is None:
        return get_empty_book(Isolation(), depth)
    with open(filename, 'rb') as file:
        latest_book = pickle.load(file)
    return latest_book


def remove_old_books():
    book_list = get_book_list()
    titles = [book for book, date in book_list]
    latest_name = get_latest_book_name(book_list)
    titles.remove(latest_name)
    for title in titles:
        command = 'rm {}'.format(title)
        print(command)
        subprocess.call(command, shell=True)


def show_board(board, W, H):
    """
    This function shows a regular board (not a bitboard).
    Borders are not counted in W, so the board has W + 2.
    """
    for i in range(H):
        print('  '.join([str(v) for v in board[i*(W+2):(i+1)*(W+2)]]))


def h_symmetry(loc):
    if loc is None:
        return None
    row = loc // (W + 2)
    center = W + (row - 1) * (W + 2) + (W + 2) // 2 + 1 if row != 0 else W // 2
    return 2 * center - loc


def v_symmetry(loc):
    if loc is None:
        return None
    col = loc % (W + 2)
    center = (H // 2) * (W + 2) + col
    return 2 * center - loc


def c_symmetry(loc):
    if loc is None:
        return None
    center = (H // 2) * (W + 2) + W // 2
    return 2 * center - loc


def q1_symmetric(state, W=11, H=9):
    # The first quadrant
    mod = (W + 2) // 2 + 1
    q1 = [(W + 2) * (loc // mod) + loc % mod for loc in range(mod * (H // 2 + 1))]

    if state.locs[0] is None:
        return state
    if state.locs[0] in q1:
        return state
    # Horizontal
    if h_symmetry(state.locs[0]) in q1:
        return Isolation(board=state.board,
                         ply_count=state.ply_count,
                         locs=tuple(map(h_symmetry, state.locs)))
    # Vertical
    if v_symmetry(state.locs[0]) in q1:
        return Isolation(board=state.board,
                         ply_count=state.ply_count,
                         locs=tuple(map(v_symmetry, state.locs)))
    # Central
    if c_symmetry(state.locs[0]) in q1:
        return Isolation(board=state.board,
                         ply_count=state.ply_count,
                         locs=tuple(map(c_symmetry, state.locs)))


def value2action(value):
    return next(a for a in iso.Action if a.value == value)


def action_symmetric(action, cardinal_sym):
    code = str(action).split('.')[1]
    action_value = reduce(lambda x,y: x + y,
                          map(lambda x: actions_dict[cardinal_sym[x]], code))
    return value2action(action_value)


def sym_sa(s_a, loc_sym, cardinal_sym):
    """
    Symmetry for a (state, action) pair.
    Don't use if this state is from move 3 or more.
    loc_sym is a function that transforms a location.
    cardinal_sym is a dictionary that maps cardinal points in the symmetry.
    """
    state = s_a[0]
    action = s_a[1]

    new_board = iso._BLANK_BOARD
    new_locs = tuple(map(loc_sym, state.locs))
    if new_locs[0] is not None:
        new_board = new_board ^ (1 << (new_locs[0]))
    if new_locs[1] is not None:
        new_board = new_board ^ (1 << (new_locs[1]))
    new_state = Isolation(board=new_board,
                          ply_count=state.ply_count,
                          locs=new_locs)
    if isinstance(action, iso.Action):
        new_action = action_symmetric(action, cardinal_sym)
    else:
        new_action = loc_sym(action)
    return new_state, new_action
