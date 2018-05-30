
from sample_players import DataPlayer
import random
from copy import deepcopy


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """

    def my_moves(self, state):
        return len(state.liberties(state.locs[self.player_id]))

    def opponent_moves(self, state):
        return len(state.liberties(state.locs[1 - self.player_id]))

    def alpha_beta_search(self, state, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.

        Assumprtions:  depth > 0,  the state is not terminal.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in state.actions():
            v = self.min_value(state.result(a), alpha, beta, depth - 1)
            if v > best_score:
                best_score = v
                best_move = a
            alpha = max(v, alpha)
        print('Turn {}) With depth {}, I will play: {}'.format(state.ply_count, depth, best_move))
        return best_move

    def min_value(self, state, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)

        if depth <= 0:
            return self.my_moves(state) - self.opponent_moves(state)

        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_value(state.result(a), alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(v, beta)
        return v

    def max_value(self, state, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)

        if depth <= 0:
            return self.my_moves(state) - self.opponent_moves(state)

        v = float("-inf")
        for a in state.actions():
            v = max(v, self.min_value(state.result(a), alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(v, alpha)
        return v

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        print('get_action was called with {} possible actions.'.format(len(state.actions())))
        self.queue.put(random.choice(state.actions()))

        # if self.context is None:
        #    self.context = dict()
        # print('Previous turns max depths: {}'.format(self.context))

        # Iterative deepening
        depth = 1
        while True:
            # I could deepcopy the state (safer) but it would take more time to run
            self.queue.put(self.alpha_beta_search(state, depth))
            # self.context[state.ply_count] = depth  # Save the last depth for each turn
            depth += 1