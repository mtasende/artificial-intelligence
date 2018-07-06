
from sample_players import DataPlayer
import random


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
        alpha = -float('inf')
        beta = float('inf')
        best_score = -float('inf')
        first_move = True
        best_move = None
        for a in state.actions():
            v = self.min_value(state.result(a), alpha, beta, depth - 1)
            if first_move:
                first_move = False
                best_move = a
            if v > best_score:
                best_score = v
                best_move = a
            alpha = max(v, alpha)
        return best_move

    def min_value(self, state, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)

        v = float('inf')
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

        v = -float('inf')
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
        self.queue.put(random.choice(state.actions()))

        book_depth = 2
        random_book = False
        # Play by the book...
        if state.ply_count < book_depth:
            if random_book:
                return  # One random action is always in the queue
            if state in self.data.keys():
                book_action = max(self.data[state].items(), key=lambda x: x[1])[0]
                self.queue.put(book_action)
        else:
            # ...or iterative deepening
            depth = 1
            while True:
                self.queue.put(self.alpha_beta_search(state, depth))
                depth += 1
