from sample_players import DataPlayer
import random
import math
from functools import partial
import logging

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Memory inefficient, but let's do it by now...
        if parent is not None:
            self.parents = {(parent, action)}
        else:
            self.parents = set()
        self.children = dict()
        self.Q = 0
        self.N = 0

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.state == other.state)

    def is_fully_expanded(self):
        return len(self.unexplored_actions()) == 0

    def unexplored_actions(self):
        return list(set(self.state.actions()) - set(self.children.keys()))

    def add_child(self, child_node, action):
        self.children[action] = child_node

    def __str__(self):
        return '\n'.join(['{:25}: {}\n'.format(key, value) + '_' * 100 + '\n'
                          for key, value in self.__dict__.items()])


class Tree:
    def __init__(self, root):
        self.root = root
        self.nodes = {root}

    def add_retrieve(self, node):
        """If the node exists, it retrieves it. Else, it adds it to the tree."""
        self.nodes |= {node}
        return next(item for item in self.nodes if item == node)

    def expand(self, node):
        # Adds the new nodes to the tree, or modifies it adding a new parent,
        # if the node already existed from another parent
        a = random.choice(node.unexplored_actions())
        new_node = Node(node.state.result(a), node, a)
        new_node = self.add_retrieve(new_node)
        new_node.parents.add((node, a))
        node.add_child(new_node, a)
        return new_node

    def __str__(self):
        return ('\n' + '.' * 100 + '\n').join(str(node) for node in self.nodes)


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

    @staticmethod
    def new_node(state, parent=None, from_action=None, N=0, Q=0, children=None):
        node = {'state': state,
                'parent': parent,
                'from_action': from_action,
                'N': N,
                'Q': Q
                }
        if children is None:
            node['children'] = list()
        else:
            node['children'] = children
        if parent is not None:
            parent['children'].append(node)
        return node

    @staticmethod
    def ucb(node, c=1):
        if node['N'] == 0:
            if c != 0:
                return float('inf')
            else:
                return 0
        return (node['Q'] / node['N']) + c * math.sqrt((2 * math.log(node['N'])) / node['N'])

    @staticmethod
    def best_node(node_list, c=1):
        return max(node_list, key=partial(CustomPlayer.ucb, c=c))

    @staticmethod
    def best_child(node, c=1):
        return CustomPlayer.best_node(node['children'], c)

    @staticmethod
    def expand(node):
        a = random.choice(CustomPlayer.unexplored_actions(node))
        child = CustomPlayer.new_node(state=node['state'].result(a),
                                      parent=node,
                                      from_action=a)
        node['children'].append(child)
        return child

    @staticmethod
    def unexplored_actions(node):
        return list(set(node['state'].actions()) -
                    set(child['from_action'] for child in node['children']))

    @staticmethod
    def is_fully_expanded(node):
        return len(CustomPlayer.unexplored_actions(node)) == 0

    @staticmethod
    def tree_policy(node, nodes=None):
        if nodes is None:
            nodes = dict()
        temp_node = node
        while not temp_node['state'].terminal_test():
            if CustomPlayer.is_fully_expanded(temp_node):
                temp_node = CustomPlayer.best_child(temp_node)
            else:
                child = CustomPlayer.expand(temp_node)
                if child['state'] not in nodes.keys():
                    nodes[child['state']] = [child]
                else:
                    nodes[child['state']].append(child)
                return child
        return temp_node

    @staticmethod
    def backup_negamax(node, reward):
        temp_node = node
        temp_node['N'] += 1
        temp_node['Q'] += reward
        while temp_node['parent'] is not None:
            temp_node = temp_node['parent']
            temp_node['N'] += 1
            temp_node['Q'] += reward

    @staticmethod
    def get_root_node(state, nodes=None):
        """
        nodes = {
            state1: [node1_1, node1_2],
            state2: [node2_1],
            state3: [node3_1, node3_2, node3_3]
        }
        """
        if nodes is None:
            nodes = dict()
        if state not in nodes.keys():
            node = CustomPlayer.new_node(state)
            nodes[state] = [node]
            return node, nodes
        else:
            return CustomPlayer.best_node(nodes[state]), nodes

    def uct_search(self, root_n, nodes=None):
        """
        nodes contains a dictionary of state:node with the nodes that have
        already been visited.
        """
        edge_node = CustomPlayer.tree_policy(root_n, nodes)
        reward = self.default_policy(edge_node['state'])
        CustomPlayer.backup_negamax(edge_node, reward)
        return CustomPlayer.best_child(root_n, 0)['from_action'], nodes  # Greedy, so c = 0

    def default_policy(self, state):
        temp_state = state
        while not temp_state.terminal_test():
            a = random.choice(temp_state.actions())
            temp_state = temp_state.result(a)

        return temp_state.utility(self.player_id)

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

        #if self.context is None:
        #    self.context = dict()
        #nodes = self.context
        nodes = dict()
        root_n, nodes = CustomPlayer.get_root_node(state, nodes)
        root_n['parent'] = None  # Set this one as the new root
        root_n['from_action'] = None  # Just for completeness, not used by now

        while True:
            action, nodes = self.uct_search(root_n, nodes)
            # print('\naction {}, nodes {}'.format(action, len(nodes)))
            # print([str(state) for state in nodes.keys()])
            self.queue.put(action)
            #self.context = nodes
