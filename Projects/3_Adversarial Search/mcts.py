import random
import math
from functools import partial


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

    def expand(self, node):
        # Adds the new nodes to the tree, or modifies it adding a new parent,
        # if the node already existed from another parent
        a = random.choice(node.unexplored_actions())
        new_node = Node(node.state.result(a), node, a)
        self.nodes |= {new_node}
        new_node = next(item for item in self.nodes if item == new_node)
        new_node.parents.add((node, a))
        node.add_child(new_node, a)
        return new_node

    def __str__(self):
        return ('\n' + '.' * 100 + '\n').join(str(node) for node in self.nodes)


def ucb(node, c=1):
    if node.N == 0:
        return float('inf')
    return (node.Q/node.N) + c * math.sqrt((2 * math.log(node.N)) / node.N)


def best_child(node, c=1):
    return max(list(node.children.values()), key=partial(ucb, c=c))


def tree_policy(node, tree):
    temp_node = node
    while not temp_node.state.terminal_test():
        if temp_node.is_fully_expanded():
            temp_node = best_child(temp_node)
        else:
            return tree.expand(temp_node)
    return temp_node


def default_random_policy(state, player_id):
    temp_state = state
    while not temp_state.terminal_test():
        a = random.choice(temp_state.actions())
        temp_state = temp_state.result(a)

    return temp_state.utility(player_id)


def backup_negamax(node, reward):
    """ This is a version of backup for multiple parents."""
    temp_node = node
    temp_node.N += 1
    temp_node.Q += reward
    for parent, _ in temp_node.parents:
        backup_negamax(parent, -reward)


def uct_search(root_state,
               tree=None,
               default_policy=default_random_policy,
               player_id=1):
    """
    If an existing tree is passed with a root_state different from
    root state, the search will start in root_state, but will affect the whole
    tree (upper nodes included).
    The root_state HAS to be unique parent of all its children.
    """
    root_n = Node(root_state)
    if tree is None:
        tree = Tree(root_n)
    edge_node = tree_policy(root_n, tree)
    reward = default_policy(edge_node.state, player_id)
    backup_negamax(edge_node, reward)
    best_root_child = best_child(root_n, 0)  # Greedy, so c = 0
    best_root_action_l = [a for p, a in best_root_child.parents if p == root_n]
    return best_root_action_l[0], tree
