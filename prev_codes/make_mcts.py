import random


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0


    def mcts(root, iterations):
        for _ in range(iterations):
            node = root
            while not node.state.is_terminal() and not node.children:
                node = select(node)

            if not node.state.is_terminal():
                node = expand(node)

            result = simulate(node.state)
            backpropagate(node, result)

        return best_child(root)
    def select(node):
        return max(node.children, key=ucb_score)

    def expand(node):
        actions = node.state.get_possible_actions()
        action = random.choice(actions)
        next_state = node.state.perform_action(action)
        child = Node(next_state, parent=node)
        node.children.append(child)
        return child

    def simulate(state):
        while not state.is_terminal():
            actions = state.get_possible_actions()
            action = random.choice(actions)
            state = state.perform_action(action)
        return state.get_score()

    def backpropagate(node, result):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent

    def best_child(node):
        return max(node.children, key=lambda child: child.value / child.visits)
