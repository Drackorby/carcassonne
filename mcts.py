import numpy as np


class MCTS:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.valid_actions = state.get_possible_actions()
        self.player = 1


def mcts_search(initial_state, num_simulations):
    root = MCTS(initial_state)

    for _ in range(num_simulations):
        node = root
        while node.valid_actions == [] and len(node.children) > 0:
            # Select the child with the highest UCB score
            node = select_child(node)

        if node.valid_actions:
            action = node.valid_actions.pop()  # Select a valid action
            next_state = node.state.perform_action(action)  # Simulate the action
            new_child = MCTSNode(next_state, parent=node)
            node.children.append(new_child)
            node = new_child

        # Simulate from the selected node
        result = simulate(node.state)

        # Backpropagate results
        backpropagate(node, result)

    # Select a valid action from the root node based on visits
    valid_actions = root.valid_actions
    best_action = None
    if valid_actions:
        best_action = np.random.choice(valid_actions)  # Select a random valid action
    return best_action


def select_child(node):
    # Implement a simple selection strategy, e.g., select the child with the highest visit count
    return max(node.children, key=lambda child: child.visits)


def simulate(state):
    # Implement a basic rollout policy, e.g., selecting random valid actions until the game ends
    while not state.is_terminated() and state.get_possible_actions():
        action = np.random.choice(state.get_possible_actions())
        state = state.step_copy(action)
    return state.get_reward(self.player)  # Return the reward when the game ends


def backpropagate(node, result):
    # Update the visit count of nodes along the path to the root
    while node is not None:
        node.visits += 1
        node = node.parent