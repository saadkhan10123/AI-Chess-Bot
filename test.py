import chess
import chess.engine
import math
import random

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.state.legal_moves))

    def best_child(self, exploration_weight=1.414):
        return max(
            self.children,
            key=lambda child: child.value / (child.visits + 1) +
                              exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1))
        )

class MonteCarloTreeSearch:
    def __init__(self, engine_path):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    def simulate(self, state):
        result = self.engine.play(state, chess.engine.Limit(time=0.1))
        if state.is_checkmate():
            return 1 if state.turn == chess.BLACK else -1
        return 0  # Assume a neutral outcome for other positions

    def expand(self, node):
        untried_moves = [
            move for move in node.state.legal_moves
            if move not in [child.state.peek() for child in node.children]
        ]
        move = random.choice(untried_moves)
        new_state = node.state.copy()
        new_state.push(move)
        child_node = Node(new_state, parent=node)
        node.children.append(child_node)
        return child_node

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def search(self, initial_state, iterations=1000):
        root = Node(initial_state)

        for _ in range(iterations):
            node = root

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            if not node.is_fully_expanded():
                node = self.expand(node)

            # Simulation
            reward = self.simulate(node.state)

            # Backpropagation
            self.backpropagate(node, reward)

        best_move = root.best_child(0).state.peek()
        return best_move

# Example usage
if __name__ == "__main__":
    engine_path = "path_to_your_chess_engine"  # Replace with your UCI-compatible chess engine path
    mcts = MonteCarloTreeSearch(engine_path)

    board = chess.Board()
    best_move = mcts.search(board, iterations=1000)
    print("Best Move:", best_move)
