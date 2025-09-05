import random
from simpleai.search import (
    breadth_first,
    depth_first,
    uniform_cost,
    greedy,
    astar,
    SearchProblem
)
from simpleai.search.viewers import BaseViewer

GOAL = (
    (1, 2, 3),
    (8, 0, 4),
    (7, 6, 5)
)


class PuzzleProblem(SearchProblem):
    def actions(self, state):
        row, col = self.find_blank(state)
        moves = []
        if row > 0:
            moves.append('up')
        if row < 2:
            moves.append('down')
        if col > 0:
            moves.append('left')
        if col < 2:
            moves.append('right')
        return moves

    def result(self, state, action):
        row, col = self.find_blank(state)
        new_state = [list(r) for r in state]
        if action == 'up':
            new_row, new_col = row - 1, col
        elif action == 'down':
            new_row, new_col = row + 1, col
        elif action == 'left':
            new_row, new_col = row, col - 1
        elif action == 'right':
            new_row, new_col = row, col + 1
        new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
        return tuple(tuple(r) for r in new_state)

    def is_goal(self, state):
        return state == GOAL

    def cost(self, state1, action, state2):
        return 1

    def heuristic(self, state):
        dist = 0
        goal_pos = {val: (r, c) for r, row in enumerate(GOAL)
                    for c, val in enumerate(row)}
        for r in range(3):
            for c in range(3):
                val = state[r][c]
                if val != 0:
                    gr, gc = goal_pos[val]
                    dist += abs(r - gr) + abs(c - gc)
        return dist

    @staticmethod
    def find_blank(state):
        for r in range(3):
            for c in range(3):
                if state[r][c] == 0:
                    return r, c


def run_searches(initial_state):
    problem = PuzzleProblem(initial_state)
    algorithms = [
        ('BFS', breadth_first),
        ('DFS', depth_first),
        ('UCS', uniform_cost),
        ('Greedy (manhattan)', greedy),
        ('A* (manhattan)', astar),
    ]

    for name, algo in algorithms:
        viewer = BaseViewer()
        result = algo(problem, graph_search=True, viewer=viewer)

        # Track the path of 0 (blank tile)
        zero_positions = []
        for _, state in result.path():
            zero_positions.append(PuzzleProblem.find_blank(state))

        print(f"\n{name}")
        print("Moves sequence:", [
              action for action, state in result.path()[1:]])
        print("Zero position:", zero_positions)
        print(f"Moves: {len(result.path()) - 1}")
        final_solution(result)


def print_matrix(state: tuple) -> None:
    for row in state:
        print(" ".join(str(x) for x in row))
    print()


def final_solution(result) -> None:
    print("Final matrix:")
    print_matrix(result.path()[-1][1])


def random_state():
    nums = list(range(9))
    random.shuffle(nums)
    return tuple(tuple(nums[i:i+3]) for i in range(0, 9, 3))


if __name__ == "__main__":
    initial = (
        (2, 8, 3),
        (1, 6, 4),
        (7, 0, 5)
    )
    # initial = random_state()
    print("Inital:")
    print_matrix(initial)
    run_searches(initial)
