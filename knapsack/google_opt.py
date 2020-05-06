from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver
from collections import namedtuple

Item = namedtuple("Item", ['index', 'value', 'weight'])


def solve(capacity, items):
    """Runs a KnapsackSolver from Google Optimization Tools to solve a problem.
    :param capacity: int
        Capacity of knapsack
    :param items: list<Item>
        List of items, contains its values and
    :return: int, list
        value - total value of a knapsack
        taken - list containing zeros and ones which encode which items to take
    """
    # Prepare input data
    values = []
    weights = [[]]
    capacity = [capacity]
    for item in items:
        weights[0].append(item.weight)
        values.append(item.value)

    num_items = len(items)

    # Define the solving method
    if num_items < 64:
        method = KnapsackSolver.KNAPSACK_64ITEMS_SOLVER
    else:
        method = KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER

    # Define the solver
    solver = KnapsackSolver(method, 'KnapsackSolver')
    solver.Init(values, weights, capacity)
    result_value = solver.Solve()

    # Prepare the answer
    taken = [0] * num_items
    for i in range(num_items):
        if solver.BestSolutionContains(i):
            taken[i] = 1

    return result_value, taken
