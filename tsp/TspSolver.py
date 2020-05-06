import math
from itertools import combinations
import itertools
from time import time
import random
from collections import namedtuple
import math
Point = namedtuple("Point", ['x', 'y'])


class TspSolver(object):
    def __init__(self, points):
        self.CMP_THRESHOLD = 10 ** -6
        self.points = points
        self.node_count = len(points)
        self.cycle = list(range(len(points))) + [0]
        self.obj = self.cycle_length()

    def __str__(self):
        obj = self.cycle_length()
        opt = 0
        if not self.is_valid_solution():
            raise ValueError("Solution is not valid")
        output_str = "{:.2f} {}\n".format(obj, opt)
        output_str += ' '.join(map(str, self.cycle[:-1]))
        return output_str

    @staticmethod
    def point_dist(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def is_valid_solution(self):
        return len(set(self.cycle[:-1])) == len(self.points) == len(self.cycle[:-1])

    def edge_length(self, v1, v2):
        p1 = self.points[v1]
        p2 = self.points[v2]
        return self.point_dist(p1, p2)

    def random_init(self):
        """
        Randomly initialize the TSP cycle.
        """
        tmp = list(range(0, self.node_count))
        random.shuffle(tmp)
        self.cycle = tmp + [tmp[0]]

    def cycle_length(self):
        return sum(self.edge_length(v1, v2) for v1, v2 in zip(self.cycle[:-1], self.cycle[1:]))

    def greedy(self):
        """
        Greedy algorithm to solve TSP.
        On every iteration, choose the nearest neighbour and attach it to the path.
        """
        cycle = [0]
        candidates = set(self.cycle[1:-1])
        while candidates:
            curr_point = cycle[-1]
            nearest_neighbor = None
            nearest_dist = math.inf
            for neighbor in candidates:
                neighbor_dist = int(self.edge_length(curr_point, neighbor) * 1e3)
                if neighbor_dist < nearest_dist:
                    nearest_neighbor = neighbor
                    nearest_dist = neighbor_dist
            cycle.append(nearest_neighbor)
            candidates.remove(nearest_neighbor)
        cycle.append(0)
        self.cycle = cycle
        self.obj = self.cycle_length()
        return cycle, self.obj


class TwoOptSolver(TspSolver):
    def swap(self, start, end):
        """
        2-opt iteration: remove edges (start - 1, start) and (end, end + 1) and make new cross edges
        :param start: start point
        :param end: end point
        :return: True if objective value was improved
        """
        improved = False
        new_cycle = self.cycle[:start] + self.cycle[start:end + 1][::-1] + self.cycle[end + 1:]
        new_obj = self.obj - \
                  (self.edge_length(self.cycle[start - 1], self.cycle[start]) +
                   self.edge_length(self.cycle[end], self.cycle[(end + 1)])) + \
                  (self.edge_length(new_cycle[start - 1], new_cycle[start]) +
                   self.edge_length(new_cycle[end], new_cycle[(end + 1)]))
        if self.obj - new_obj > self.CMP_THRESHOLD:
            self.cycle = new_cycle
            self.obj = new_obj
            improved = True
        return improved

    def solve(self, t_threshold=600):
        """
        Solve the TSP using 2-opt algorithm.
        :param t_threshold: time limit for the solver
        """
        # greedy approximation + initialization
        greedy_cycle, greedy_obj = self.greedy()
        improved = True
        t = time()
        while improved:
            if t_threshold and time() - t >= t_threshold:
                break
            improved = False
            workspace = itertools.combinations(range(1, self.node_count), 2)
            for start, end in workspace:
                if self.swap(start, end):
                    improved = True
                    break
        if self.obj > greedy_obj:
            self.cycle = greedy_cycle
            self.obj = greedy_obj
