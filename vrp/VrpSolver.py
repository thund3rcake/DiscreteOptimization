"""
Vehicle Routing Solver Module.
Provides local search in neighbourhoods:
    1. Shift part of one tour into another tour
    2. Swap two sub-tours
    3. Reverse sub-tour
    4. Divide two tours into 'head' and 'tail' and concatenate them in different ways
Search time can be limited by the time limit.
"""

import math
import itertools
from time import time
from collections import namedtuple

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])


class VrpSolver(object):
    def __init__(self, customers: Customer, vehicle_count, vehicle_capacity):
        self.CMP_THRESHOLD = 10 ** -6
        self.customers = customers
        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity
        self.obj = 0
        self.tours = self.greedy_init()
        return

    def __str__(self):
        obj = self.total_tour_dist()
        opt = 0
        if not self.is_valid_solution():
            raise ValueError("Solution not valid")
        output_str = "{:.2f} {}\n".format(obj, opt)
        for tour in self.tours:
            output_str += (' '.join(map(str, [c for c in tour])) + '\n')
        return output_str

    @staticmethod
    def dist(c1, c2):
        return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

    def tour_demand(self, tour):
        return sum([self.customers[i].demand for i in tour])

    def is_valid_tour(self, tour):
        """
        a tour is valid if:
        1. total demand doesn't not exceed vehicle cap
        2. start and end at customer[0], which is the base
        3. does not go back to base during the tour
        4. does not contain duplicate stop
        :param tour: list of int, tour of a single vehicle
        :return: True if tour is valid

        """
        is_valid = (self.tour_demand(tour) <= self.vehicle_capacity) and \
                   (tour[0] == 0) and (tour[-1] == 0) and \
                   (0 not in tour[1:-1]) and \
                   (len(set(tour[1:-1])) == len(tour[1:-1]))
        return is_valid

    def is_valid_solution(self):
        """
        a solution is valid if:
        1. every tour is valid
        :return: True if the whole solution is valid

        """
        return all([self.is_valid_tour(tour) for tour in self.tours])

    def single_tour_dist(self, tour):
        if not self.is_valid_tour(tour):
            return math.inf
        tour_dist = 0
        for i in range(1, len(tour)):
            customer_1 = self.customers[tour[i - 1]]
            customer_2 = self.customers[tour[i]]
            tour_dist += self.dist(customer_1, customer_2)
        return tour_dist

    def every_tour_dists(self):
        return [self.single_tour_dist(tour) for tour in self.tours]

    def total_tour_dist(self):
        dists = self.every_tour_dists()
        if math.inf in dists:
            raise ValueError("Invalid tour detected.")
        else:
            return sum(dists)

    def greedy_init(self):
        """
        Greedy initialization for the CVRP problem.
        assign customers to the vehicle until it gets out of the capacity
        """
        tours = []
        remaining_customers = set(self.customers[1:])
        for v in range(self.vehicle_count):
            remaining_cap = self.vehicle_capacity
            tours.append([])
            tours[-1].append(0)
            while remaining_customers and remaining_cap > min([c.demand for c in remaining_customers]):
                for customer in sorted(remaining_customers, reverse=True, key=lambda c: c.demand):
                    if customer.demand <= remaining_cap:
                        tours[-1].append(customer.index)
                        remaining_cap -= customer.demand
                        remaining_customers.remove(customer)
            tours[-1].append(0)
        if remaining_customers:
            raise ValueError("Greedy solution does not exist.")
        else:
            self.tours = tours
            self.obj = self.total_tour_dist()
            return self.tours

    def shift(self, i_from, start_from, end_from, i_to, j_to):
        """
        Shift a segment of tour into another tour
        2 possible ways:
        shift directly and reverse after shift
        :param i_from: index of tour shift from
        :param start_from: start index of segment
        :param end_from: end index of segment (inclusive)
        :param i_to: index of tour shift to
        :param j_to: location
        :return True if objective value has improved
        """
        tour_from_old = self.tours[i_from]
        tour_to_old = self.tours[i_to]
        improved = False

        seg_shift = tour_from_old[start_from: end_from + 1]

        tour_from_new = tour_from_old[:start_from] + tour_from_old[end_from + 1:]
        tour_to_new_1 = tour_to_old[:j_to] + seg_shift + tour_to_old[j_to:]
        tour_to_new_2 = tour_to_old[:j_to] + seg_shift[::-1] + tour_to_old[j_to:]

        dist_from_old = self.single_tour_dist(tour_from_old)
        dist_to_old = self.single_tour_dist(tour_to_old)

        dist_from_new = self.single_tour_dist(tour_from_new)

        dist_to_new_1 = self.single_tour_dist(tour_to_new_1)
        dist_to_new_2 = self.single_tour_dist(tour_to_new_2)

        obj_new_1 = self.obj - \
                    (dist_from_old + dist_to_old) + \
                    (dist_from_new + dist_to_new_1)

        obj_new_2 = self.obj - \
                    (dist_from_old + dist_to_old) + \
                    (dist_from_new + dist_to_new_2)

        if obj_new_1 < self.obj - self.CMP_THRESHOLD:
            self.tours[i_from] = tour_from_new
            self.tours[i_to] = tour_to_new_1
            # self.obj = obj_new_1
            self.obj = self.total_tour_dist()
            improved = True

        if obj_new_2 < self.obj - self.CMP_THRESHOLD:
            self.tours[i_from] = tour_from_new
            self.tours[i_to] = tour_to_new_2
            # self.obj = obj_new_2
            self.obj = self.total_tour_dist()
            improved = True

        return improved

    def swap(self, i_1, start_1, end_1, i_2, start_2, end_2):
        """
        Swap 2 segments from 2 tours
        4 possible ways:
            swap directly, reverse either segment, reverse both segments
        :param i_1: index of the first tour
        :param start_1: start index of the segment in the first tour
        :param end_1: end index of the segment in the first tour
        :param i_2: index of the second tour
        :param start_2: start index of the segment in the second tour
        :param end_2: end index of the segment in the second tour
        :return True if objective value has improved
        """
        tour_1_old = self.tours[i_1]
        tour_2_old = self.tours[i_2]
        improved = False

        seg_1 = tour_1_old[start_1: end_1 + 1]
        seg_2 = tour_2_old[start_2: end_2 + 1]

        # tour1 <- seg2, not reversed
        tour_1_new_1 = tour_1_old[: start_1] + seg_2 + tour_1_old[end_1 + 1:]
        # tour1 <- seg2, reversed
        tour_1_new_2 = tour_1_old[: start_1] + seg_2[::-1] + tour_1_old[end_1 + 1:]
        # tour2 <- seg1, not reversed
        tour_2_new_1 = tour_2_old[: start_2] + seg_1 + tour_2_old[end_2 + 1:]
        # tour2 <- seg1, reversed
        tour_2_new_2 = tour_2_old[: start_2] + seg_1[::-1] + tour_2_old[end_2 + 1:]

        # old tour lengths
        dist_1_old = self.single_tour_dist(tour_1_old)
        dist_2_old = self.single_tour_dist(tour_2_old)

        # new tour lengths
        dist_1_new_1 = self.single_tour_dist(tour_1_new_1)
        dist_1_new_2 = self.single_tour_dist(tour_1_new_2)
        dist_2_new_1 = self.single_tour_dist(tour_2_new_1)
        dist_2_new_2 = self.single_tour_dist(tour_2_new_2)

        new_obj_1 = self.obj - (dist_1_old + dist_2_old) + (dist_1_new_1 + dist_2_new_1)
        new_obj_2 = self.obj - (dist_1_old + dist_2_old) + (dist_1_new_1 + dist_2_new_2)
        new_obj_3 = self.obj - (dist_1_old + dist_2_old) + (dist_1_new_2 + dist_2_new_1)
        new_obj_4 = self.obj - (dist_1_old + dist_2_old) + (dist_1_new_2 + dist_2_new_2)

        if new_obj_1 < self.obj - self.CMP_THRESHOLD:
            self.tours[i_1] = tour_1_new_1
            self.tours[i_2] = tour_2_new_1
            # self.obj = new_obj_1
            self.obj = self.total_tour_dist()
            improved = True

        if new_obj_2 < self.obj - self.CMP_THRESHOLD:
            self.tours[i_1] = tour_1_new_1
            self.tours[i_2] = tour_2_new_2
            # self.obj = new_obj_2
            self.obj = self.total_tour_dist()
            improved = True

        if new_obj_3 < self.obj - self.CMP_THRESHOLD:
            self.tours[i_1] = tour_1_new_2
            self.tours[i_2] = tour_2_new_1
            # self.obj = new_obj_3
            self.obj = self.total_tour_dist()
            improved = True

        if new_obj_4 < self.obj - self.CMP_THRESHOLD:
            self.tours[i_1] = tour_1_new_2
            self.tours[i_2] = tour_2_new_2
            # self.obj = new_obj_4
            self.obj = self.total_tour_dist()
            improved = True

        return improved

    def reverse(self, i, start, end):
        """
        Reverse segment of a tour
        :param i: index of the tour
        :param start: start index of the segment in the tour
        :param end: end index of the segment in the tour
        :return True if objective value has improved
        """
        improved = False
        tour_old = self.tours[i]
        seg = tour_old[start: end + 1]
        tour_new = tour_old[:start] + seg[::-1] + tour_old[end + 1:]

        new_obj = self.obj - self.single_tour_dist(tour_old) + self.single_tour_dist(tour_new)
        if new_obj < self.obj - self.CMP_THRESHOLD:
            self.tours[i] = tour_new
            # self.obj = new_obj
            self.obj = self.total_tour_dist()
            improved = True
        return improved

    def ladder(self, i_1, i_2, j_1, j_2):
        """
        split two tours into head and tail respectively, and re-shuffle them
        2 possible ways
        :param i_1: first tour index
        :param i_2: second tour index
        :param j_1: index to divide the first tour
        :param j_2: index to divide the second tour
        """
        tour_1_old = self.tours[i_1]
        tour_2_old = self.tours[i_2]
        improved = False

        seg_1_head = tour_1_old[:j_1]
        seg_1_tail = tour_1_old[j_1:]
        seg_2_head = tour_2_old[:j_2]
        seg_2_tail = tour_2_old[j_2:]

        # head + tail
        tour_1_new_1 = seg_1_head + seg_2_tail
        tour_2_new_1 = seg_2_head + seg_1_tail

        # head + head(reversed) / tail(reversed) + tail
        tour_1_new_2 = seg_1_head + seg_2_head[::-1]
        tour_2_new_2 = seg_1_tail[::-1] + seg_2_tail

        # old tour lengths
        dist_1_old = self.single_tour_dist(tour_1_old)
        dist_2_old = self.single_tour_dist(tour_2_old)

        # new tour lengths
        dist_1_new_1 = self.single_tour_dist(tour_1_new_1)
        dist_1_new_2 = self.single_tour_dist(tour_1_new_2)
        dist_2_new_1 = self.single_tour_dist(tour_2_new_1)
        dist_2_new_2 = self.single_tour_dist(tour_2_new_2)

        new_obj_1 = self.obj - (dist_1_old + dist_2_old) + (dist_1_new_1 + dist_2_new_1)
        new_obj_2 = self.obj - (dist_1_old + dist_2_old) + (dist_1_new_2 + dist_2_new_2)

        if new_obj_1 < self.obj - self.CMP_THRESHOLD:
            self.tours[i_1] = tour_1_new_1
            self.tours[i_2] = tour_2_new_1
            # self.obj = new_obj_1
            self.obj = self.total_tour_dist()
            improved = True

        if new_obj_2 < self.obj - self.CMP_THRESHOLD:
            self.tours[i_1] = tour_1_new_2
            self.tours[i_2] = tour_2_new_2
            # self.obj = new_obj_2
            self.obj = self.total_tour_dist()
            improved = True

        return improved

    def swap_iteration(self):
        swap_improved = False
        for i_1, tour_1 in enumerate(self.tours):
            if swap_improved:
                break
            for start_1, end_1 in itertools.combinations(range(0, len(tour_1) - 1), 2):
                if swap_improved:
                    break
                for i_2, tour_2 in enumerate(self.tours):
                    if swap_improved:
                        break
                    if i_1 == i_2:
                        continue
                    for start_2, end_2 in itertools.combinations(range(0, len(tour_2) - 1), 2):
                        if self.swap(i_1, start_1, end_1, i_2, start_2, end_2):
                            swap_improved = True
                            break
        return swap_improved

    def shift_iteration(self):
        shift_improved = False
        for i_from, tour_from in enumerate(self.tours):
            if shift_improved:
                break
            for start_from, end_from in itertools.combinations(range(0, len(tour_from) - 1), 2):
                if shift_improved:
                    break
                for i_to, tour_to in enumerate(self.tours):
                    if shift_improved:
                        break
                    if i_from == i_to:
                        continue
                    for j_to in range(0, len(tour_to) - 1):
                        if self.shift(i_from, start_from, end_from, i_to, j_to):
                            shift_improved = True
                            break
        return shift_improved

    def reverse_iteration(self, debug=False):
        reverse_improved = False
        for i, tour in enumerate(self.tours):
            for start, end in itertools.combinations(range(0, len(tour) - 1), 2):
                if self.reverse(i, start, end):
                    reverse_improved = True
                    break
        return reverse_improved

    def ladder_iteration(self):
        ladder_improved = False
        for i_1, tour_1 in enumerate(self.tours):
            if ladder_improved:
                break
            for j_1 in range(1, len(tour_1) - 2):
                if ladder_improved:
                    break
                for i_2, tour_2 in enumerate(self.tours):
                    if i_1 == i_2:
                        continue
                    if ladder_improved:
                        break
                    for j_2 in range(1, len(tour_2) - 2):
                        if self.ladder(i_1, i_2, j_1, j_2):
                            ladder_improved = True
                            break
        return ladder_improved

    def solve(self, shift=True, swap=True, reverse=True, ladder=True, t_threshold=600):
        """
        Solve the CVRP  problem with local search.
        Provides local search in neighbourhoods:
            1. Shift part of one tour into another tour ('shift' method)
            2. Swap two sub-tours ('swap' method)
            3. Reverse sub-tour ('reverse' method)
            4. Divide two tours into 'head' and 'tail' and concatenate them in different ways ('ladder' method)
        :param shift: if True, applies local search on shifting
        :param swap: if True, applies local search on swapping
        :param reverse: if True, applies local search on reversing
        :param ladder: if True, applies local search on ladder method
        :param t_threshold: time limit in seconds for solver
        :return: tours for vehicles
        """

        improved = True
        t_start = time()

        while improved:
            if time() - t_start >= t_threshold:
                break
            shift_improved = False
            swap_improved = False
            reverse_improved = False
            ladder_improved = False
            self.obj = self.total_tour_dist()
            prev_obj = self.obj

            # try shift
            if shift:
                shift_improved = self.shift_iteration()

            # try swap
            if swap:
                swap_improved = self.swap_iteration()

            # try reverse
            if reverse:
                reverse_improved = self.reverse_iteration()

            # try ladder
            if ladder:
                ladder_improved = self.ladder_iteration()

            improved = shift_improved or swap_improved or reverse_improved or ladder_improved
        return self.tours

