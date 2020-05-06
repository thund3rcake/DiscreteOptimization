from ortools.linear_solver import pywraplp
from collections import namedtuple
from ortools.sat.python import cp_model
import math
from datetime import datetime

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


class FacilitySolver:
    def __init__(self, facilities, customers):
        self.customers = customers
        self.facilities = facilities
        self.customer_count = len(customers)
        self.facility_count = len(facilities)
        self.used = None
        self.solution = None
        self.matrix = self.get_distance_matrix()

    @staticmethod
    def length(point1, point2):
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def __str__(self):
        # calculate the cost of the solution
        obj = sum([f.setup_cost * self.used[f.index] for f in self.facilities])
        for customer in self.customers:
            obj += self.length(customer.location, self.facilities[self.solution[customer.index]].location)

        # prepare the solution in the specified output format
        output_data = '%.2f' % obj + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, self.solution))
        return output_data

    def trivial_solution(self):
        solution = [-1] * self.customer_count
        capacity_remaining = [f.capacity for f in self.facilities]

        facility_index = 0
        for customer in self.customers:
            if capacity_remaining[facility_index] >= customer.demand:
                solution[customer.index] = facility_index
                capacity_remaining[facility_index] -= customer.demand
            else:
                facility_index += 1
                assert capacity_remaining[facility_index] >= customer.demand
                solution[customer.index] = facility_index
                capacity_remaining[facility_index] -= customer.demand

        used = [0] * self.facility_count
        for facility_index in solution:
            used[facility_index] = 1
        self.used = used
        self.solution = solution

    def get_distance_matrix(self):
        matrix = [[0] * self.customer_count for _ in range(self.facility_count)]
        for i in range(self.facility_count):
            for j in range(self.customer_count):
                matrix[i][j] = self.length(self.facilities[i].location, self.customers[j].location)
        return matrix

    def solve(self, print_info=True, n_sec=600):
        """
        Solve the Facility Location problem using MIP.
        :param print_info: if True, prints debugging information
        :param n_sec: time limit for the solver
        """

        # Create the mip solver with the CBC backend.
        solver = pywraplp.Solver('facility', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        solver.SetTimeLimit(n_sec * 1000)

        costs = [0] * self.facility_count
        for w in range(self.facility_count):
            costs[w] = self.facilities[w].setup_cost

        # Add variables
        x = [0] * self.facility_count
        for w in range(self.facility_count):
            x[w] = solver.BoolVar(f'x[{w}]')

        y = [[0] * self.customer_count for _ in range(self.facility_count)]
        for w in range(self.facility_count):
            for c in range(self.customer_count):
                y[w][c] = solver.BoolVar('y[{}][{}]'.format(w, c))

        if print_info:
            print('Number of variables =', solver.NumVariables())

        # If warehouse is not open then customer c can't belong to that warehouse
        for w in range(self.facility_count):
            for c in range(self.customer_count):
                solver.Add(y[w][c] <= x[w])

        # Every customer can belong only to one warehouse
        for c in range(self.customer_count):
            warehouses = [y[w][c] for w in range(self.facility_count)]
            solver.Add(solver.Sum(warehouses) == 1)

        # sum(demands) for every facility is less or equal then facility capacity
        for w in range(self.facility_count):
            demands = []
            for c in range(self.customer_count):
                demands.append(y[w][c] * self.customers[c].demand)
            solver.Add(solver.Sum(demands) <= self.facilities[w].capacity)

        if print_info:
            print('Number of constraints =', solver.NumConstraints())

        # Objective function
        objective = solver.Objective()
        for w in range(self.facility_count):
            objective.SetCoefficient(x[w], costs[w])
            for c in range(self.customer_count):
                objective.SetCoefficient(y[w][c], self.matrix[w][c])
        objective.SetMinimization()

        # Solve:
        status = solver.Solve()

        used = [-1] * self.facility_count
        solution = [-1] * self.customer_count
        if status == pywraplp.Solver.OPTIMAL and print_info:
            print('Found optimal solution.')
        elif status == pywraplp.Solver.FEASIBLE and print_info:
            print('Found feasible solution.')
        if (status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.NOT_SOLVED) and print_info:
            print('The MIP solver haven\'t found a feasible solution. Returning trivial solution.')
            used, solution = trivial_solution()
        elif print_info:
            print('Problem solved in %f milliseconds' % solver.wall_time())

        for w in range(self.facility_count):
            used[w] = int(x[w].solution_value())
        for c in range(self.customer_count):
            for w in range(self.facility_count):
                if y[w][c].solution_value() > 0:
                    solution[c] = w

        self.used = used
        self.solution = solution

