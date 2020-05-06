from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
from collections import namedtuple
from numpy.random import randint
import numpy as np

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


class GSolver:
    def __init__(self, node_count, points, scale=1e3):
        self.node_count = node_count
        self.points = points
        self.scale = scale

    def get_distance_matrix(self):
        matrix = [[0] * self.node_count for _ in range(self.node_count)]
        for i in range(self.node_count):
            for j in range(self.node_count):
                matrix[i][j] = int(length(self.points[i], self.points[j]) * self.scale)
        return matrix

    def create_data_model(self, start_point=0):
        """Stores the data for the problem."""
        data = {'distance_matrix': self.get_distance_matrix(), 'num_vehicles': 1, 'depot': start_point}
        return data

    def print_solution(self, manager, routing, solution, flag=True):
        """Prints solution on console."""
        value = round(solution.ObjectiveValue() / self.scale, 2)
        if flag:
            print('Objective: {} miles'.format(value))
        index = routing.Start(0)
        plan_output = 'Route for vehicle 0:\n'
        route_distance = 0
        path = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            path.append(node)
            plan_output += ' {} ->'.format(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        if flag:
            print(plan_output)
        plan_output += 'Route distance: {}miles\n'.format(route_distance)
        return value, path

    def search_over_start(self):
        # Specify trivial solution
        optimal_path = range(self.node_count)
        min_value = length(self.points[optimal_path[-1]], self.points[optimal_path[0]])
        for index in range(0, self.node_count - 1):
            min_value += length(self.points[optimal_path[index]], self.points[optimal_path[index + 1]])

        # Specify solution depending on problem size
        if self.node_count <= 100:
            num_of_elements = self.node_count
        elif self.node_count <= 200:
            num_of_elements = int(np.sqrt(self.node_count))
        elif self.node_count <= 2000:
            num_of_elements = int(np.log(self.node_count))
        else:
            return min_value, optimal_path

        # Search for better solution
        start_positions = randint(0, self.node_count, num_of_elements)
        for start_pos in start_positions:
            value, path = self.solve(int(start_pos))
            if value < min_value:
                min_value = value
                optimal_path = path
            print("value = {}".format(value))
        return min_value, optimal_path

    def release(self):
        return self.solve(0)

    def solve(self, start_pos):
        """Solve the TSP."""
        data = self.create_data_model(start_pos)
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC

        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 600
        search_parameters.log_search = False

        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            value, path = self.print_solution(manager, routing, solution, False)
            return value, path
