"""Capacited Vehicles Routing Problem (CVRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver.routing_enums_pb2 import FirstSolutionStrategy as FSS
from ortools.constraint_solver.routing_enums_pb2 import LocalSearchMetaheuristic as LSM
import math
from collections import namedtuple
from itertools import product

"""Documentation on search strategies and local search options can be found here:
https://developers.google.com/optimization/routing/routing_options#first_sol_options"""
strategies = (FSS.AUTOMATIC,
              FSS.PATH_CHEAPEST_ARC,
              FSS.PATH_MOST_CONSTRAINED_ARC,
              FSS.EVALUATOR_STRATEGY,
              FSS.SAVINGS,
              FSS.SWEEP,
              FSS.CHRISTOFIDES,
              FSS.ALL_UNPERFORMED,
              FSS.BEST_INSERTION,
              FSS.PARALLEL_CHEAPEST_INSERTION,
              FSS.LOCAL_CHEAPEST_INSERTION,
              FSS.GLOBAL_CHEAPEST_ARC,
              FSS.LOCAL_CHEAPEST_ARC,
              FSS.FIRST_UNBOUND_MIN_VALUE)

ls_options = (LSM.AUTOMATIC,
              LSM.GREEDY_DESCENT,
              LSM.GUIDED_LOCAL_SEARCH,
              LSM.SIMULATED_ANNEALING,
              LSM.TABU_SEARCH)


Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])


def trivial_solution(vehicle_count, vehicle_capacity, customers: Customer):
    depot = customers[0]
    customer_count = len(customers)
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []

    remaining_customers = set(customers)
    remaining_customers.remove(depot)

    for v in range(0, vehicle_count):
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand * customer_count + customer.index)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    used.add(customer)
            remaining_customers -= used
    return vehicle_tours


def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


def get_distance_matrix(customers: Customer, big_number=1):
    count = len(customers)
    matrix = [[-1] * count for _ in range(count)]
    for i in range(count):
        for j in range(count):
            matrix[i][j] = int(length(customers[i], customers[j]) * big_number)
    return matrix


def create_data_model(vehicle_count, vehicle_capacity, customers: Customer, big_number=1):
    """Stores the data for the problem."""
    data = dict()
    data['distance_matrix'] = get_distance_matrix(customers, big_number)
    data['demands'] = [c.demand for c in customers]
    data['vehicle_capacities'] = [vehicle_capacity for _ in range(vehicle_count)]
    data['num_vehicles'] = vehicle_count
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, solution, customers: Customer, big_number=1, do_print=True):
    """Prints solution on console."""
    total_distance = 0
    total_load = 0
    tours = []
    for vehicle_id in range(data['num_vehicles']):
        tours.append([])
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:
                tours[vehicle_id].append(customers[node_index])
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index), route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance / big_number)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        if do_print:
            print(plan_output)
        total_distance += route_distance
        total_load += route_load
    if do_print:
        print('Total distance of all routes: {}m'.format(total_distance / big_number))
        print('Total load of all routes: {}'.format(total_load))
    return tours, total_distance / big_number


def solve_vrp(vehicle_count, vehicle_capacity, customers: Customer, big_number=100000, strategy=FSS.FIRST_UNBOUND_MIN_VALUE,
              ls_option=LSM.GUIDED_LOCAL_SEARCH):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model(vehicle_count, vehicle_capacity, customers, big_number)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    if 10 <= vehicle_count < 16:
        search_parameters.time_limit.seconds = 600
    else:
        search_parameters.time_limit.seconds = 90
    search_parameters.first_solution_strategy = strategy
    search_parameters.local_search_metaheuristic = ls_option

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    objective = -1
    if solution:
        tours, objective = print_solution(data, manager, routing, solution, customers, big_number, do_print=False)
    else:
        tours = trivial_solution(vehicle_count, vehicle_capacity, customers)
    return tours, objective



