from ortools.sat.python import cp_model
from datetime import datetime
from random import randint
from ortools.linear_solver import pywraplp


class GraphColouringSolver:
    def __init__(self, node_count, edges):
        self.node_count = node_count
        self.edges = edges
        self.upper_bound = self.calc_upper_bound()
        self.obj = node_count
        self.colours = list(range(node_count))

    def __str__(self):
        output_data = str(self.obj) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, self.colours))
        return output_data

    def calc_upper_bound(self):
        """
        Calculate the upper bound value of the target function.
        Let A = min(node_count, max_degree, edges_count)
        We always can colour the graph in A colours.
        :return: upper bound -- integer value
        """
        edges_count = len(self.edges)
        degrees = [0] * self.node_count
        for edge in self.edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1

        upper_bound = min(self.node_count, edges_count, max(degrees))
        return upper_bound

    def cp_colouring(self, time_limit=600):
        """
        Constraint programming solution.
        :param time_limit: time limit for solver in seconds
        :return: int, list
          value -- sum of the colours, colours[i] maps to the colour of the i-th node
        """
        model = cp_model.CpModel()

        # Add optimization variables
        colours = []
        for i in range(self.node_count):
            var = model.NewIntVar(0, self.upper_bound, f"colours{i}")
            colours.append(var)

        # Specify constraints
        for edge in self.edges:
            u = edge[0]
            v = edge[1]
            model.Add(colours[u] != colours[v])

        # Fix the first edge
        edge = self.edges[randint(0, len(self.edges) - 1)]
        u = edge[0]
        v = edge[1]
        model.Add(colours[u] == 0)
        model.Add(colours[v] == 1)

        # Target is special variable which will be the objective function: target = max(colours)
        target = model.NewIntVar(0, self.upper_bound, "target")
        model.AddMaxEquality(target, colours)
        model.Minimize(target)

        # Run the solver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        start = datetime.now()
        status = solver.Solve(model)
        finish = datetime.now()

        if status == cp_model.OPTIMAL:
            print("OPTIMAL. Solved. Time: {}".format(finish - start))
            for i in range(self.node_count):
                self.colours[i] = solver.Value(colours[i])

        if status == cp_model.FEASIBLE:
            print("FEASIBLE. Solved. Time: {}".format(finish - start))
            for i in range(self.node_count):
                self.colours[i] = solver.Value(colours[i])

        self.obj = max(self.colours) + 1


