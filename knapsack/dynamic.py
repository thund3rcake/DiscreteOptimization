from time import time
from collections import namedtuple, OrderedDict
import operator

Item = namedtuple("Item", ['index', 'value', 'weight'])


class DynamicSolver:
    def __init__(self, capacity, items: Item):
        self.capacity = capacity
        self.items = items
        self.taken = None
        self.value = None

    def __str__(self):
        if self.value is None:  # greedy solution
            self.greedy()

        output_data = str(self.value) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, self.taken))
        return output_data

    def solve(self, time_limit=600):
        """
        Method of solving knapsack problem using dynamic programming.
        :param time_limit: time limit in seconds for solver
        """
        item_count = len(self.items)
        # Initialize the table
        table = [[0] * (item_count + 1) for _ in range(self.capacity + 1)]

        start = time()
        # Calculate values
        for item in self.items:
            if time() - start >= time_limit:
                print("Time limit exceeded.")
                return
            j = item.index + 1
            for k in range(self.capacity + 1):
                if item.weight <= k:
                    table[k][j] = max(table[k][j - 1], item.value + table[k - item.weight][j - 1])
                else:
                    table[k][j] = table[k][j - 1]

        value = int(table[self.capacity][item_count])
        taken = [0] * (item_count + 1)

        # Do backtrace
        k = self.capacity
        current = table[k][item_count]
        for item in reversed(self.items):
            j = item.index + 1
            if current != table[k][j - 1]:
                taken[j] = 1
                k -= item.weight
                current = table[k][j - 1]
	
        print('Problem solved in %.3f seconds' % (time() - start))
        taken.remove(taken[0])

        self.value = value
        self.taken = taken

    def greedy(self):
        """
	Order items by its value per kg and makes a greedy decision
        """
        wpk = dict()
        for item in self.items:
            wpk[item.index] = item.value / item.weight

        wpk = OrderedDict(sorted(wpk.items(), key=operator.itemgetter(1), reverse=True))

        value = 0
        weight = 0
        taken = [0] * len(self.items)

        for index in wpk.keys():
            if weight + self.items[index].weight <= self.capacity:
                taken[index] = 1
                weight += self.items[index].weight
                value += self.items[index].value

        self.taken = taken
        self.value = value
