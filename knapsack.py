import pygad
import csv
import numpy as np


# Parses the input CSV file and returns the knapsack capacity,
# item names, weights, and values
def parse_input(file_path):
    weights = []
    values = []
    capacity = None
    item_names = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'capacity':
                # First row contains the knapsack capacity
                capacity = int(row[1])
            elif row[0] == 'item' and row[1] == 'weight':
                # Skip the header row
                continue
            else:
                # Each remaining row is an item
                item_names.append(row[0])
                weights.append(int(row[1]))
                values.append(int(row[2]))

    return capacity, weights, values, item_names

# Parse the input file
capacity, weights, values, item_names = parse_input('input.csv')

# Calculates the fitness of a solution
# Returns total value if weight constraint is satisfied, otherwise 0
def fitness_func(ga_instance, solution, solution_idx):
    total_weight = np.sum(solution * weights)
    total_value = np.sum(solution * values)

    # Penalize solutions that exceed the knapsack capacity
    if total_weight > capacity:
        return 0

    return total_value
