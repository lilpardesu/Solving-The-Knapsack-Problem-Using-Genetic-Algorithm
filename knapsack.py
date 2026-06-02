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
