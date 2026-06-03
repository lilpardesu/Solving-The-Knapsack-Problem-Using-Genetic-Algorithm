import pygad
import csv
import numpy as np
import argparse

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
capacity, weights, values, item_names = parse_input('./Input.csv')

# Calculates the fitness of a solution
# Returns total value if weight constraint is satisfied, otherwise 0
def fitness_func(ga_instance, solution, solution_idx):
    total_weight = np.sum(solution * weights)
    total_value = np.sum(solution * values)

    # Penalize solutions that exceed the knapsack capacity
    if total_weight > capacity:
        return 0

    return total_value

# Tracks generations without improvement
no_improvement_count = 0
best_fitness_so_far = 0

# Called after each generation to check for improvement
def on_generation(ga_instance):
    global no_improvement_count, best_fitness_so_far

    current_best = ga_instance.best_solution()[1]

    if current_best > best_fitness_so_far:
        # Solution improved, reset counter
        best_fitness_so_far = current_best
        no_improvement_count = 0
    else:
        # No improvement, increment counter
        no_improvement_count += 1
        print(f"No improvement for {no_improvement_count} generations...")

    # Stop if no improvement for 50 generations
    if no_improvement_count >= 50:
        print("Converged! Stopping early.")
        return "stop"

# Configure and initialize the Genetic Algorithm
ga_instance = pygad.GA(
    num_generations=1000,      # Maximum generations
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=100,           # Larger population for complex problems
    num_genes=len(weights),    # One gene per item
    gene_type=int,
    gene_space=[0, 1],         # Binary: 1 = selected, 0 = not selected
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=5,
    on_generation=on_generation  # Adaptive termination callback
)

# Run the Genetic Algorithm
ga_instance.run()

# Get the best solution found
solution, solution_fitness, solution_idx = ga_instance.best_solution()

# Print results
print("Selected items:")
for i, gene in enumerate(solution):
    if gene == 1:
        print(f"  - {item_names[i]} (weight: {weights[i]}, value: {values[i]})")

print(f"\nTotal weight: {int(np.sum(solution * weights))}")
print(f"Total value:  {int(solution_fitness)}")

