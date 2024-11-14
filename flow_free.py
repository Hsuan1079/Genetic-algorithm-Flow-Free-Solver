import random

# Define grid size and color requirements
grid_size = 5
colors = {'Y': 7, 'B': 3, 'R': 3, 'G': 2}
target_grid = [
    ['Y', 'N', 'B', 'R', 'G'],
    ['N', 'N', 'N', 'N', 'N'],
    ['N', 'N', 'N', 'G', 'N'],
    ['N', 'B', 'N', 'N', 'R'],
    ['N', 'N', 'N', 'N', 'Y']
]
target_points = {
    'Y': [(0, 0), (4, 4)],
    'B': [(0, 2), (3, 1)],
    'R': [(0, 3), (3, 4)],
    'G': [(0, 4), (2, 3)],
}

# Generate random initial population
def generate_individual(grid, colors):
    individual = [row[:] for row in grid]
    empty_positions = [(i, j) for i in range(grid_size) for j in range(grid_size) if grid[i][j] == 'N']
    
    for color, min_count in colors.items():
        positions = random.sample(empty_positions, min_count)
        for pos in positions:
            individual[pos[0]][pos[1]] = color
            empty_positions.remove(pos)
    
    # Randomly assign remaining empty positions
    for pos in empty_positions:
        color = random.choice(list(colors.keys()))
        individual[pos[0]][pos[1]] = color
    
    return individual

def findMinimumColor():
    NotImplementedError
    
# Fitness function
def fitness(individual):
    grid_size = len(individual)
    colors = ['Y', 'B', 'R', 'G']  # List of colors to track
    cluster_count = 0
    target_connections = 0
    visited = [[False for _ in range(grid_size)] for _ in range(grid_size)]

    # Helper function to perform DFS and count clusters
    def dfs(x, y, color):
        stack = [(x, y)]
        cluster_size = 0
        while stack:
            cx, cy = stack.pop()
            if visited[cx][cy] or individual[cx][cy] != color:
                continue
            visited[cx][cy] = True
            cluster_size += 1
            # Check four directions
            for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                if 0 <= nx < grid_size and 0 <= ny < grid_size and not visited[nx][ny]:
                    stack.append((nx, ny))
        return cluster_size

    # Count clusters for each color
    color_clusters = {color: 0 for color in colors}
    for i in range(grid_size):
        for j in range(grid_size):
            if individual[i][j] in colors:
                if not visited[i][j]:
                    color = individual[i][j]
                    if dfs(i, j, color) > 0:
                        color_clusters[color] += 1

    # Calculate total clusters as deviation from ideal (1 cluster per color)
    cluster_count = sum(abs(color_clusters[color]) for color in colors)
    # Calculate target connections
    target_points = {
        'Y': [(0, 0), (4, 4)],
        'B': [(0, 2), (3, 1)],
        'R': [(0, 3), (3, 4)],
        'G': [(0, 4), (2, 3)],
    }
    
    for color, points in target_points.items():
        for x, y in points:
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if 0 <= nx < grid_size and 0 <= ny < grid_size and individual[nx][ny] == color:
                    target_connections += 1
                    break
    # Fitness score calculation
    # target_connections = 8(越大越好，但不大於8)
    # cluster_count = 4 (越小越好，但不小於4)
    max_cluster_deviation = 4  # Worst case if cluster_count is 8
    max_target_connections = 8  # Optimal target connections

    # Normalized scores
    cluster_score = max(0, (max_cluster_deviation/cluster_count))
    connection_score = max(0, (target_connections / max_target_connections))
    # Each component contributes 50% of the total score
    score = (0.8 * cluster_score) + (0.2 * connection_score)
    return score

# Selection, crossover, and mutation functions
def select_parents(population, fitness_scores, tournament_size=5):
    # Lists to hold selected parents
    #parents1 = []  # Store best individuals
    #parents2 = []  # Store second best individuals
    #for _ in range(len(population)):
    # Randomly select individuals for the tournament
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_individuals = [population[i] for i in tournament_indices]
    tournament_scores = [fitness_scores[i] for i in tournament_indices]

    # Sort the tournament participants by their fitness scores
    sorted_indices = [x for _, x in sorted(zip(tournament_scores, tournament_indices), reverse=True)]
    
    # Select the best and the second best individuals
    best_individual_index = sorted_indices[0]
    second_best_individual_index = sorted_indices[1]

    # Add best to parents1 and second best to parents2
    parents1 = population[best_individual_index]
    parents2 = population[second_best_individual_index]

    return parents1, parents2

import random

def crossover(parent1, parent2):
    # Choose a random row as the crossover point
    cut_row = random.randint(1, grid_size - 1)  # Ensures at least one row from each parent

    # Combine rows from each parent based on the crossover point
    child = parent1[:cut_row] + parent2[cut_row:]
    
    return child

def mutate(individual):
    # Flatten grid size for easier indexing
    grid_size = len(individual)
    flat_grid = [cell for row in individual for cell in row]

    # Gather all protected indices from target_points
    protected_indices = set()
    for points in target_points.values():
        for (x, y) in points:
            protected_indices.add(x * grid_size + y)  # Flattened index calculation

    # Get list of all indices except protected ones
    mutate_indices = [i for i in range(len(flat_grid)) if i not in protected_indices]

    # Choose a single random index to mutate
    if mutate_indices:
        mutate_position = random.choice(mutate_indices)
        
        # Select a new color randomly from the available colors
        current_color = flat_grid[mutate_position]
        possible_colors = [color for color in colors.keys() if color != current_color]
        new_color = random.choice(possible_colors)
        
        # Apply the mutation
        flat_grid[mutate_position] = new_color

    # Convert flat grid back to 2D format
    mutated_individual = [flat_grid[i:i + grid_size] for i in range(0, len(flat_grid), grid_size)]
    return mutated_individual

def showGrid(population):
    for i in population:
        for j in i:
            print(j)
        print()    
# Genetic Algorithm
def genetic_algorithm(grid, colors, generations=5000):
    population = [generate_individual(grid, colors) for _ in range(100)]
    best1 = 0
    best2 = 0
    #showGrid(population)
    for gen in range(generations):
        fitness_scores = [fitness(ind) for ind in population]
        if max(fitness_scores) == 1:  # Define desired_fitness threshold
            return max(fitness_scores), population[fitness_scores.index(max(fitness_scores))]
        # Generate new population through selection, crossover, mutation
        parent1, parent2 = select_parents(population, fitness_scores)
        if best1 != parent1 and best2 != parent2:        
            offspring = crossover(parent1, parent2)
            population.append(offspring)
            best1 = parent1
            best2 = parent2
        mutate_indice = random.sample(range(len(population)),1)
        mutate_result = mutate(population[mutate_indice[0]])
        population[mutate_indice[0]] = mutate_result
        print(gen, max(fitness_scores))
    """"""
    return max(fitness_scores), population[fitness_scores.index(max(fitness_scores))]

# Run algorithm
fitness_score, solution = genetic_algorithm(target_grid, colors)
"""
"""
print(fitness_score)
for row in solution:
    print(" ".join(row))