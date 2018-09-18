#%% Solving the Knapsack Problem with Genetic Algorithms

# Purpose: To fill a 0/1 knapsack using various genetic algorithms

# Author:  Matthew Renze

# Description: The Knapsack Problem is a common combinatorial optimization
# problem in computer science. It's defined as follows:
# Given a knapsack of a limited size, the goal is to fill the knapsack
# with as many valuable items as possible subject to size constraint.

# This script runs six algorithms (3 heuristic and 3 genetic)
# to fill a knapsack with items and compares their results.

# The algorithms are as follows:
#  - Value - fills by value (decending) and size (ascending)
#  - Size - fills by size (ascending) and value (decending)
#  - Density - fills by value / size (decending)
#  - Mutation - fills by genetic mutation only
#  - Crossover - fills by genetic crossover only
#  - Genetic - fills by both genetic crossover and mutation

# The genetic algorithm (i.e. Genetic) generally performs as well as 
# the best heuristic algorithms (i.e Density). However, a lot of fine-tuning
# is often necessary to get the algorithm to perform well.

# Note: For more info on the Knapsack Problem, please see the following:
# https://en.wikipedia.org/wiki/Knapsack_problem

#%% Import libraries
import random

#%% Define an item class
class Item(object):
    def __init__(self, value, size):
        self.value = value
        self.size = size

#%% Define a knapsack class
class Knapsack():
    def __init__(self, size):
        self.size = size
        self.capacity = size
        self.value = 0
        self.items = []

    def add_item(self, item):
        self.items.append(item)
        self.value += item.value
        self.capacity -= item.size

    def can_add_item(self, item):
        return self.capacity - item.size >= 0

#%% Create random items
def create_random_items(n, min_value, max_value, min_size, max_size):
    items = []
    for i in range(n):
        value = random.randint(min_value, max_value)
        size = random.randint(min_size, max_size)
        item = Item(value, size)
        items.append(item)
    return items

#%% Fill a knapsack with value-sort algorithm
def fill_knapsack_by_value(knapsack_size, items):
    
    # Create a knapsack    
    knapsack = Knapsack(knapsack_size)
    
    # Order items by value
    ordered_items = sorted(
        items,
        key = lambda x: (-x.value, x.size))
    
    # Fill the knapsack with items until full
    for item in ordered_items:
        if (knapsack.can_add_item(item)):
            knapsack.add_item(item)
    
    # Return fitness
    return knapsack.value

#%% Fill a knapsack with size-sort algorithm
def fill_knapsack_by_size(knapsack_size, items):  
   
    # Create a knapsack
    knapsack = Knapsack(knapsack_size)
    
    # Order items by size
    ordered_items = sorted(
        items,
        key = lambda x: (x.size, -x.value))
    
    # Fill the knapsack with items until full
    for item in ordered_items:
        if (knapsack.can_add_item(item)):
            knapsack.add_item(item)
            
    # Return fitness
    return knapsack.value
            
#%% Fill a knapsack with density-sort algorithm
def fill_knapsack_by_density(knapsack_size, items):  
   
    # Create a knapsack
    knapsack = Knapsack(knapsack_size)
    
    # Order items by size
    ordered_items = sorted(
        items,
        key = lambda x: x.value / x.size,
        reverse = True)
    
    # Fill the knapsack with items until full
    for item in ordered_items:
        if (knapsack.can_add_item(item)):
            knapsack.add_item(item)
            
    # Return fitness
    return knapsack.value

#%% Create initial population of genotypes with random genes
def create_initial_population(population_size, number_of_genes):
    
    # Initialize variables
    population = []
    
    # Create each genotype
    for i in range(population_size):
        genotype = []
        
        # Create each gene (bit) randomly
        for j in range(number_of_genes): 
            gene = random.randint(0, 1)    
            genotype.append(gene)
            
        # Add the genotype to the population
        population.append(genotype)
        
    return population

#%% Fill a knapsack using a genotype mapped to a list of items
#   NOTE: Once full, remaining items are ignored
def fill_knapsack_from_genotype(knapsack, genotype, items):
    
    # Insert each items based on corresponding gene
    for i in range(0, len(genotype)):
        
        # Get the gene at the current index
        gene = genotype[i]
        
        # Get the item at the current index
        item = items[i]
        
        # Add the item if the gene says to add it and it can be added
        if gene == 1 and knapsack.can_add_item(item):
            knapsack.add_item(item)

#%% Fill a knapsack for each genotype in the population
def fill_knapsack_from_genotype_for_population(population, knapsack_size):
    
    # Initialize variables
    pairs = []
    
    # Iterate through the population
    for genotype in population:
        
        # Create a knapsack
        knapsack = Knapsack(knapsack_size)        
        
        # Fill the knapsack with a genotype-to-item mapping
        fill_knapsack_from_genotype(knapsack, genotype, items)
        
        # Add the key-value (fitness-genotype) pair to the results
        pairs.append((knapsack.value, genotype))
        
    return pairs

#%% Select top elite genetypes by fitness
def select_top_elite(pairs, elitism):
    
    # Sort genotypes based on fitness
    sorted_pairs = sorted(
        pairs,
        key = lambda x: x[0], 
        reverse = True)
    
    # Compute top n survivors based on elitism parameter
    top_n = int(elitism * len(pairs))
    
    # Select top n most fit genotypes to reproduce
    elite_pairs = sorted_pairs[:top_n]
    
    # Project the genotype from the pairs
    elite = [e[1] for e in elite_pairs]
    
    return elite

#%% Clone a population to create a new population
def clone_population(population, new_size):

    # Compute the number of offspring for each genotype
    num_of_offspring = int(new_size / len(population))
    
    # Create a new population for the next generation
    new_population = []
    
    # Iterate through all gentypes in the old population
    for j in range(len(population)):
            
        # Get the genotype from the population
        genotype = population[j]                    
        
        # Clone the old genetype k times
        for k in range(0, num_of_offspring):
            
            # Clone the genotype
            offspring = genotype.copy()
            
            # Add the clone to the new population
            new_population.append(offspring)
        
    # Return the new population
    return new_population
      
#%% Breed a population using genetic crossover to create a new population
def breed_population(population, new_size):
    
    new_population = []
    
    for i in range(new_size):
        
        # Select a father at random
        father_index = random.randint(0, len(population) - 1)
        
        father = population[father_index]
        
        # Select a mother at random
        mother_index = random.randint(0, len(population) - 1)
        
        mother = population[mother_index]
        
        # Create a child
        child = []
        
        # Select crossover point at random
        crossover_index = random.randint(0, len(father) - 1)
        
        # Join father's genes until crossover point
        for j in range(0, crossover_index):
            child.append(father[j])
            
        # Join mother's genes after crossover point
        for k in range(crossover_index, len(mother)):
            child.append(mother[k])
            
        # Add child to next generation
        new_population.append(child)
        
    return new_population

#%% Mutate a population of genotypes
def mutate_population(population):
    
    # Iterate through each genotype in the population
    for j in range(len(population)):
        
        # Select the genotype
        genotype = population[j]
        
        # Select a single gene to mutate
        gene_index = random.randint(0, len(genotype) - 1)
        
        # Flip a single bit
        genotype[gene_index] = int(not genotype[gene_index])
        
    return population

#%% Get fitness statistics for a population
def get_fitness_statistics(population, items, knapsack_size):
     
    # Initialize variables
    best_genotype = []
    best_fitness = 0
    total_fitness = 0
    
    # Iterate through the population
    for genotype in population:
        
        # Create a new knapsack
        knapsack = Knapsack(knapsack_size)
        
        # Fill the knapsack to determine fitness
        fill_knapsack_from_genotype(knapsack, genotype, items)
        
        # If fitness is better, then update best fitness
        if knapsack.value > best_fitness:
            best_fitness = knapsack.value
            best_genotype = genotype
            
        # Update total fitness
        total_fitness += knapsack.value
    
    # Compute average fitness
    avg_fitness = total_fitness / len(population)
    
    # Return statistics for population
    return (best_fitness, avg_fitness, best_genotype)

#%% Fill knapsack with genetic mutation algorithm
def fill_knapsack_by_mutation(knapsack, items):
    return fill_knapsack_by_genetics(knapsack, items, True, False)

#%% Fill knapsack with genetic crossover algorithm
def fill_knapsack_by_crossover(knapsack, items):
    return fill_knapsack_by_genetics(knapsack, items, False, True)
    
#%% Fill knapsack with both mutation and crossover
def fill_knapsack_by_both(knapsack, items):
    return fill_knapsack_by_genetics(knapsack, items, True, True)

#%% Fill knapsack with genetic algorithm
def fill_knapsack_by_genetics(knapsack_size, items, use_mutation, use_crossover):
    
    # Set parameters
    population_size = 1000
    number_of_genes = len(items)
    max_generations = 1000
    convergence_threshold = 100
    elitism = 0.1
          
    # Initialize variables
    best_fitness = 0
    best_genotype = []
    best_average_fitness = 0
    generations_without_improvement = 0
    knapsack = Knapsack(knapsack_size)
    
    # Create initial population
    population = create_initial_population(
        population_size = population_size, 
        number_of_genes = number_of_genes)
    
    # Iterate through n generations
    for i in range(0, max_generations):
    
        # Fill knapsacks and get key-value pairs (fitness and genotype)
        pairs = fill_knapsack_from_genotype_for_population(
            population = population, 
            knapsack_size = knapsack.size)      
        
        # Select top elite genotypes by fitness
        population = select_top_elite(pairs, elitism)              
        
        # Breed the population using genetic crossover or cloning
        if use_crossover:
            population = breed_population(population, population_size)
        else:
            population = clone_population(population, population_size)
        
        # Mutate the population
        if use_mutation:
            population = mutate_population(population)
            
        # Compute statistics for current generation
        statistics = get_fitness_statistics(population, items, knapsack.size)
        
        # Print statistics for current generation
        print("G" + str(i) + ": Max = " +  str(statistics[0]) + " Avg = " + str(statistics[1]))
        
        # Update best fitness
        if statistics[0] > best_fitness:
            best_fitness = statistics[0]
            best_genotype = statistics[2]
            
        # Update convergence based on average statistics
        if statistics[1] > best_average_fitness:
            best_average_fitness = statistics[1]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
            
        # If algorithm has converged then exit loop
        if (generations_without_improvement >= convergence_threshold):
            break        
        
    # Fill knapsack with final solution
    fill_knapsack_from_genotype(knapsack, best_genotype, items)
    
    # Return fitness
    return (knapsack.value, i)

#%% Set parameters
knapsack_size = 100
item_count = 100
min_value = 1
max_value = 10
min_size = 1
max_size = 10

#%% Set random seed
random.seed(42)

#%% Create random items
items = create_random_items(
    n = item_count, 
    min_value = min_value, 
    max_value = max_value, 
    min_size = min_size, 
    max_size = max_size)

#%% Fill the knapsacks
value_fitness = fill_knapsack_by_value(knapsack_size, items)
size_fitness = fill_knapsack_by_size(knapsack_size, items)
density_fitness = fill_knapsack_by_density(knapsack_size, items)
(mutation_fitness, mutation_generations) = fill_knapsack_by_mutation(knapsack_size, items)
(crossover_fitness, crossover_generations) = fill_knapsack_by_crossover(knapsack_size, items)
(genetic_fitness, genetic_generations) = fill_knapsack_by_both(knapsack_size, items)
        
#%% Print results
print("--- Final Results ---")
print("Value:     " + str(value_fitness))
print("Size:      " + str(size_fitness))
print("Density:   " + str(density_fitness))
print("Mutation:  " + str(mutation_fitness)+ " at G" + str(mutation_generations))
print("Crossover: " + str(crossover_fitness) + " at G" + str(crossover_generations))
print("Genetic:   " + str(genetic_fitness) + " at G" + str(genetic_generations))
