import random
import numpy as np
import matplotlib.pyplot as plt


N = 15
POPULATION_SIZE = 70
NON_ELITE_SIZE = int(0.90 * POPULATION_SIZE)
GENERATIONS = 4000
MUTATION_RATE = 0.25


class GeneticAlgorithm:
    def __init__(self):
        self.population = self.initialize_population()
        self.fitness = np.zeros(POPULATION_SIZE)
        self.best_fitness = []

    @staticmethod
    def initialize_population():
        return np.random.uniform(low=0.0, high=1.0, size=(POPULATION_SIZE, N))

    @staticmethod
    def calculate_ideal_fitness():
        return 1 / 3 - (3 * N - 1) / (6 * N ** 2) - 1 / (2 * N ** 3) * np.sum(np.square(range(N)))

    @staticmethod
    def evaluate(individual):
        x = np.zeros((N, 2))
        u = individual
        for k in range(N-1):
            x[k + 1, 0] = x[k, 1]
            x[k + 1, 1] = 2 * x[k, 1] - x[k, 0] + (1 / N ** 2) * u[k]
        return x[N - 1, 1] - (1 / (2*N)) * np.sum(u ** 2)

    def evaluate_population(self):
        for i in range(POPULATION_SIZE):
            self.fitness[i] = self.evaluate(self.population[i])

    def select_parents(self):
        sorted_indices = np.argsort(self.fitness)
        return self.population[sorted_indices[-1]], self.population[sorted_indices[-2]]

    def wheel_selection(self):
        normalized_fitness = np.maximum(self.fitness, 0)
        total_fitness = np.sum(normalized_fitness)
        selection_probabilities = normalized_fitness / total_fitness
        selected_indices = np.random.choice(len(self.population), 2, p=selection_probabilities)
        selected_individuals = [self.population[i] for i in selected_indices]
        return selected_individuals

    @staticmethod
    def single_point_crossover(parent1, parent2):
        random_part = random.randint(0, N)
        child = parent1.copy()
        child[random_part:] = parent2[random_part:]
        return child

    @staticmethod
    def mutate(individual):
        if np.random.rand() < MUTATION_RATE:
            mutation_idx = np.random.randint(low=0, high=N)
            individual[mutation_idx] = np.random.uniform(low=0.0, high=1.0)
        return individual

    def elite(self):
        sorted_indices = np.argsort(self.fitness)
        return self.population[sorted_indices[NON_ELITE_SIZE:]]

    def evolve(self):
        new_population = np.zeros((NON_ELITE_SIZE, N))
        self.evaluate_population()
        elite_population = self.elite()
        for i in range(NON_ELITE_SIZE):
            parent1, parent2 = self.wheel_selection()
            child = self.single_point_crossover(parent1, parent2)
            new_population[i] = self.mutate(child)
        self.population = np.concatenate((new_population, elite_population))

    def run(self):
        for _ in range(GENERATIONS):
            self.evolve()
            self.evaluate_population()
            self.best_fitness.append(np.max(self.fitness))

    def plot_fitness_evolution(self):
        plt.figure()
        plt.plot(self.best_fitness)
        plt.title(f'Fitness Evolution ({self.best_fitness[-1]:.6f})')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()


if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.run()
    ga.plot_fitness_evolution()
