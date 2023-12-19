import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt


N = 15
POPULATION_SIZE = 70
NON_ELITE_RATE = 0.9
NON_ELITE_SIZE = int(NON_ELITE_RATE * POPULATION_SIZE)
GENERATIONS = 1000
MUTATION_RATE = 0.25

LOGS_PATH = "logs"
LOG_NAME = f"{LOGS_PATH}/data_N{N}_PS{POPULATION_SIZE}_NER{NON_ELITE_RATE}_G{GENERATIONS}_MR{MUTATION_RATE}.txt"


class GeneticAlgorithm:
    def __init__(self):
        self.population = self.initialize_population()
        self.fitness = np.zeros(POPULATION_SIZE)
        self.best_fitness = []
        self.best_individuals = []

    @staticmethod
    def initialize_population():
        return np.random.uniform(low=0.0, high=1.0, size=(POPULATION_SIZE, N))

    @staticmethod
    def calculate_ideal_fitness(N_num):
        return 1 / 3 - (3 * N_num - 1) / (6 * N_num ** 2) - 1 / (2 * N_num ** 3) * np.sum(np.square(range(N_num)))

    @staticmethod
    def evaluate(individual):
        x = np.zeros((N, 2))
        u = individual
        for k in range(N-1):
            x[k + 1, 0] = x[k, 1]
            x[k + 1, 1] = 2 * x[k, 1] - x[k, 0] + (1 / N ** 2) * u[k]
        return x[N - 1, 1] - (1 / (2*N)) * np.sum(u ** 2)

    def evaluate_population(self):
        self.fitness = np.apply_along_axis(self.evaluate, 1, self.population)

    def wheel_selection(self):
        normalized_fitness = np.maximum(self.fitness, 0)
        total_fitness = np.sum(normalized_fitness)
        selection_probabilities = normalized_fitness / total_fitness
        selected_indices = np.random.choice(len(self.population), 2, p=selection_probabilities)
        return self.population[selected_indices]

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

    def best(self):
        sorted_indices = np.argsort(self.fitness)
        return self.population[sorted_indices[-1]]

    def evolve(self):
        new_population = np.zeros((NON_ELITE_SIZE, N))
        self.evaluate_population()
        elite_population = self.elite()
        for i in range(NON_ELITE_SIZE):
            parents = self.wheel_selection()
            child = self.single_point_crossover(*parents)
            new_population[i] = self.mutate(child)
        self.population = np.concatenate((new_population, elite_population))
        self.evaluate_population()
        self.best_individuals.append(self.population[np.argmax(self.fitness)])
        # self.best_individuals.append(self.best())

    def run(self):
        for _ in range(GENERATIONS):
            self.evolve()
            self.evaluate_population()
            self.best_fitness.append(np.max(self.fitness))
        self.save_data_to_file()

    def plot_fitness_evolution(self):
        best_fitness = self.calculate_ideal_fitness(N)
        plt.figure()
        plt.plot(self.best_fitness)
        plt.axhline(y=best_fitness, color='r', linestyle='--', label=f"J* = {best_fitness:.6f}")
        plt.title(f'Fitness Evolution')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

    def plot_multiple_fitness_evolutions(self):
        plt.figure()
        files = glob.glob(f"{LOGS_PATH}/data_N*_PS*_NER*_G*_MR*.txt")
        files.sort(key=lambda x: int(x.split('_')[1][1:]))
        for file in files:
            data = np.loadtxt(file)
            N_file = int(os.path.basename(file).split('_')[1][1:])
            plt.plot(data, label=f'N = {N_file} ({self.calculate_ideal_fitness(N_file):.6f})')
        plt.title('Fitness Evolution Comparison')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

    def plot_u_evolution(self):
        population_best_individuals = np.array(self.best_individuals)
        plt.figure(figsize=(10, 6))
        for i in range(N):
            plt.plot(population_best_individuals[:, i], label=f'u({i + 1})')
        plt.title('Evolution of best u(N)')
        plt.xlabel('Generation')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def save_data_to_file(self):
        os.makedirs(LOGS_PATH, exist_ok=True)
        np.savetxt(LOG_NAME, self.best_fitness)

    @staticmethod
    def load_data_from_file(filename):
        return np.loadtxt(filename)


if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.run()
    # ga.plot_fitness_evolution()
    ga.plot_u_evolution()
    # ga.plot_multiple_fitness_evolutions()
