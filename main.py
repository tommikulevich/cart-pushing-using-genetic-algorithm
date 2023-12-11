import numpy as np
import matplotlib.pyplot as plt

N = 15
POPULATION_SIZE = 70
GENERATIONS = 200
MUTATION_RATE = 0.5


class GeneticAlgorithm:
    def __init__(self):
        self.population = self.initialize_population()
        self.fitness = np.zeros(POPULATION_SIZE)

    @staticmethod
    def initialize_population():
        return np.random.uniform(low=0.0, high=1.0, size=(POPULATION_SIZE, N))

    @staticmethod
    def evaluate(individual):
        x = np.zeros((N, 2))
        u = individual[:-1]
        u_last = individual[-1]
        for k in range(N-1):
            x[k+1, 0] = x[k, 1]
            x[k+1, 1] = 2*x[k, 1] - x[k, 0] + (1/N**2) * u[k]
        u_last_calculated = u_last - (x[N-1, 1] - x[N-1, 0])
        if u_last_calculated < 0:
            return -np.inf
        return x[N-1, 1] - (1/(2*N)) * np.sum(u**2)

    def evaluate_population(self):
        for i in range(POPULATION_SIZE):
            self.fitness[i] = self.evaluate(self.population[i])

    def select_parents(self):
        return self.population[np.argmax(self.fitness)]

    @staticmethod
    def cx_crossover(parent1, parent2):
        child = parent1.copy()
        for i in range(N):
            if np.random.rand() < MUTATION_RATE:
                child[i] = parent2[i]
        return child

    @staticmethod
    def mutate(individual):
        mutation_idx = np.random.randint(low=0, high=N)
        individual[mutation_idx] = np.random.uniform(low=0.0, high=1.0)
        return individual

    def evolve(self):
        new_population = np.zeros((POPULATION_SIZE, N))
        self.evaluate_population()
        for i in range(POPULATION_SIZE):
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            child = self.cx_crossover(parent1, parent2)
            new_population[i] = self.mutate(child)
        self.population = new_population

    def run(self):
        best_fitness = []
        for _ in range(GENERATIONS):
            self.evolve()
            self.evaluate_population()
            best_fitness.append(np.max(self.fitness))
        self.plot_fitness_evolution(best_fitness)

    @staticmethod
    def plot_fitness_evolution(best_fitness):
        plt.figure()
        plt.plot(best_fitness)
        plt.title(f'Fitness Evolution ({best_fitness[-1]:.6f})')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()


if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.run()
    