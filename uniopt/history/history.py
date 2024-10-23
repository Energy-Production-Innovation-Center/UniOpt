import numpy as np


class History:
    def __init__(self):
        self.best_solutions = []
        self.best_fitness = []
        self.mean_fitness = []
        self.std_fitness = []

    def append(self, best_solution, best_fitness, mean_fitness, std_fitness):
        """Add a new entry to the history."""
        self.log(best_solution, best_fitness, mean_fitness, std_fitness)

    def log(self, best_solution, best_fitness, mean_fitness, std_fitness):
        self.best_solutions.append(best_solution)
        self.best_fitness.append(best_fitness)
        self.mean_fitness.append(mean_fitness)
        self.std_fitness.append(std_fitness)

    def get_history(self):
        return {
            "best_solutions": self.best_solutions,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
        }

    def clear(self):
        self.best_solutions.clear()
        self.best_fitness.clear()
        self.mean_fitness.clear()
        self.std_fitness.clear()

    def get_best_solution(self):
        if not self.best_solutions:
            return None, None
        return self.best_solutions[-1], self.best_fitness[-1]

    def calculate_mean_fitness(self, fitness_values):
        return np.mean(fitness_values)

    def calculate_std_fitness(self, fitness_values):
        return np.std(fitness_values)
