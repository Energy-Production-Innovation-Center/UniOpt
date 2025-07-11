import numpy as np

from uniopt.optimization.optimizer import BaseOptimizer


class GAOptimizer(BaseOptimizer):
    def __init__(
        self,
        optimization_context,
        population_size=50,
        generations=100,
        mutation_rate=0.01,
    ):
        super().__init__(optimization_context, population_size, generations)
        self.mutation_rate = mutation_rate
        seed = self.optimization_context.seed + self.population_size
        self.rng = np.random.default_rng(seed=seed)

    def before_initialization(self):
        self.population = []

    def initialization(self):
        for _ in range(self.population_size):
            solution = self.optimization_context.generate_solution()
            self.population.append(solution)

    def after_initialization(self):
        self.evaluate_population()

    def evaluate_population(self):
        self.solutions = [
            (sol, self.optimization_context.evaluate_solution(sol)) for sol in self.population
        ]
        self.solutions.sort(key=lambda x: x[1])
        self.best_solution, self.best_fitness = self.solutions[0]

    def select_parents(self):
        tournament_size = 3
        tournament_idx = self.rng.choice(range(len(self.solutions)), tournament_size, replace=False)
        tournament = [sol for idx, sol in enumerate(self.solutions) if idx in tournament_idx]
        tournament.sort(key=lambda x: x[1])
        return tournament[0][0], tournament[1][0]

    def crossover(self, parent1, parent2):
        crossover_point = self.rng.integers(1, len(parent1))
        offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

        return self.adjust_solution(offspring)

    def mutate(self, solution):
        for i in range(len(solution)):
            if self.rng.random() < self.mutation_rate:
                solution[i] = 1 - solution[i]

        return self.adjust_solution(solution)

    def adjust_solution(self, solution):
        current_ones = np.sum(solution)

        if current_ones > self.optimization_context.bounds.number_variables:
            indices = np.where(solution == 1)[0]
            indices_to_zero = self.rng.choice(
                indices,
                current_ones - self.optimization_context.bounds.number_variables,
                replace=False,
            )
            solution[indices_to_zero] = 0

        elif current_ones < self.optimization_context.bounds.number_variables:
            indices = np.where(solution == 0)[0]
            indices_to_one = self.rng.choice(
                indices,
                self.optimization_context.bounds.number_variables - current_ones,
                replace=False,
            )
            solution[indices_to_one] = 1

        return solution

    def evolve(self):
        new_population = []

        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            offspring = self.crossover(parent1, parent2)
            offspring = self.mutate(offspring)
            new_population.append(offspring)

        self.population = new_population
        self.evaluate_population()
