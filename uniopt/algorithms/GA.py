import random

from uniopt.optimization.optimizer import BaseOptimizer


class GAOptimizer(BaseOptimizer):
    def __init__(
        self, problem, population_size=50, generations=100, mutation_rate=0.01
    ):
        super().__init__(problem, population_size, generations)
        self.mutation_rate = mutation_rate

    def before_initialization(self):
        self.population = []

    def initialization(self):
        for _ in range(self.population_size):
            solution = self.problem.create_random_solution()
            self.population.append(solution)

    def after_initialization(self):
        self.evaluate_population()

    def evaluate_population(self):
        self.solutions = [
            (sol, self.problem.evaluate(sol)) for sol in self.population
        ]
        self.solutions.sort(key=lambda x: x[1])
        self.best_solution, self.best_fitness = self.solutions[0]

    def select_parents(self):
        tournament_size = 3
        tournament = random.sample(self.solutions, tournament_size)
        tournament.sort(key=lambda x: x[1])
        return tournament[0][0], tournament[1][0]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring = parent1[:crossover_point] + parent2[crossover_point:]
        return offspring

    def mutate(self, solution):
        for i in range(len(solution)):
            if random.random() < self.mutation_rate:
                solution[i] = 1 - solution[i]
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
