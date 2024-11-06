import numpy as np

from uniopt.optimization.optimizer import BaseOptimizer


class LOCALOptimizer(BaseOptimizer):
    def __init__(
        self,
        optimization_context,
        population_size=1,
        generations=1,
        no_improve_num_max=None,
        no_improve_num_factor=5,
        swaps=4,
    ):
        super().__init__(optimization_context)
        self.population_size = population_size
        self.generations = generations
        self.models_num = len(self.optimization_context.bounds.input_values)
        self.rms_num = self.optimization_context.bounds.number_variables
        self.no_improve_num_max = no_improve_num_max
        self.no_improve_num_factor = no_improve_num_factor
        self.swaps = swaps
        seed = self.optimization_context.seed + self.population_size
        self.rng = np.random.default_rng(seed=seed)

    def initialization(self):
        if self.no_improve_num_max is None:
            self.no_improve_num_max = (
                self.no_improve_num_factor
                * (self.models_num - self.rms_num)
                * (self.rms_num)
            )

    def swap(self, array, swap_n=1):
        array = array.copy() if array.ndim == 1 else array.flatten()
        ones = np.argwhere(array == 1).flatten()
        zeros = np.argwhere(array == 0).flatten()

        assert len(array) == len(zeros) + len(ones)

        zeros_to_swap = self.rng.choice(zeros, swap_n, replace=False)
        ones_to_swap = self.rng.choice(ones, swap_n, replace=False)
        bits_to_swap = np.hstack((zeros_to_swap, ones_to_swap))

        array[bits_to_swap] = np.logical_not(array[bits_to_swap])
        return array

    def evolve(self):
        model_pairs_num = range(self.swaps, 0, -1)
        no_improve_swap_max = int(self.no_improve_num_max / self.swaps)
        no_improve_abs_count = 0
        iters_total_count = 0

        solution = self.optimization_context.generate_solution()
        of_value = self.optimization_context.evaluate_solution(solution)

        for swap_n in model_pairs_num:
            no_improve_swap_count = 0
            while no_improve_swap_count < no_improve_swap_max:
                solution_candidate = self.swap(solution, swap_n=swap_n)
                of_value_candidate = (
                    self.optimization_context.evaluate_solution(
                        solution_candidate
                    )
                )
                self.solutions.append((solution_candidate, of_value_candidate))
                if of_value_candidate < of_value:
                    solution = solution_candidate
                    of_value = of_value_candidate
                    no_improve_swap_count = 0
                else:
                    no_improve_swap_count += 1

                iters_total_count += 1
            no_improve_abs_count += no_improve_swap_count

        self.best_solution, self.best_fitness = solution, of_value
        self.solutions.sort(key=lambda x: x[1])
