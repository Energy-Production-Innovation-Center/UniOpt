import time

import numpy as np

from uniopt.context.variables import Variables
from uniopt.logger import Logger


class OptimizationContext:
    ALLOWED_VALUES = [tuple, list, np.ndarray, int, float]

    def __init__(
        self,
        bounds,
        target="min",
        number_objectives=1,
        permutation=False,
        weights=None,
        log_path=None,
        initial_solution=None,
        seed=None,
        **kwargs,
    ):
        self.bounds = self.set_bounds(bounds)
        self.target = target
        self.number_objectives = number_objectives
        self.permutation = permutation
        self.weights = (
            weights if weights is not None else np.ones(number_objectives)
        )
        self.log_path = log_path
        self.initial_solution = initial_solution
        self.logger = Logger()
        self.seed = self.set_seed(seed)
        seed = self.seed + self.seed
        self.rng = np.random.default_rng(seed=seed)

        self.__set_functions()

    def set_bounds(self, bounds):
        if not isinstance(bounds, Variables):
            raise TypeError("The bounds must be an instance of type Variables")

        return bounds

    # TODO: Include the generation of other types
    def generate_solution(self):
        """Generate a random solution based on bounds."""

        solution = []

        if self.permutation:
            solution = self.bounds.input_values
            solution = self.rng.permutation(solution)

        return solution

    def set_seed(self, seed) -> None:
        """Set the seed for random number generation."""

        seed_value = seed
        if seed is not None:
            seed_value = seed
        else:
            seed_value = int(time.time())

        self.logger.log_info(f"Value set for seed: {seed_value}")

        return seed_value

    def __set_functions(self):
        """Initialize objective function validation."""

        solution = self.generate_solution()
        result = self.obj_func(solution)

        if isinstance(result, (tuple, list, np.ndarray, float, int)):
            result = np.array(result).flatten()
            self.validate_objective_function_result(result)
        else:
            raise ValueError(
                "The value returned by the objective function is not supported"
            )

    def validate_objective_function_result(self, result):
        """Validate the result returned by the objective function."""
        if self.number_objectives == 1:
            if len(result) != 1 or not isinstance(result[0], (int, float)):
                raise ValueError(
                    "The return value of the objective function must "
                    "be a single integer or float"
                )

            if len(self.weights) != 1 or not isinstance(
                self.weights[0], (int, float)
            ):
                raise ValueError("Weight must be a single integer or float")

        elif self.number_objectives > 1:
            if len(result) != self.number_objectives:
                raise ValueError(
                    "The objective function must return "
                    f"{self.number_objectives} values"
                )

            if len(self.weights) != self.number_objectives:
                raise ValueError(
                    "The number of weights must be equal to the number "
                    "of objective functions"
                )

            for i in range(self.number_objectives):
                if not isinstance(result[i], (int, float)):
                    raise ValueError(
                        f"The value {result[i]} must be of "
                        "type integer or float"
                    )
                if not isinstance(self.weights[i], (int, float)):
                    raise ValueError(
                        f"Weight {self.weights[i]} is not of "
                        "type integer or float"
                    )

        else:
            raise ValueError(
                "The number of objectives must be greater than zero"
            )

    def obj_func(self, x):
        """Objective function to be defined by subclasses"""
        raise NotImplementedError

    def evaluate_solution(self, solution):
        return self.obj_func(solution)
