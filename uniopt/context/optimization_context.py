from abc import abstractmethod
from time import time
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from uniopt.context.variables import Variables
from uniopt.logger import Logger
from uniopt.utils.custom_types import ResultsType, SolutionType


class OptimizationContext:
    ALLOWED_OF_VALUES: tuple[type, type, type, type, type, type] = (
        tuple,
        list,
        np.ndarray,
        int,
        float,
        np.float64,
    )
    ALLOWED_OF_RESULTS: tuple[type, type, type] = (int, float, np.float64)

    def __init__(
        self,
        bounds: Any,
        target: str = "min",
        number_objectives: int = 1,
        permutation: bool = False,
        weights: NDArray[np.float64] | None = None,
        log_path: str | None = None,
        initial_solution: SolutionType | None = None,
        seed: int | None = None,
        logger: Logger | None = None,
        **kwargs: Any,
    ):
        self.bounds: Variables = self.set_bounds(bounds)
        self.target: str = target
        self.number_objectives: int = number_objectives
        self.permutation: bool = permutation
        self.weights: NDArray[np.float64] = (
            weights if weights is not None else np.ones(number_objectives)
        )
        self.log_path: str | None = log_path
        self.initial_solution: SolutionType | None = initial_solution
        self.seed: int = self.set_seed(seed)
        self.rng: Generator = np.random.default_rng(seed=self.seed * 2)
        self.logger: Logger = Logger() if logger is None else logger

        self.__set_functions()

    def set_bounds(self, bounds: Any) -> Variables:
        if not isinstance(bounds, Variables):
            raise TypeError("The bounds must be an instance of type Variables")
        return bounds

    # TODO: Include the generation of other types
    def generate_solution(self) -> SolutionType:
        """Generate a random solution based on bounds."""
        if self.permutation:
            solution = self.bounds.input_values.astype(np.bool)
            solution = self.rng.permutation(solution)
        else:
            solution = SolutionType([])
        return solution

    def set_seed(self, seed: int | None) -> int:
        """Set the seed for random number generation."""
        seed_value = seed if seed is not None else int(time())
        self.logger.log_info(f"Value set for seed: {seed_value}")
        return seed_value

    def __set_functions(self):
        """Initialize objective function validation."""

        solution = self.generate_solution()
        of_value = self.obj_func(solution)[0]

        if isinstance(of_value, self.ALLOWED_OF_VALUES):
            result = np.array(of_value).flatten()
            self.validate_objective_function_result(result)
        else:
            raise ValueError("The value returned by the objective function is not supported")

    def validate_objective_function_result(self, result: Any):
        """Validate the result returned by the objective function."""
        if self.number_objectives == 1:
            if len(result) != 1 or not isinstance(result[0], self.ALLOWED_OF_RESULTS):
                raise ValueError(
                    "The return value of the objective function must be a single integer or float"
                )

            if len(self.weights) != 1 or not isinstance(self.weights[0], self.ALLOWED_OF_RESULTS):
                raise ValueError("Weight must be a single integer or float")

        elif self.number_objectives > 1:
            if len(result) != self.number_objectives:
                raise ValueError(
                    "The objective function must return " + f"{self.number_objectives} values"
                )

            if len(self.weights) != self.number_objectives:
                raise ValueError(
                    "The number of weights must be equal to the number of objective functions"
                )

            for i in range(self.number_objectives):
                if not isinstance(result[i], self.ALLOWED_OF_RESULTS):
                    raise ValueError(f"The value {result[i]} must be of type integer or float")
                if not isinstance(self.weights[i], self.ALLOWED_OF_RESULTS):
                    raise ValueError(f"Weight {self.weights[i]} is not of type integer or float")

        else:
            raise ValueError("The number of objectives must be greater than zero")

    @abstractmethod
    def obj_func(self, solution: SolutionType) -> tuple[np.float64, ResultsType]:
        """Objective function to be defined by subclasses.

        Args:
            solution (SolutionType): Solution binary array.

        Returns:
            tuple[np.float64, ResultsType]: Objective function value and generated results.
        """
        raise NotImplementedError

    def evaluate_solution(self, solution: SolutionType) -> tuple[np.float64, ResultsType]:
        return self.obj_func(solution)

    @abstractmethod
    def get_global_context(self) -> dict[str, Any]:
        """Retrieve the global context dictionary that's managed by a singleton on the main thread.
        It's necessary to obtain its references when using spawn() to create new processes.

        Returns:
            dict[str, Any]: Global context dictionary.
        """
        raise NotImplementedError
