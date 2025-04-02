from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, cast

import numpy as np

from uniopt.context.optimization_context import OptimizationContext
from uniopt.history.history import History
from uniopt.logger import Logger
from uniopt.utils.custom_types import ResultsType, SolutionType


class BaseOptimizer(ABC):
    def __init__(
        self,
        optimization_context: OptimizationContext,
        population_size: int = 50,
        generations: int = 100,
    ):
        if population_size <= 0:
            raise ValueError("Population size must be greater than 0.")
        if generations <= 0:
            raise ValueError("Number of generations must be greater than 0.")

        self.optimization_context: OptimizationContext = self.check_optimization_context(
            optimization_context
        )
        self.population_size: int = population_size
        self.generations: int = generations
        self.population: list[SolutionType] = []
        self.best_solution: SolutionType | None = None
        self.best_fitness: np.float64 = np.float64("inf")

        self.history: History = History()
        self.solutions: list[tuple[SolutionType, np.float64]] = []
        self._logger: Logger = Logger()

    # TODO: Population/variables initialization
    def before_initialization(self):
        """Preparation before initializing the population."""
        self._logger.log_info("Starting initialization phase.")

    def initialization(self):
        """Method to initialize the population."""
        self._logger.log_info("Initializing population.")

    def after_initialization(self):
        """Post-initialization steps."""
        self._logger.log_info("Initialization complete.")

    def check_optimization_context(self, optimization_context: Any) -> OptimizationContext:
        """Checks and sets the optimization context."""
        if isinstance(optimization_context, OptimizationContext):
            pass
        elif isinstance(optimization_context, dict):
            optimization_context = OptimizationContext(**cast(dict[str, Any], optimization_context))
        else:
            raise ValueError(
                "The optimization context must be a dictionary or an instance of "
                + "OptimizationContext"
            )

        return optimization_context

    def check_stopping_criteria(self) -> bool:
        """Implement logic to check stopping criteria."""
        return False

    def evaluate_population(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def evolve(self) -> Generator[ResultsType]:
        raise NotImplementedError

    def run(self) -> Generator[ResultsType]:
        """Runs the optimization process."""

        self.before_initialization()
        self.initialization()
        self.after_initialization()

        for generation in range(1, self.generations + 1):
            for results in self.evolve():
                yield results

            self._logger.log_info(f"Generation {generation}: Best fitness = {self.best_fitness}")

            if self.check_stopping_criteria():
                self._logger.log_info("Stopping criteria met, ending optimization.")
                break

            # TODO
            # self.history.append(self.best_solution, self.best_fitness)
