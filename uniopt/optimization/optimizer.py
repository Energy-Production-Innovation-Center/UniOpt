from uniopt.context.optimization_context import OptimizationContext
from uniopt.history.history import History
from uniopt.logger import Logger


class BaseOptimizer:
    def __init__(
        self, optimization_context, population_size=50, generations=100
    ):
        if population_size <= 0:
            raise ValueError("Population size must be greater than 0.")
        if generations <= 0:
            raise ValueError("Number of generations must be greater than 0.")

        self.check_optimization_context(optimization_context)

        self.optimization_context = optimization_context
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_solution = None
        self.best_fitness = float("inf")

        self.history = History()
        self.logger = Logger()
        self.solutions = []

    # TODO: Population/variables initialization
    def before_initialization(self):
        """Preparation before initializing the population."""
        self.logger.log_info("Starting initialization phase.")

    def initialization(self):
        """Method to initialize the population."""
        self.logger.log_info("Initializing population.")

    def after_initialization(self):
        """Post-initialization steps."""
        self.logger.log_info("Initialization complete.")

    def check_optimization_context(self, optimization_context):
        """Checks and sets the optimization context."""
        if isinstance(optimization_context, OptimizationContext):
            self.optimization_context = optimization_context
        elif isinstance(optimization_context, dict):
            self.optimization_context = OptimizationContext(
                **optimization_context
            )
        else:
            raise ValueError(
                "The optimization context must be a dictionary or an instance "
                "of OptimizationContext"
            )

        self.history = History()

    def check_stopping_criteria(self):
        """Implement logic to check stopping criteria."""
        return False

    def evaluate_population(self):
        pass

    def evolve():
        pass

    def run(self):
        """Runs the optimization process."""
        self.check_stopping_criteria()

        self.before_initialization()
        self.initialization()
        self.after_initialization()

        for generation in range(1, self.generations + 1):
            self.evolve()

            self.logger.log_info(
                f"Generation {generation}: Best fitness = {self.best_fitness}"
            )

            if self.check_stopping_criteria():
                self.logger.log_info(
                    "Stopping criteria met, ending optimization."
                )

                break

            self.history.append(self.best_solution, self.best_fitness)

        return self.best_solution, self.best_fitness
