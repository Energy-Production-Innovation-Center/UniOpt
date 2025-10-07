import time
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, cpu_count, get_start_method
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.random import Generator as RandomGenerator
from psutil import Process, virtual_memory
from typing_extensions import override

from uniopt.context.optimization_context import OptimizationContext
from uniopt.optimization.optimizer import BaseOptimizer
from uniopt.utils.custom_types import ResultsType, SolutionType

if TYPE_CHECKING:
    from queue import Queue

    from uniopt.logger import Logger


class LOCALOptimizer(BaseOptimizer):
    def __init__(  # noqa: PLR0913
        self,
        optimization_context: OptimizationContext,
        population_size: int = 1,
        generations: int = 1,
        n_processes: int | None = 1,
        n_swaps: int = 1,
        stopping_crit_factor: float = 1,
        stopping_crit_behavior: str = "equal",
        best_solution_ratio: float = 1,
    ):
        super().__init__(optimization_context)
        self.population_size: int = population_size
        self.generations: int = generations
        self.models_num: int = len(self.optimization_context.bounds.input_values)
        self.rms_num: int = self.optimization_context.bounds.number_variables
        seed: int = self.optimization_context.seed + self.population_size
        self.rng: RandomGenerator = np.random.default_rng(seed)
        self.n_swaps: int = n_swaps
        self.stopping_crit_factor: float = stopping_crit_factor
        self.stopping_crit_behavior: str = stopping_crit_behavior
        self.best_solution_ratio: float = best_solution_ratio
        self.no_improve_num_max: int = self._get_no_improve_num_max()

        # Shadow current logger to use the user-configured one from RMSel
        self._logger: Logger = self.optimization_context.logger

        # Check whether to enable auto scaling and multiprocessing
        self.max_processes: int = self._get_max_processes()
        self.auto_scaling: bool = n_processes is None
        self.n_processes: int = (
            min(cpu_count() - 1, self.max_processes) if n_processes is None else n_processes
        )

        # Initialize collections that'll be used during the optimization process
        self._solutions_dict: dict[bytes, int] = {}
        self._shared_queue: (
            Queue[tuple[SolutionType | None, np.float64 | None, ResultsType | None]] | None
        ) = None

        # Super class type annotations
        self.best_solution: SolutionType | None = None
        self.best_fitness: np.float64 = np.float64("inf")
        self.solutions: list[tuple[SolutionType, np.float64]] = []

        self._check_fields()

    def _check_fields(self):
        """Check whether the fields of this class are correctly configured and warns the user if
        that's not the case.
        """
        if self.n_swaps > self.rms_num:
            self._logger.log_warning(
                f"Number of swaps ({self.n_swaps}) is higher than RMs ({self.rms_num}), "
                " using latter"
            )
            self.n_swaps = self.rms_num
        n_cpu = cpu_count()
        if self.n_processes > n_cpu:
            self._logger.log_warning(
                f"Configured number of processes ({self.n_processes}) is higher than CPU count "
                f"({n_cpu}), this is not recommended"
            )

    def _swap(self, array: SolutionType, swap_n: int) -> SolutionType:
        """Given an initial solution, generate a new and unique random solution from it.

        Args:
            array (SolutionType): Initial solution binary array.
            swap_n (int): Number of array elements to swap.

        Returns:
            SolutionType: A new unique solution with `swap_n` elements swapped.
        """
        start_time = time.time()
        while True:
            array = array.copy() if array.ndim == 1 else array.flatten()
            ones = np.argwhere(array == True).flatten()  # noqa: E712
            zeros = np.argwhere(array == False).flatten()  # noqa: E712

            zeros_to_swap = self.rng.choice(zeros, swap_n, replace=False)
            ones_to_swap = self.rng.choice(ones, swap_n, replace=False)
            bits_to_swap = np.hstack((zeros_to_swap, ones_to_swap))

            array[bits_to_swap] = np.logical_not(array[bits_to_swap])
            if not self._solution_exists(array):
                break

            if time.time() - start_time > 60:  # noqa: PLR2004
                raise TimeoutError("Execution exceeded the time limit for generating solutions")
        return array

    def _solution_exists(self, solution: SolutionType) -> bool:
        """Check whether a given solution is unique.

        Args:
            solution (SolutionType): Solution to evaluate.

        Returns:
            bool: Whether this solution has already been evaluated previously.
        """
        key = solution.data.tobytes()
        if key not in self._solutions_dict:
            self._solutions_dict[key] = 1
            return False
        self._solutions_dict[key] += 1
        return True

    def _log_total_skipped(self):
        """Log the amount of duplicated solutions skipped during the optimization process."""
        skipped: int = 0
        total: int = 0
        for n in self._solutions_dict.values():
            total += n
            if n > 1:
                skipped += n - 1
        percent = (skipped / total) * 100
        self._logger.log_debug(f"{skipped}/{total} ({percent:.2f}%) solutions skipped (duplicates)")

    def _get_model_pairs_num(self) -> list[int]:
        """Get the `swap_n` array. If `self.swap` is positive, it'll be in ascending order. If it's
        negative, it'll be in descending order.

        Returns:
            list[int]: The `swap_n` array.
        """
        if self.n_swaps > 0:
            return list(range(1, self.n_swaps + 1))
        if self.n_swaps < 0:
            return list(range(abs(self.n_swaps), 0, -1))
        raise ValueError("Cannot perform zero swaps")

    def _get_no_improve_num_max(self) -> int:
        """Get the max stopping criterion value based on its factor and models/RMs number.

        Returns:
            int: Max stopping criterion value.
        """
        return round(self.stopping_crit_factor * (self.models_num - self.rms_num) * (self.rms_num))

    def _get_no_improve_swap_max(self, swap_n: int) -> int:
        """Get the current iteration stopping criterion.

        Args:
            swap_n (int): Current iteration swap number.

        Returns:
            int: Corresponding stopping criterion according to swap number, the configured behavior,
            and number of processes.
        """
        match self.stopping_crit_behavior:
            case "increase":
                increase = True
            case "decrease":
                increase = False
            case _:  # equal
                increase = None

        if increase is not None:
            if (self.n_swaps > 0 and increase) or (self.n_swaps < 0 and not increase):
                no_improve_swap_max = self.no_improve_num_max / (abs(self.n_swaps) - swap_n + 1)
            else:
                no_improve_swap_max = self.no_improve_num_max / swap_n
        else:
            no_improve_swap_max = self.no_improve_num_max / abs(self.n_swaps)

        if self.n_processes > 1:
            no_improve_swap_max = no_improve_swap_max / self.n_processes

        no_improve_swap_max = round(no_improve_swap_max)
        self._logger.log_debug(
            f"Swap {swap_n} stopping criterion: {no_improve_swap_max} iterations without "
            "improvement"
        )
        return no_improve_swap_max

    def _divide_evenly(self, n: int, div: int) -> list[int]:
        """Divide a integer `n` into `div` almost equal parts.

        Args:
            n (int): Numerator.
            div (int): Denominator.

        Returns:
            list[int]: `div` batches that sum up to `n`.
        """
        return [n // div + (1 if x < n % div else 0) for x in range(div)]

    def _get_max_processes(self) -> int:
        """Based on the current working set and the total available memory at the time, estimate how
        many parallel processes at most could be running.

        Returns:
            int: Estimated maximum multiprocessing level.
        """
        current_memory = int(Process().memory_info().rss)
        available_memory = int(virtual_memory().available)
        max_processes = int(
            available_memory / (current_memory * self._get_memory_increase_factor())
        )
        return max(1, max_processes)

    def _get_memory_increase_factor(self) -> float:
        """Get the approximate percentage in which memory will grow. Larger models tend to use more
        memory, so they should have a greater weight.

        Returns:
            float: Estimated memory grow factor.
        """
        if get_start_method() == "spawn":
            return ((3 * self.models_num / 4) + 998.25) / 999
        return 0.6  # fork() uses Copy-on-Write, memory usage is basically constant

    def _check_available_memory(self, pool: ProcessPoolExecutor) -> Literal[-1, 0, 1]:
        """Check whether the host machine has enough memory to continue the optimization methods.

        Args:
            pool (ProcessPoolExecutor): Current pool executor to abort tasks in case of no memory.

        Raises:
            MemoryError: if memory usage is above 99%.

        Returns:
            Literal[-1, 0, 1]: `-1` if multiprocessing level should decrease, `0` if it should
            remain the same, `1` if it should increase.
        """
        used_percent = float(virtual_memory().percent)
        if used_percent >= 99:  # noqa: PLR2004
            self._logger.log_error("Memory usage is above 99%, aborting!")
            for process in pool._processes.values():
                process.kill()
            pool.shutdown(cancel_futures=True)
            raise MemoryError(
                "No available memory to continue, consider lowering "
                "'optimization_method/n_processes' value"
            )
        if used_percent >= 90:  # noqa: PLR2004
            return -1
        if used_percent <= 50:  # noqa: PLR2004
            return 1
        return 0

    def _check_multiprocessing_scaling(self, pool: ProcessPoolExecutor):
        """Dynamically increase or decrease `n_processes` value based on current host machine
        resources usage.

        Args:
            pool (ProcessPoolExecutor): Current pool executor to abort tasks in case of no memory.
        """
        if self.auto_scaling:
            process_scaling = self._check_available_memory(pool)
            if process_scaling != 0:
                new_value = self.n_processes + process_scaling
                if (new_value > 1) and (new_value < cpu_count()):
                    self.n_processes = new_value
                    self._logger.log_warning(f"Adjusting number of processes to {self.n_processes}")

    def _can_use_global_pool(self, log: bool = False) -> bool:
        """Check whether there'll be sufficient resources to use the memory-hungry global pool for
        better performance.

        Args:
            log (bool, optional): Whether to log a message. Defaults to False.

        Returns:
            bool: Whether the global pool should be used.
        """
        result = cpu_count() < int(self.max_processes / (1 + abs(self.n_swaps)))
        if log:
            self._logger.log_debug(f"Using {'global' if result else 'local'} process pool")
        return result

    @override
    def evolve(self) -> Generator[ResultsType]:
        generator: Generator[ResultsType]
        if self.n_processes > 1:
            generator = self._evolve_multi()
            # Overwrite thread-safe objects by the shared ones
            manager = Manager()
            self._solutions_dict = cast("dict[bytes, int]", cast("object", manager.dict()))
            self._shared_queue = manager.Queue()
        else:
            generator = self._evolve_single()

        yield from generator

        self._log_total_skipped()
        self.solutions.sort(key=lambda x: x[1])
        self.best_solution, self.best_fitness = self.solutions[0]

    def _evolve_single(self) -> Generator[ResultsType]:
        """Run the optimization methods using a single thread.

        Yields:
            ResultsType: Generated results.
        """
        self._logger.log_info("Starting single-threaded pre-optimization")

        best_sol: SolutionType | None = None
        best_of = np.float64("inf")
        for sol_candidate, of_candidate, results in self._temperature():
            yield results
            self.solutions.append((sol_candidate, of_candidate))
            if of_candidate < best_of:
                best_sol = sol_candidate
                best_of = of_candidate
        self._logger.log_info(f"Pre-optimization best fitness: {best_of:.5f}")
        assert best_sol is not None

        self._logger.log_info("Starting single-threaded optimization")
        for swap_n in self._get_model_pairs_num():
            no_improve_swap_max = self._get_no_improve_swap_max(swap_n)
            for sol_candidate, of_candidate, results in self._local_search(
                no_improve_swap_max, swap_n, best_sol, best_of
            ):
                yield results
                self.solutions.append((sol_candidate, of_candidate))
                if of_candidate < best_of:
                    best_sol = sol_candidate
                    best_of = of_candidate

    def _evolve_multi(self) -> Generator[ResultsType]:
        """Run the optimization methods using multiple processes.

        Yields:
            ResultsType: Generated results.
        """

        # The global context is a singleton that will not be copied when using spawn()
        # As a workaround, it's passed as an argument so that its references exists in memory
        global_context: dict[str, Any] | None
        if get_start_method() == "spawn":
            self._logger.log_warning(
                "Consider running on Linux for better multi-processing performance"
            )
            global_context = self.optimization_context.get_global_context()
        else:
            global_context = None

        with ProcessPoolExecutor() as global_pool:
            self._logger.log_info(
                f"Starting pre-optimization using {min(10, self.n_processes)} processes"
            )
            best_solutions = []
            for best, res in self._temperature_multi(global_pool, global_context):
                # During execution, yield intermediary results
                if res is not None:
                    yield res
                # In the end, get the list of best solutions
                elif best is not None:
                    best_solutions = best
            self._logger.log_info(f"Pre-optimization best fitness: {best_solutions[-1][1]:.5f}")

            # Those are then used as initial solutions to local_search()
            self._logger.log_info(f"Starting optimization using {self.n_processes} processes")
            yield from self._local_search_multi(global_pool, best_solutions, global_context)

    def _temperature(
        self, repeat_n: int = 10
    ) -> Generator[tuple[SolutionType, np.float64, ResultsType]]:
        """Run the pre-optimization method using a single thread.

        Args:
            repeat_n (int): Amount of times to repeat the same algorithm.

        Yields:
            tuple[SolutionType, np.float64, ResultsType]: Solution array, objective function value,
            and generated results.
        """
        best_solution = self.optimization_context.generate_solution()
        best_of, _ = self.optimization_context.obj_func(best_solution)
        try:
            for _ in range(repeat_n):
                for swap_n in range(self.rms_num, 0, -1):
                    for __ in range(30):
                        sol_candidate = self._swap(best_solution, swap_n)
                        of_candidate, results = self.optimization_context.obj_func(sol_candidate)
                        yield sol_candidate, of_candidate, results
                        if of_candidate < best_of:
                            best_solution = sol_candidate
                            best_of = of_candidate

        except TimeoutError as ex:
            self._logger.log_debug(str(ex))

    def _temperature_multi(
        self,
        global_pool: ProcessPoolExecutor,
        global_context: dict[str, Any] | None = None,
    ) -> Generator[tuple[list[tuple[SolutionType, np.float64]] | None, ResultsType | None]]:
        """Run the pre-optimization method using multiple processes.

        Args:
            global_pool (ProcessPoolExecutor): Executor to reuse for small models.
            global_context (dict[str, Any], optional): Global context dictionary.

        Yields:
            tuple[list[tuple[SolutionType, np.float64]] | None, ResultsType | None]: List of the
            best solutions and their respective objective function values if optimization has ended,
            otherwise a generated intermediary solution.
        """
        assert self._shared_queue is not None

        # Only spawn at most 10 processes
        n_processes = min(10, self.n_processes)

        # Spawn more seeds using the main thread Generator
        # This ensures that each process will generate different solutions
        seeds = self.rng.spawn(n_processes)
        n_repeats = self._divide_evenly(10, n_processes)
        args: list[tuple[RandomGenerator, int, dict[str, Any] | None]] = []
        for i in range(n_processes):
            args.append((seeds[i], n_repeats[i], global_context))

        # Copy solutions to local variable to avoid copying them when spawning processes
        solutions = self.solutions
        self.solutions = []

        with ProcessPoolExecutor() as local_pool:
            pool = global_pool if self._can_use_global_pool(True) else local_pool
            # Spawn all processes at once
            _ = pool.map(self._process_temperature, args)
            # Consume the shared queue until all processes have finished
            best_solutions: list[tuple[SolutionType, np.float64]] = []
            finished: int = 0
            while finished < n_processes:
                _ = self._check_available_memory(pool)
                sol_candidate, of_candidate, results = self._shared_queue.get()
                if sol_candidate is None:
                    finished += 1
                    self._logger.log_debug(f"{finished} of {n_processes} processes finished")
                elif of_candidate is not None:
                    if (len(best_solutions) < n_processes) or (of_candidate < best_solutions[0][1]):
                        if len(best_solutions) >= n_processes:
                            best_solutions = best_solutions[1:n_processes]
                        best_solutions.append((sol_candidate, of_candidate))
                        best_solutions.sort(key=lambda x: x[1], reverse=True)
                    yield None, results
                    solutions.append((sol_candidate, of_candidate))
            self._check_multiprocessing_scaling(pool)

        self.solutions = solutions  # restore saved solutions to class variable
        yield best_solutions, None

    def _process_temperature(self, args: tuple[RandomGenerator, int, dict[str, Any] | None]):
        """Prepare a new process to run the pre-optimization method.

        Args:
            seed (RandomGenerator): Random generator for the new process.
            repeat_n (int): Amount of times to repeat the same algorithm.
            _global_context (dict[str, Any], optional): Global context dictionary.
        """
        assert self._shared_queue is not None
        seed, repeat_n, _global_context = args
        self.rng = np.random.default_rng(seed=seed)
        for sol_candidate, of_candidate, results in self._temperature(repeat_n):
            self._shared_queue.put((sol_candidate, of_candidate, results))
        self._shared_queue.put((None, None, None))  # indicates that the process has ended

    def _local_search(
        self,
        no_improve_swap_max: int,
        swap_n: int,
        initial_sol: SolutionType,
        initial_of: np.float64,
    ) -> Generator[tuple[SolutionType, np.float64, ResultsType]]:
        """Run the optimization method using a single thread.

        Args:
            no_improve_swap_max (int): Stopping criterion.
            swap_n (int): Current iteration swap number.
            initial_sol (SolutionType): Initial solution.
            initial_of (np.float64): Objective function value for the initial solution.

        Yields:
            tuple[SolutionType, np.float64, ResultsType]: Solution array, objective function value,
            and generated results.
        """
        best_sol = initial_sol
        best_of = initial_of
        no_improve_swap_count = 0

        try:
            while no_improve_swap_count < no_improve_swap_max:
                sol_candidate = self._swap(best_sol, swap_n=swap_n)
                of_candidate, results = self.optimization_context.obj_func(sol_candidate)
                yield sol_candidate, of_candidate, results
                if of_candidate < best_of:
                    best_sol = sol_candidate
                    best_of = of_candidate
                    no_improve_swap_count = 0
                else:
                    no_improve_swap_count += 1
        except TimeoutError as ex:
            self._logger.log_debug(str(ex))

    def _local_search_multi(
        self,
        global_pool: ProcessPoolExecutor,
        initial_solutions: list[tuple[SolutionType, np.float64]],
        global_context: dict[str, Any] | None,
    ) -> Generator[ResultsType]:
        """Run the pre-optimization method using multiple processes.

        Args:
            global_pool (ProcessPoolExecutor): Executor to reuse for small models.
            initial_solutions (list[tuple[SolutionType, np.float64]]): List of the initial solutions
            and their respective objective function values.
            global_context (dict[str, Any], optional): Global context dictionary.

        Yields:
            ResultsType: Generated results.
        """
        assert self._shared_queue is not None

        sol_cut = int(self.n_processes * (1 - self.best_solution_ratio)) + 1
        initial_solutions = initial_solutions[-sol_cut:]
        model_pairs_num = self._get_model_pairs_num()
        best_sol = initial_solutions[-1][0]
        best_of = initial_solutions[-1][1]

        # Copy solutions to local variable to avoid copying them when spawning processes
        solutions = self.solutions
        self.solutions = []

        for swap_n in model_pairs_num:
            # Spawn more seeds using the main thread Generator
            # This ensures that each process will generate different solutions
            seeds = self.rng.spawn(self.n_processes)

            # Consider the initial solutions when spawning new processes
            no_improve_swap_max = self._get_no_improve_swap_max(swap_n)
            args: list[
                tuple[
                    int,
                    int,
                    SolutionType,
                    np.float64,
                    RandomGenerator,
                    dict[str, Any] | None,
                ]
            ] = []
            for i in range(self.n_processes):
                if i + 1 < len(initial_solutions):
                    self._logger.log_debug(
                        f"Spawning process {i + 1} from good solution: "
                        f"{initial_solutions[i][1]:.5f}"
                    )
                    args.append(
                        (
                            no_improve_swap_max,
                            swap_n,
                            initial_solutions[i][0],
                            initial_solutions[i][1],
                            seeds[i],
                            global_context,
                        )
                    )
                else:
                    self._logger.log_debug(
                        f"Spawning process {i + 1} from best solution: {best_of:.5f}"
                    )
                    args.append(
                        (
                            no_improve_swap_max,
                            swap_n,
                            best_sol,
                            best_of,
                            seeds[i],
                            global_context,
                        )
                    )

            with ProcessPoolExecutor() as local_pool:
                pool = global_pool if self._can_use_global_pool() else local_pool
                # Spawn all processes at once
                _ = pool.map(self._process_local_search, args)
                # Consume the shared queue until all processes have finished
                finished: int = 0
                while finished < self.n_processes:
                    _ = self._check_available_memory(pool)
                    sol_candidate, of_candidate, results = self._shared_queue.get()
                    if sol_candidate is None:
                        finished += 1
                        self._logger.log_debug(
                            f"{finished} of {self.n_processes} processes finished"
                        )
                    elif of_candidate is not None and results is not None:
                        yield results
                        solutions.append((sol_candidate, of_candidate))
                        # Find best values to be used in the next iteration
                        if of_candidate < initial_solutions[0][1]:
                            initial_solutions = initial_solutions[1:sol_cut]
                            initial_solutions.append((sol_candidate, of_candidate))
                            initial_solutions.sort(key=lambda x: x[1], reverse=True)
                            if of_candidate < best_of:
                                best_sol = sol_candidate
                                best_of = of_candidate
                self._check_multiprocessing_scaling(pool)

        # Restore saved solutions to class variable
        self.solutions = solutions

    def _process_local_search(
        self,
        args: tuple[int, int, SolutionType, np.float64, RandomGenerator, dict[str, Any] | None],
    ):
        """Prepare a new process to run the optimization method.

        Args (as tuple):
            no_improve_swap_max (int): Stopping criterion.
            swap_n (int): Current iteration swap number.
            initial_sol (SolutionType): Initial solution.
            initial_of (np.float64): Objective function value for the initial solution.
            seed (RandomGenerator): Random generator for the new process.
            _global_context (dict[str, Any], optional): Global context dictionary.
        """
        assert self._shared_queue is not None
        no_improve_swap_max, swap_n, initial_sol, initial_of, seed, _global_context = args
        self.rng = np.random.default_rng(seed=seed)
        for t in self._local_search(no_improve_swap_max, swap_n, initial_sol, initial_of):
            self._shared_queue.put(t)
        self._shared_queue.put((None, None, None))
