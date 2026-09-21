from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, get_start_method
from typing import Literal

from psutil import Process, virtual_memory


def divide_evenly(n: int, div: int) -> list[int]:
    """Divide a integer `n` into `div` almost equal parts.

    Args:
        n (int): Numerator.
        div (int): Denominator.

    Returns:
        list[int]: `div` batches that sum up to `n`.
    """
    return [n // div + (1 if x < n % div else 0) for x in range(div)]


def get_max_processes() -> int:
    """Based on the current working set and the total available memory at the time, estimate how
    many parallel processes at most could be running.

    Returns:
        int: Estimated maximum multiprocessing level.
    """
    current_memory = int(Process().memory_info().rss)
    available_memory = int(virtual_memory().available)
    max_processes = int(available_memory / (current_memory * get_memory_increase_factor()))
    return max(1, max_processes)


def get_memory_increase_factor(models_num) -> float:
    """Get the approximate percentage in which memory will grow. Larger models tend to use more
    memory, so they should have a greater weight.

    Returns:
        float: Estimated memory grow factor.
    """
    if get_start_method() == "spawn":
        return ((3 * models_num / 4) + 998.25) / 999
    return 0.6  # fork() uses Copy-on-Write, memory usage is basically constant


def check_available_memory(_logger, pool: ProcessPoolExecutor) -> Literal[-1, 0, 1]:
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
        _logger.log_error("Memory usage is above 99%, aborting!")
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


def check_multiprocessing_scaling(auto_scaling, n_processes, _logger, pool: ProcessPoolExecutor):
    """Dynamically increase or decrease `n_processes` value based on current host machine
    resources usage.

    Args:
        pool (ProcessPoolExecutor): Current pool executor to abort tasks in case of no memory.
    """
    if auto_scaling:
        process_scaling = check_available_memory(pool)
        if process_scaling != 0:
            new_value = n_processes + process_scaling
            if (new_value > 1) and (new_value < cpu_count()):
                n_processes = new_value
                _logger.log_warning(f"Adjusting number of processes to {n_processes}")


def can_use_global_pool(max_processes, n_swaps, _logger, log: bool = False) -> bool:
    """Check whether there'll be sufficient resources to use the memory-hungry global pool for
    better performance.

    Args:
        log (bool, optional): Whether to log a message. Defaults to False.

    Returns:
        bool: Whether the global pool should be used.
    """
    result = cpu_count() < int(max_processes / (1 + abs(n_swaps)))
    if log:
        _logger.log_debug(f"Using {'global' if result else 'local'} process pool")
    return result
