import numpy as np
from numpy.typing import NDArray

SolutionType = NDArray[np.bool_]
ResultsType = dict[str, list[int] | float | np.float64 | np.int64]
