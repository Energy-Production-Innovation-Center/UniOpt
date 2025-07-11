from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from uniopt.context.enums import VariableType


class Variables:
    ALLOWED_INPUT_TYPE: tuple[type, type, type] = (tuple, list, np.ndarray)

    # TODO: Include lower and upper input
    def __init__(self, number_variables: Any, input_values: Any):
        self.number_variables: int = self.set_n_vars(number_variables)
        self.input_values: NDArray[np.bool] = self.validate_input(input_values)
        self.var_type: VariableType = self.set_variable_type(input_values)

    def set_n_vars(self, n_vars: Any) -> int:
        if type(n_vars) is int and n_vars > 0:
            return n_vars
        raise ValueError(f"Invalid {n_vars}. It should be integer and > 0.")

    def validate_input(self, input_values: Any) -> NDArray[np.bool]:
        if type(input_values) not in self.ALLOWED_INPUT_TYPE:
            raise ValueError(f"The type of variable '{type(input_values)}' is not supported")

        if not isinstance(input_values, np.ndarray):
            input_values = np.array(input_values)

        cast_values = cast("NDArray[np.bool]", input_values)
        if cast_values.size == 0:
            raise ValueError("Empty input value")

        return cast_values

    def set_variable_type(self, input_values: NDArray[np.bool]) -> VariableType:
        var_type = None
        unique_values: NDArray[np.bool] = np.unique(input_values)

        if np.all(np.isin(unique_values, [0, 1])):
            var_type = VariableType.BINARY
        elif np.issubdtype(input_values.dtype, np.integer):
            var_type = VariableType.INTEGER
        elif np.issubdtype(input_values.dtype, np.floating):
            var_type = VariableType.FLOAT
        elif np.issubdtype(input_values.dtype, np.str_):
            var_type = VariableType.STRING
        else:
            raise ValueError("Value in input not supported")

        return var_type

    def correct(self):
        pass

    def generate(self):
        pass
