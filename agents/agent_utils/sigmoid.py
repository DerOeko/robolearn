import numpy as np
from numpy import exp
from typing import Optional
from numpy.typing import NDArray

def sigmoid(go_value: float, nogo_value: float, temperature: Optional[float] = 1.0) -> NDArray[np.float64]:
    """
    Computes the action probabilities for go and nogo values.

    :param go_value: The value associated with the 'go' action.
    :param nogo_value: The value associated with the 'nogo' action.
    :param temperature: The temperature parameter for softmax scaling.
    :return: An array of probabilities for the 'go' and 'nogo' actions.
    """
    if temperature is None:
        temperature = 1.0

    scaled_diff = go_value - nogo_value

    if temperature is not None:
        scaled_diff *= temperature

    p1 = 1 / (1 + exp(-scaled_diff))
    p2 = 1 - p1
    return np.array([p1, p2], dtype=np.float64)