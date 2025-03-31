from typing import NamedTuple

import numpy as np

class ConfidenceInterval(NamedTuple):
    low: float | int | bool | np.float64
    high: float | int | bool | np.float64
