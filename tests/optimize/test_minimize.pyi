from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from scipy.optimize import minimize

NDArrayFloat: TypeAlias = npt.NDArray[np.float16 | np.float32 | np.float64]

def objective_function(CCT: NDArrayFloat, uv_: NDArrayFloat) -> np.float64: ...

uv = np.atleast_1d([0.1978, 0.3124])

minimize(objective_function, x0=6400, args=(uv,), method="Nelder-Mead", options={"fatol": 1e-10})

# https://github.com/scipy/scipy-stubs/issues/635
minimize(lambda x: x**2, 0.0)  # type: ignore[misc]
