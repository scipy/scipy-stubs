from collections.abc import Callable
from typing import Concatenate, Literal, TypeAlias

import numpy as np
import optype.numpy as onp
from scipy._typing import ToRNG
from scipy.optimize import OptimizeResult as _OptimizeResult
from ._constraints import Bounds
from ._typing import MinimizerKwargs

__all__ = ["dual_annealing"]

_Float1D: TypeAlias = onp.Array1D[np.float64]

###

class OptimizeResult(_OptimizeResult):
    success: bool
    status: int | bool
    x: _Float1D
    fun: float | int | bool
    nit: int | bool
    nfev: int | bool
    njev: int | bool
    nhev: int | bool
    message: str

def dual_annealing(
    func: Callable[Concatenate[_Float1D, ...], onp.ToFloat],
    bounds: Bounds | tuple[onp.ToFloat1D, onp.ToFloat1D] | onp.ToFloat2D,
    args: tuple[object, ...] = (),
    maxiter: onp.ToJustInt = 1_000,
    minimizer_kwargs: MinimizerKwargs | None = None,
    initial_temp: onp.ToFloat = 5_230.0,
    restart_temp_ratio: onp.ToFloat = 2e-05,
    visit: onp.ToFloat = 2.62,
    accept: onp.ToFloat = -5.0,
    maxfun: onp.ToFloat = 10_000_000.0,
    rng: ToRNG = None,
    no_local_search: onp.ToBool = False,
    callback: Callable[[_Float1D, float | int | bool, Literal[0, 1, 2]], bool | None] | None = None,
    x0: onp.ToFloat1D | None = None,
) -> _OptimizeResult: ...
