from collections.abc import Callable
from typing_extensions import Never

import numpy as np
import optype.numpy as onp
from .base import DenseOutput, OdeSolver

class LSODA(OdeSolver):
    def __init__(
        self,
        /,
        fun: Callable[[float | int | bool, onp.Array1D[np.float64]], onp.Array1D[np.float64]],
        t0: float | int | bool,
        y0: onp.Array1D[np.float64],
        t_bound: float | int | bool,
        first_step: float | int | bool | None = None,
        min_step: float | int | bool = 0.0,
        max_step: float | int | bool = ...,
        rtol: onp.ToFloat | onp.ToFloat1D = 0.001,
        atol: onp.ToFloat | onp.ToFloat1D = 1e-06,
        jac: Callable[[float | int | bool, onp.Array1D[np.float64]], onp.Array2D[np.float64]] | None = None,
        lband: int | bool | None = None,
        uband: int | bool | None = None,
        vectorized: bool = False,
        **extraneous: Never,
    ) -> None: ...

class LsodaDenseOutput(DenseOutput):
    h: float | int | bool
    yh: onp.Array1D[np.float64]
    p: onp.Array1D[np.intp]

    def __init__(
        self,
        /,
        t_old: float | int | bool,
        t: float | int | bool,
        h: float | int | bool,
        order: int | bool,
        yh: onp.Array1D[np.float64],
    ) -> None: ...
