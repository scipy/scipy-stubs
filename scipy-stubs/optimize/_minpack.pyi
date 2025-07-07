from collections.abc import Callable
from typing import Any

import numpy as np
import optype.numpy as onp

def _chkder(
    m: int,
    n: int,
    x: onp.Array1D[np.float64],
    fvec: onp.Array1D[np.float64],
    fjac: onp.Array2D[np.float64],
    ldfjac: int,
    xp: onp.Array1D[np.float64],
    fvecp: onp.Array1D[np.float64],
    mode: int,
    err: float,
) -> tuple[int, onp.Array1D[np.float64]]: ...  # (good, err)

#
def _hybrd(
    fun: Callable[..., onp.Array1D[np.float64]],
    x0: onp.Array1D[np.float64],
    args: tuple[object, ...],
    full_output: bool,
    xtol: float,
    maxfev: int,
    ml: int,
    mu: int,
    epsfcn: float,
    factor: float,
    diag: onp.Array1D[np.float64],
) -> tuple[onp.Array1D[np.float64], dict[str, Any], int]: ...
def _hybrj(
    fun: Callable[..., onp.Array1D[np.float64]],
    Dfun: Callable[..., onp.Array1D[np.float64]],
    x0: onp.Array1D[np.float64],
    args: tuple[object, ...],
    full_output: bool,
    col_deriv: bool,
    xtol: float,
    maxfev: int,
    factor: float,
    diag: onp.Array1D[np.float64],
) -> tuple[onp.Array1D[np.float64], dict[str, Any], int]: ...
def _lmdif(
    fun: Callable[..., onp.Array1D[np.float64]],
    x0: onp.Array1D[np.float64],
    args: tuple[object, ...],
    full_output: bool,
    ftol: float,
    xtol: float,
    gtol: float,
    maxfev: int,
    epsfcn: float,
    factor: float,
    diag: onp.Array1D[np.float64],
) -> tuple[onp.Array1D[np.float64], dict[str, Any], int]: ...
def _lmder(
    fun: Callable[..., onp.Array1D[np.float64]],
    Dfun: Callable[..., onp.Array1D[np.float64]],
    x0: onp.Array1D[np.float64],
    args: tuple[object, ...],
    full_output: bool,
    col_deriv: bool,
    ftol: float,
    xtol: float,
    gtol: float,
    maxfev: int,
    factor: float,
    diag: onp.Array1D[np.float64],
) -> tuple[onp.Array1D[np.float64], dict[str, Any], int]: ...
