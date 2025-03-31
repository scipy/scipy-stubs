from collections.abc import Callable, Iterable, Sequence
from typing import Concatenate, Literal, TypeAlias, TypedDict, TypeVar, type_check_only

import numpy as np
import optype.numpy as onp
from ._constraints import Bounds
from ._optimize import OptimizeResult as _OptimizeResult
from ._typing import Constraints, MinimizerKwargs

__all__ = ["shgo"]

_VT = TypeVar("_VT")
_RT = TypeVar("_RT")

_Float: TypeAlias = float | int | bool | np.float64
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Fun1D: TypeAlias = Callable[Concatenate[_Float1D, ...], _RT]

@type_check_only
class _SHGOOptions(TypedDict, total=False):
    f_min: _Float
    f_tol: _Float
    maxiter: int | bool
    maxfev: int | bool
    maxev: int | bool
    mmaxtime: _Float
    minhgrd: int | bool
    symmetry: Sequence[int | bool] | onp.ToBool
    jac: _Fun1D[onp.ToFloat1D] | onp.ToBool  # gradient
    hess: _Fun1D[onp.ToFloat2D]
    hessp: Callable[Concatenate[_Float1D, _Float1D, ...], onp.ToFloat1D]
    minimize_every_iter: onp.ToBool
    local_iter: int | bool
    infty_constraints: onp.ToBool
    disp: onp.ToBool

###

class OptimizeResult(_OptimizeResult):
    x: _Float1D
    xl: Sequence[_Float1D]
    fun: _Float
    funl: Sequence[_Float]
    success: bool
    message: str
    nfev: int | bool
    nlfev: int | bool
    nljev: int | bool  # undocumented
    nlhev: int | bool  # undocumented
    nit: int | bool

def shgo(
    func: _Fun1D[onp.ToFloat],
    bounds: tuple[onp.ToFloat | onp.ToFloat1D, onp.ToFloat | onp.ToFloat1D] | Bounds,
    args: tuple[object, ...] = (),
    constraints: Constraints | None = None,
    n: onp.ToJustInt = 100,
    iters: onp.ToJustInt = 1,
    callback: Callable[[_Float1D], None] | None = None,
    minimizer_kwargs: MinimizerKwargs | None = None,
    options: _SHGOOptions | None = None,
    sampling_method: Callable[[int | bool, int | bool], onp.ToFloat2D] | Literal["simplicial", "halton", "sobol"] = "simplicial",
    *,
    workers: onp.ToJustInt | Callable[[Callable[[_VT], _RT], Iterable[_VT]], Sequence[_RT]] = 1,
) -> OptimizeResult: ...
