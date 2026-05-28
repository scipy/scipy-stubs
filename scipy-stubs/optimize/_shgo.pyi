from collections.abc import Callable, Iterable, Sequence
from typing import Concatenate, Literal, Protocol, TypedDict, type_check_only

import numpy as np
import optype.numpy as onp

from ._constraints import Bounds
from ._optimize import OptimizeResult as _OptimizeResult
from ._typing import Constraints, MinimizerKwargs

__all__ = ["shgo"]

###

type _Float = float | np.float64
type _Float1D = onp.Array1D[np.float64]
type _Fun1D[_RT] = Callable[Concatenate[_Float1D, ...], _RT]

@type_check_only
class _SHGOOptions(TypedDict, total=False):
    f_min: _Float
    f_tol: _Float
    maxiter: int
    maxfev: int
    maxev: int
    mmaxtime: _Float
    minhgrd: int
    symmetry: Sequence[int] | bool
    jac: _Fun1D[onp.ToFloat1D] | bool  # gradient
    hess: _Fun1D[onp.ToFloat2D]
    hessp: Callable[Concatenate[_Float1D, _Float1D, ...], onp.ToFloat1D]
    minimize_every_iter: bool
    local_iter: int
    infty_constraints: bool
    disp: bool

@type_check_only
class _DoesMap(Protocol):
    def __call__[VT, RT](self, func: Callable[[VT], RT], iterable: Iterable[VT], /) -> Iterable[RT]: ...

###

class OptimizeResult(_OptimizeResult):
    x: _Float1D
    xl: Sequence[_Float1D]
    fun: _Float
    funl: Sequence[_Float]
    success: bool
    message: str
    nfev: int
    nlfev: int
    nljev: int  # undocumented
    nlhev: int  # undocumented
    nit: int

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
    sampling_method: Callable[[int, int], onp.ToFloat2D] | Literal["simplicial", "halton", "sobol"] = "simplicial",
    *,
    workers: onp.ToJustInt | _DoesMap = 1,
) -> OptimizeResult: ...
