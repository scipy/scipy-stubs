from typing import Final, Literal, SupportsIndex, TypeAlias, type_check_only

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.optimize import OptimizeResult
from scipy.optimize._typing import Bound
from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

_Int1D: TypeAlias = onp.Array1D[np.intp]

_Float: TypeAlias = float | np.float64
_Float1D: TypeAlias = onp.Array1D[np.float64]

_ToBounds: TypeAlias = tuple[float | onp.ToFloat1D, float | onp.ToFloat1D] | Bound
_TerminationStatus: TypeAlias = Literal[-1, 0, 1, 2, 3]

@type_check_only
class _OptimizeResult(OptimizeResult):
    x: _Float1D
    fun: _Float1D
    const: _Float
    optimality: _Float
    active_mask: _Int1D
    unbounded_sol: tuple[float | onp.ArrayND[npc.number], ...]
    nit: int
    status: _TerminationStatus
    message: str
    success: bool

###

TERMINATION_MESSAGES: Final[dict[_TerminationStatus, str]] = ...

def lsq_linear(
    A: onp.ToFloat2D | _spbase | LinearOperator,
    b: onp.ToFloat1D,
    bounds: _ToBounds = ...,  # (inf, inf)
    method: Literal["trf", "bvls"] = "trf",
    tol: float = 1e-10,
    lsq_solver: Literal["exact", "lsmr"] | None = None,
    lsmr_tol: float | Literal["auto"] | None = None,
    max_iter: int | None = None,
    verbose: Literal[0, 1, 2] = 0,
    *,
    lsmr_maxiter: int | None = None,
) -> _OptimizeResult: ...

# undocumented
def prepare_bounds(bounds: _ToBounds, n: SupportsIndex) -> tuple[_Float, _Float] | tuple[_Float1D, _Float1D]: ...
