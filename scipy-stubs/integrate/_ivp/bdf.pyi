from collections.abc import Callable
from typing import Any, Final, Generic, TypeAlias
from typing_extensions import Never, TypeVar

import numpy as np
import optype.numpy as onp
from scipy.sparse import sparray, spmatrix
from .base import DenseOutput, OdeSolver

###

_SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.inexact[Any], default=np.float64 | np.complex128)

_LU: TypeAlias = tuple[onp.ArrayND[np.inexact[Any]], onp.ArrayND[np.integer[Any]]]
_FuncLU: TypeAlias = Callable[[onp.ArrayND[np.float64]], _LU] | Callable[[onp.ArrayND[np.complex128]], _LU]
_FuncSolveLU: TypeAlias = Callable[[_LU, onp.ArrayND], onp.ArrayND[np.inexact[Any]]]

_ToJac: TypeAlias = onp.ToComplex2D | spmatrix | sparray

###

MAX_ORDER: Final = 5
NEWTON_MAXITER: Final = 4
MIN_FACTOR: Final = 0.2
MAX_FACTOR: Final = 10

class BDF(OdeSolver, Generic[_SCT_co]):
    max_step: float | int | bool
    h_abs: float | int | bool
    h_abs_old: float | int | bool | None
    error_norm_old: None
    newton_tol: float | int | bool
    jac_factor: onp.ArrayND[np.float64] | None  # 1d

    LU: _LU
    lu: _FuncLU
    solve_lu: _FuncSolveLU

    I: onp.ArrayND[_SCT_co]
    error_const: onp.ArrayND[np.float64]
    gamma: onp.ArrayND[np.float64]
    alpha: onp.ArrayND[np.float64]
    D: onp.ArrayND[np.float64]
    order: int | bool
    n_equal_steps: int | bool

    def __init__(
        self,
        /,
        fun: Callable[[float | int | bool, onp.Array1D[_SCT_co]], onp.ToComplex1D],
        t0: onp.ToFloat,
        y0: onp.Array1D[_SCT_co] | onp.ToComplexND,
        t_bound: onp.ToFloat,
        max_step: onp.ToFloat = ...,
        rtol: onp.ToFloat = 0.001,
        atol: onp.ToFloat = 1e-06,
        jac: _ToJac | Callable[[float | int | bool, onp.ArrayND[_SCT_co]], _ToJac] | None = None,
        jac_sparsity: _ToJac | None = None,
        vectorized: bool = False,
        first_step: onp.ToFloat | None = None,
        **extraneous: Never,
    ) -> None: ...

class BdfDenseOutput(DenseOutput):
    order: int | bool
    t_shift: onp.ArrayND[np.float64]
    denom: onp.ArrayND[np.float64]
    D: onp.ArrayND[np.float64]
    def __init__(
        self,
        /,
        t_old: float | int | bool,
        t: float | int | bool,
        h: float | int | bool,
        order: int | bool,
        D: onp.ArrayND[np.float64],
    ) -> None: ...

def compute_R(order: int | bool, factor: float | int | bool) -> onp.ArrayND[np.float64]: ...
def change_D(D: onp.ArrayND[np.float64], order: int | bool, factor: float | int | bool) -> None: ...
def solve_bdf_system(
    fun: Callable[[float | int | bool, onp.ArrayND[_SCT_co]], onp.ToComplex1D],
    t_new: onp.ToFloat,
    y_predict: onp.ArrayND[_SCT_co],
    c: float | int | bool,
    psi: onp.ArrayND[np.float64],
    LU: _FuncLU,
    solve_lu: _FuncSolveLU,
    scale: onp.ArrayND[np.float64],
    tol: float | int | bool,
) -> tuple[bool, int | bool, onp.ArrayND[_SCT_co], onp.ArrayND[_SCT_co]]: ...
