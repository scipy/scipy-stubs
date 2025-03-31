# mypy: disable-error-code="override"

import abc
from collections.abc import Callable
from typing import Final, Generic, Literal, Protocol, TypeAlias, TypedDict, overload, type_check_only
from typing_extensions import TypeVar, Unpack, override

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

__all__ = [
    "BroydenFirst",
    "InverseJacobian",
    "KrylovJacobian",
    "NoConvergence",
    "anderson",
    "broyden1",
    "broyden2",
    "diagbroyden",
    "excitingmixing",
    "linearmixing",
    "newton_krylov",
]

_Floating: TypeAlias = np.float32 | np.float64
_Inexact: TypeAlias = _Floating | np.complex64 | np.complex128
_Inexact1D: TypeAlias = onp.Array1D[_Inexact]
_InexactND: TypeAlias = onp.ArrayND[_Inexact]

_JacobianMethod: TypeAlias = Literal[
    "krylov",
    "broyden1",
    "broyden2",
    "anderson",
    "diagbroyden",
    "linearmixing",
    "excitingmixing",
]
_KrylovMethod: TypeAlias = Literal["lgmres", "gmres", "bicgstab", "cgs", "minres", "tfqmr"]
_ReductionMethod: TypeAlias = Literal["restart", "simple", "svd"]
_LineSearch: TypeAlias = Literal["armijo", "wolfe"]

_Callback: TypeAlias = (
    Callable[[onp.ArrayND[np.float64], np.float64], None] | Callable[[onp.ArrayND[np.complex128], np.float64], None]
)
_ResidFunc: TypeAlias = Callable[[onp.ArrayND[np.float64]], onp.ToFloat] | Callable[[onp.ArrayND[np.complex128]], onp.ToFloat]

_InexactT = TypeVar("_InexactT", bound=_Inexact, default=_Inexact)
_InexactT_co = TypeVar("_InexactT_co", bound=_Inexact, default=_Inexact, covariant=True)

_JacobianLike: TypeAlias = (
    Jacobian[_InexactT]
    | type[Jacobian[_InexactT]]
    | onp.ArrayND[_InexactT]
    | _spbase[_InexactT]
    | _SupportsJacobian[_InexactT]
    | Callable[[onp.Array1D[np.float64]], onp.ArrayND[_InexactT] | _spbase[_InexactT]]
    | Callable[[onp.Array1D[np.complex128]], onp.ArrayND[_InexactT] | _spbase[_InexactT]]
)

@type_check_only
class _SupportsJacobian(Protocol[_InexactT_co]):
    @property
    def shape(self, /) -> tuple[int | bool, int | bool]: ...
    @property
    def dtype(self, /) -> np.dtype[_InexactT_co]: ...
    def solve(self, v: _InexactND, /, tol: float | int | bool = 0) -> onp.ToComplex2D: ...

@type_check_only
class _JacobianKwargs(TypedDict, Generic[_InexactT_co], total=False):
    solve: Callable[[_InexactND], onp.Array2D[_InexactT_co]] | Callable[[_InexactND, onp.ToFloat], onp.Array2D[_InexactT_co]]
    rsolve: Callable[[_InexactND], onp.Array2D[_InexactT_co]] | Callable[[_InexactND, onp.ToFloat], onp.Array2D[_InexactT_co]]
    matvec: Callable[[_InexactND], onp.Array1D[_InexactT_co]] | Callable[[_InexactND, onp.ToFloat], onp.Array2D[_InexactT_co]]
    rmatvec: Callable[[_InexactND], onp.Array1D[_InexactT_co]] | Callable[[_InexactND, onp.ToFloat], onp.Array1D[_InexactT_co]]
    matmat: Callable[[_InexactND], onp.Array2D[_InexactT_co]]
    update: Callable[[_InexactND, onp.ArrayND[_InexactT_co]], None]
    todense: Callable[[], onp.Array2D[_InexactT_co]]
    shape: tuple[int | bool, int | bool]
    dtype: np.dtype[_InexactT_co]

#
@type_check_only
class _NonlinInfoDict(TypedDict):
    fun: float | int | bool | np.float64
    nit: int | bool
    status: int | bool
    success: bool
    message: str

###

class NoConvergence(Exception): ...

class TerminationCondition:
    x_tol: Final[float | int | bool]
    x_rtol: Final[float | int | bool]
    f_tol: Final[float | int | bool]
    f_rtol: Final[float | int | bool]
    iter: Final[int | bool | None]
    norm: Final[Callable[[_InexactND], float | int | bool | np.float64]]

    f0_norm: float | int | bool | None
    iteration: int | bool

    def __init__(
        self,
        /,
        f_tol: float | int | bool | None = None,
        f_rtol: float | int | bool | None = None,
        x_tol: float | int | bool | None = None,
        x_rtol: float | int | bool | None = None,
        iter: int | bool | None = None,
        norm: Callable[[_InexactND], float | int | bool | np.float64] = ...,
    ) -> None: ...
    def check(self, /, f: _InexactND, x: _InexactND, dx: _InexactND) -> int | bool: ...

class Jacobian(Generic[_InexactT_co]):  # undocumented
    shape: Final[tuple[int | bool, int | bool]]
    dtype: np.dtype[_InexactT_co]
    func: Final[_ResidFunc]

    def __init__(self, /, **kw: Unpack[_JacobianKwargs[_InexactT_co]]) -> None: ...
    #
    @abc.abstractmethod
    def solve(self, /, v: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...
    # `x` and `F` are 1-d
    def setup(self: Jacobian[_InexactT], /, x: _InexactND, F: onp.ArrayND[_InexactT], func: _ResidFunc) -> None: ...
    def update(self: Jacobian[_InexactT], /, x: _InexactND, F: onp.ArrayND[_InexactT]) -> None: ...  # does nothing
    def aspreconditioner(self, /) -> InverseJacobian: ...

class InverseJacobian(Generic[_InexactT_co]):
    jacobian: Jacobian[_InexactT_co]
    matvec: Callable[[_InexactND], onp.Array1D[_InexactT_co]] | Callable[[_InexactND, onp.ToFloat], onp.Array1D[_InexactT_co]]
    rmatvec: Callable[[_InexactND], onp.Array1D[_InexactT_co]] | Callable[[_InexactND, onp.ToFloat], onp.Array1D[_InexactT_co]]

    @property
    def shape(self, /) -> tuple[int | bool, int | bool]: ...
    @property
    def dtype(self, /) -> np.dtype[_InexactT_co]: ...
    #
    def __init__(self, /, jacobian: Jacobian[_InexactT_co]) -> None: ...

class GenericBroyden(Jacobian[_InexactT_co], Generic[_InexactT_co], metaclass=abc.ABCMeta):
    alpha: Final[float | int | bool | None]
    last_x: _Inexact1D
    last_f: float | int | bool

    @override
    def setup(self, /, x0: _InexactND, f0: _InexactND, func: _ResidFunc) -> None: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def update(self, /, x0: _InexactND, f0: _InexactND) -> None: ...  # pyright: ignore[reportIncompatibleMethodOverride]

class LowRankMatrix(Generic[_InexactT_co]):
    dtype: np.dtype[_InexactT_co]
    alpha: Final[float | int | bool]
    n: Final[int | bool]
    cs: Final[list[_InexactND]]
    ds: Final[list[_InexactND]]
    collapsed: _InexactND | None

    def __init__(self, /, alpha: float | int | bool, n: int | bool, dtype: np.dtype[_InexactT_co]) -> None: ...
    def __array__(self, /, dtype: None = None, copy: None = None) -> onp.Array2D[_InexactT_co]: ...
    def solve(self, /, v: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...
    def rsolve(self, /, v: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...
    def matvec(self, /, v: _InexactND) -> onp.Array1D[_InexactT_co]: ...
    def rmatvec(self, /, v: _InexactND) -> onp.Array1D[_InexactT_co]: ...
    def append(self, /, c: _InexactND, d: _InexactND) -> None: ...
    def collapse(self, /) -> None: ...
    def restart_reduce(self, /, rank: int | bool) -> None: ...
    def simple_reduce(self, /, rank: int | bool) -> None: ...
    def svd_reduce(self, /, max_rank: int | bool, to_retain: int | bool | None = None) -> None: ...

class BroydenFirst(GenericBroyden[_InexactT_co], Generic[_InexactT_co]):
    max_rank: Final[int | bool]
    Gm: LowRankMatrix[_InexactT_co] | None

    def __init__(
        self,
        /,
        alpha: float | int | bool | None = None,
        reduction_method: _ReductionMethod = "restart",
        max_rank: int | bool | None = None,
    ) -> None: ...
    @override
    def solve(self, /, f: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...
    def matvec(self, /, f: _InexactND) -> onp.Array1D[_InexactT_co]: ...
    def rmatvec(self, /, f: _InexactND) -> onp.Array1D[_InexactT_co]: ...
    def todense(self, /) -> onp.Array2D[_InexactT_co]: ...

class BroydenSecond(BroydenFirst[_InexactT_co], Generic[_InexactT_co]): ...

class Anderson(GenericBroyden[_InexactT_co], Generic[_InexactT_co]):
    w0: Final[float | int | bool]
    M: Final[float | int | bool]
    dx: list[onp.Array1D[_InexactT_co]]
    df: list[onp.Array1D[_InexactT_co]]
    gamma: onp.ArrayND[_InexactT_co] | None

    def __init__(
        self, /, alpha: float | int | bool | None = None, w0: float | int | bool = 0.01, M: float | int | bool = 5
    ) -> None: ...
    @override
    def solve(self, /, f: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    def matvec(self, /, f: _InexactND) -> onp.Array1D[_InexactT_co]: ...

class DiagBroyden(GenericBroyden[_InexactT_co], Generic[_InexactT_co]):
    d: onp.Array1D[_InexactT_co]

    def __init__(self, /, alpha: float | int | bool | None = None) -> None: ...
    @override
    def solve(self, /, f: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...
    def matvec(self, /, f: _InexactND) -> onp.Array1D[_InexactT_co]: ...
    def rmatvec(self, /, f: _InexactND) -> onp.Array1D[_InexactT_co]: ...
    def todense(self, /) -> onp.Array2D[_InexactT_co]: ...

class LinearMixing(GenericBroyden[_InexactT_co], Generic[_InexactT_co]):
    def __init__(self, /, alpha: float | int | bool | None = None) -> None: ...
    @override
    def solve(self, /, f: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...
    def matvec(self, /, f: _InexactND) -> onp.Array1D[_InexactT_co]: ...
    def rmatvec(self, /, f: _InexactND) -> onp.Array1D[_InexactT_co]: ...
    def todense(self, /) -> onp.Array2D[_InexactT_co]: ...

class ExcitingMixing(GenericBroyden[_InexactT_co], Generic[_InexactT_co]):
    alphamax: Final[float | int | bool]
    beta: onp.Array1D[_InexactT_co] | None

    def __init__(self, /, alpha: float | int | bool | None = None, alphamax: float | int | bool = 1.0) -> None: ...
    @override
    def solve(self, /, f: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...
    def matvec(self, /, f: _InexactND) -> onp.Array1D[_InexactT_co]: ...
    def rmatvec(self, /, f: _InexactND) -> onp.Array1D[_InexactT_co]: ...
    def todense(self, /) -> onp.Array2D[_InexactT_co]: ...

class KrylovJacobian(Jacobian[_InexactT_co], Generic[_InexactT_co]):
    rdiff: Final[float | int | bool]
    method: Final[_KrylovMethod]
    method_kw: Final[dict[str, object]]
    preconditioner: LinearOperator | InverseJacobian | None
    x0: _Inexact1D
    f0: _Inexact1D
    op: LinearOperator

    def __init__(
        self,
        /,
        rdiff: float | int | bool | None = None,
        method: _KrylovMethod = "lgmres",
        inner_maxiter: int | bool = 20,
        inner_M: LinearOperator | InverseJacobian | None = None,
        outer_k: int | bool = 10,
        **kw: object,
    ) -> None: ...
    def matvec(self, /, v: _InexactND) -> onp.Array2D[_InexactT_co]: ...
    @override
    def solve(self, /, rhs: _InexactND, tol: float | int | bool = 0) -> onp.Array2D[_InexactT_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def update(self, /, x: _InexactND, f: _InexactND) -> None: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def setup(self, /, x: _InexactND, f: _InexactND, func: _ResidFunc) -> None: ...  # pyright: ignore[reportIncompatibleMethodOverride]

# undocumented
@overload
def asjacobian(J: _JacobianLike[_InexactT]) -> Jacobian[_InexactT]: ...
@overload
def asjacobian(J: _JacobianMethod) -> Jacobian: ...

#
def maxnorm(x: onp.ToComplexND) -> float | int | bool | np.float64: ...  # undocumented

#
@overload
def nonlin_solve(
    F: _ResidFunc,
    x0: onp.ToComplexND,
    jacobian: _JacobianMethod | _JacobianLike = "krylov",
    iter: onp.ToInt | None = None,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch = "armijo",
    callback: _Callback | None = None,
    full_output: Falsy = False,
    raise_exception: op.CanBool = True,
) -> _InexactND: ...
@overload
def nonlin_solve(
    F: _ResidFunc,
    x0: onp.ToComplexND,
    jacobian: _JacobianMethod | _JacobianLike = "krylov",
    iter: onp.ToInt | None = None,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch = "armijo",
    callback: _Callback | None = None,
    *,
    full_output: Truthy,
    raise_exception: op.CanBool = True,
) -> tuple[_InexactND, _NonlinInfoDict]: ...

#
def broyden1(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    reduction_method: _ReductionMethod = "restart",
    max_rank: onp.ToInt | None = None,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def broyden2(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    reduction_method: _ReductionMethod = "restart",
    max_rank: onp.ToInt | None = None,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def anderson(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    w0: onp.ToFloat = 0.01,
    M: onp.ToInt = 5,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def linearmixing(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    verbose: op.CanBool = False,
    maxiter: int | bool | None = None,
    f_tol: onp.ToInt | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def diagbroyden(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def excitingmixing(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    alphamax: onp.ToFloat = 1.0,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def newton_krylov(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    rdiff: onp.ToFloat | None = None,
    method: _KrylovMethod = "lgmres",
    inner_maxiter: onp.ToInt = 20,
    inner_M: LinearOperator | InverseJacobian | None = None,
    outer_k: onp.ToInt = 10,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
