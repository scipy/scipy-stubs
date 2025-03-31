from collections.abc import Callable
from typing import Any, Concatenate, Final, Generic, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
from ._optimize import OptimizeResult
from ._typing import MethodRootScalar

__all__ = ["RootResults", "bisect", "brenth", "brentq", "newton", "ridder", "toms748"]

_Flag: TypeAlias = Literal["converged", "sign error", "convergence error", "value error", "No error"]
_FlagKey: TypeAlias = Literal[0, -1, -2, -3, -4, 1]

_Float: TypeAlias = float | int | bool | np.float64
_Floating: TypeAlias = float | int | bool | np.floating[Any]

_T = TypeVar("_T")
_KT = TypeVar("_KT", bound=_FlagKey)
_RT = TypeVar("_RT", bound=_Floating)
_RT_co = TypeVar("_RT_co", bound=_Floating, default=_Float, covariant=True)
_ToFloatT = TypeVar("_ToFloatT", bound=onp.ToFloat | onp.ToFloatND, default=onp.ToFloat)

_Fun0D: TypeAlias = (
    Callable[Concatenate[float | int | bool, ...], onp.ToFloat] | Callable[Concatenate[np.float64, ...], onp.ToFloat]
)
_Fun1D: TypeAlias = Callable[Concatenate[onp.Array1D[np.float64], ...], _ToFloatT]

_State: TypeAlias = tuple[_FlagKey, _Float]
_Bracket: TypeAlias = tuple[_Float, _Float]

###

CONVERGED: Final = "converged"  # 0  # undocumented
SIGNERR: Final = "sign error"  # -1  # undocumented
CONVERR: Final = "convergence error"  # -2  # undocumented
VALUEERR: Final = "value error"  # -3  # undocumented
INPROGRESS: Final = "No error"  # 1  # undocumented

flag_map: Final[dict[_FlagKey, _Flag]] = ...  # undocumented

class RootResults(OptimizeResult, Generic[_RT_co]):
    root: _RT_co  # readonly
    iterations: Final[int | bool]
    function_calls: Final[int | bool]
    converged: Final[bool]
    flag: Final[_Flag]
    method: Final[MethodRootScalar]

    def __init__(
        self,
        /,
        root: _RT_co,
        iterations: int | bool,
        function_calls: int | bool,
        flag: _FlagKey,
        method: MethodRootScalar,
    ) -> None: ...

# undocumented
class TOMS748Solver:
    f: _Fun0D | None
    args: tuple[object, ...] | None
    function_calls: int | bool
    iterations: int | bool
    k: int | bool
    ab: list[_Float]  # size  2
    fab: list[_Float]  # size 2
    d: _Float | None
    fd: _Float | None
    e: _Float | None
    fe: _Float | None
    disp: bool
    xtol: _Float
    rtol: _Float
    maxiter: int | bool

    def __init__(self, /) -> None: ...
    def configure(self, /, xtol: _Float, rtol: _Float, maxiter: int | bool, disp: bool, k: int | bool) -> None: ...
    def _callf(self, /, x: _Float, error: bool = True) -> onp.ToFloat: ...
    @overload
    def get_result(self, /, x: _T, flag: Literal[0] = 0) -> tuple[_T, int | bool, int | bool, Literal[0]]: ...
    @overload
    def get_result(self, /, x: _T, flag: _KT) -> tuple[_T, int | bool, int | bool, _KT]: ...
    def _update_bracket(self, /, c: _Float, fc: _Float) -> _Bracket: ...
    def start(self, /, f: _Fun0D, a: _Float, b: _Float, args: tuple[object, ...] = ()) -> _State: ...
    def get_status(self, /) -> _State: ...
    def iterate(self, /) -> _State: ...
    def solve(
        self,
        /,
        f: _Fun0D,
        a: _Float,
        b: _Float,
        args: tuple[object, ...] = (),
        xtol: _Float = 2e-12,
        rtol: _Float = ...,
        k: int | bool = 2,
        maxiter: int | bool = 100,
        disp: op.CanBool = True,
    ) -> _State: ...

# undocumented
def _update_bracket(ab: list[_Float] | _Bracket, fab: list[_Float] | _Bracket, c: _Float, fc: _Float) -> _Bracket: ...

# undocumented
@overload
def results_c(full_output: Falsy, r: _T, method: MethodRootScalar) -> _T: ...
@overload
def results_c(
    full_output: Truthy,
    r: tuple[_RT, int | bool, int | bool, _FlagKey],
    method: MethodRootScalar,
) -> tuple[_RT, RootResults[_RT]]: ...

#
@overload
def newton(
    func: _Fun0D,
    x0: onp.ToFloat,
    fprime: _Fun0D | None = None,
    args: tuple[object, ...] = (),
    tol: onp.ToFloat = 1.48e-08,
    maxiter: onp.ToJustInt = 50,
    fprime2: _Fun0D | None = None,
    x1: onp.ToFloat | None = None,
    rtol: onp.ToFloat = 0.0,
    full_output: Falsy = False,
    disp: onp.ToBool = True,
) -> _Float: ...
@overload
def newton(
    func: _Fun0D,
    x0: onp.ToFloat,
    fprime: _Fun0D | None = None,
    args: tuple[object, ...] = (),
    tol: onp.ToFloat = 1.48e-08,
    maxiter: onp.ToJustInt = 50,
    fprime2: _Fun0D | None = None,
    x1: onp.ToFloat | None = None,
    rtol: onp.ToFloat = 0.0,
    *,
    full_output: Truthy,
    disp: onp.ToBool = True,
) -> tuple[_Float, RootResults[_Float]]: ...
@overload
def newton(
    func: _Fun1D,
    x0: onp.ToFloat1D,
    fprime: _Fun1D[onp.ToFloat1D] | None = None,
    args: tuple[object, ...] = (),
    tol: onp.ToFloat = 1.48e-08,
    maxiter: onp.ToJustInt = 50,
    fprime2: _Fun1D[onp.ToFloat2D] | None = None,
    x1: onp.ToFloat1D | None = None,
    rtol: onp.ToFloat = 0.0,
    full_output: Falsy = False,
    disp: onp.ToBool = True,
) -> onp.Array1D[np.float64]: ...
@overload
def newton(
    func: _Fun1D,
    x0: onp.ToFloat1D,
    fprime: _Fun1D[onp.ToFloat1D] | None = None,
    args: tuple[object, ...] = (),
    tol: onp.ToFloat = 1.48e-08,
    maxiter: onp.ToJustInt = 50,
    fprime2: _Fun1D[onp.ToFloat2D] | None = None,
    x1: onp.ToFloat1D | None = None,
    rtol: onp.ToFloat = 0.0,
    *,
    full_output: Truthy,
    disp: onp.ToBool = True,
) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.bool_], onp.Array1D[np.bool_]]: ...

#
@overload
def bisect(
    f: _Fun0D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 2e-12,
    rtol: onp.ToFloat = ...,
    maxiter: onp.ToJustInt = 100,
    full_output: Falsy = False,
    disp: op.CanBool = True,
) -> float | int | bool: ...
@overload
def bisect(
    f: _Fun0D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 2e-12,
    rtol: onp.ToFloat = ...,
    maxiter: onp.ToJustInt = 100,
    *,
    full_output: Truthy,
    disp: op.CanBool = True,
) -> tuple[float | int | bool, RootResults[_Float]]: ...

#
@overload
def ridder(
    f: _Fun0D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 2e-12,
    rtol: onp.ToFloat = ...,
    maxiter: onp.ToJustInt = 100,
    full_output: Falsy = False,
    disp: op.CanBool = True,
) -> float | int | bool: ...
@overload
def ridder(
    f: _Fun0D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 2e-12,
    rtol: onp.ToFloat = ...,
    maxiter: onp.ToJustInt = 100,
    *,
    full_output: Truthy,
    disp: op.CanBool = True,
) -> tuple[float | int | bool, RootResults[_Float]]: ...

#
@overload
def brentq(
    f: _Fun0D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 2e-12,
    rtol: onp.ToFloat = ...,
    maxiter: onp.ToJustInt = 100,
    full_output: Falsy = False,
    disp: op.CanBool = True,
) -> float | int | bool: ...
@overload
def brentq(
    f: _Fun0D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 2e-12,
    rtol: onp.ToFloat = ...,
    maxiter: onp.ToJustInt = 100,
    *,
    full_output: Truthy,
    disp: op.CanBool = True,
) -> tuple[float | int | bool, RootResults[_Float]]: ...

#
@overload
def brenth(
    f: _Fun0D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 2e-12,
    rtol: onp.ToFloat = ...,
    maxiter: onp.ToJustInt = 100,
    full_output: Falsy = False,
    disp: op.CanBool = True,
) -> float | int | bool: ...
@overload
def brenth(
    f: _Fun0D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...] = (),
    xtol: onp.ToFloat = 2e-12,
    rtol: onp.ToFloat = ...,
    maxiter: onp.ToJustInt = 100,
    *,
    full_output: Truthy,
    disp: op.CanBool = True,
) -> tuple[float | int | bool, RootResults[_Float]]: ...

#
@overload
def toms748(
    f: _Fun0D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...] = (),
    k: onp.ToJustInt = 1,
    xtol: onp.ToFloat = 2e-12,
    rtol: onp.ToFloat = ...,
    maxiter: onp.ToJustInt = 100,
    full_output: Falsy = False,
    disp: op.CanBool = True,
) -> np.float64: ...
@overload
def toms748(
    f: _Fun0D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...] = (),
    k: onp.ToJustInt = 1,
    xtol: onp.ToFloat = 2e-12,
    rtol: onp.ToFloat = ...,
    maxiter: onp.ToJustInt = 100,
    *,
    full_output: Truthy,
    disp: op.CanBool = True,
) -> tuple[np.float64, RootResults[_Float]]: ...
