from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Literal, NamedTuple, Never, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.stats.qmc import QMCEngine

__all__ = ["cumulative_simpson", "cumulative_trapezoid", "fixed_quad", "newton_cotes", "qmc_quad", "romb", "simpson", "trapezoid"]

_NDT_f = TypeVar("_NDT_f", bound=_QuadFuncOut)
_QuadFuncOut: TypeAlias = onp.ArrayND[npc.floating] | Sequence[float]

_InexactT = TypeVar("_InexactT", bound=npc.inexact)

# workaround for mypy & pyright's failure to conform to the overload typing specification
_JustAnyShape: TypeAlias = tuple[Never, Never, Never]

###

class QMCQuadResult(NamedTuple):
    integral: float
    standard_error: float

# sample-based integration

#
@overload  # ?d +complex  (mypy & pyright workaround)
def trapezoid(
    y: onp.Array[_JustAnyShape, npc.number | np.bool_], x: onp.ToFloatND | None = None, dx: float = 1.0, axis: int = -1
) -> Any: ...
@overload  # 1d T:inexact
def trapezoid(y: onp.Array1D[_InexactT], x: onp.ToFloat1D | None = None, dx: float = 1.0, axis: int = -1) -> _InexactT: ...
@overload  # 1d +int
def trapezoid(
    y: onp.ToArrayStrict1D[float, npc.integer | np.bool_], x: onp.ToFloat1D | None = None, dx: float = 1.0, axis: int = -1
) -> np.float64: ...
@overload  # 1d ~complex
def trapezoid(
    y: onp.ToJustComplex128Strict1D, x: onp.ToFloat1D | None = None, dx: float = 1.0, axis: int = -1
) -> np.complex128: ...
@overload  # 2d T:inexact
def trapezoid(
    y: onp.Array2D[_InexactT], x: onp.ToFloatND | None = None, dx: float = 1.0, axis: int = -1
) -> onp.Array1D[_InexactT]: ...
@overload  # 2d +int
def trapezoid(
    y: onp.ToArrayStrict2D[float, npc.integer | np.bool_], x: onp.ToFloatND | None = None, dx: float = 1.0, axis: int = -1
) -> onp.Array1D[np.float64]: ...
@overload  # 2d ~complex
def trapezoid(
    y: onp.ToJustComplex128Strict2D, x: onp.ToFloatND | None = None, dx: float = 1.0, axis: int = -1
) -> onp.Array1D[np.complex128]: ...
@overload  # Nd T:inexact
def trapezoid(
    y: onp.ArrayND[_InexactT], x: onp.ToFloatND | None = None, dx: float = 1.0, axis: int = -1
) -> onp.ArrayND[_InexactT] | Any: ...
@overload  # Nd +int
def trapezoid(
    y: onp.ToArrayND[float, npc.integer | np.bool_], x: onp.ToFloatND | None = None, dx: float = 1.0, axis: int = -1
) -> onp.ArrayND[np.float64] | Any: ...
@overload  # Nd ~complex
def trapezoid(
    y: onp.ToJustComplex128_ND, x: onp.ToFloatND | None = None, dx: float = 1.0, axis: int = -1
) -> onp.ArrayND[np.complex128] | Any: ...
@overload  # +float (fallback)
def trapezoid(
    y: onp.ToFloatND, x: onp.ToFloatND | None = None, dx: float = 1.0, axis: int = -1
) -> onp.ArrayND[np.float64 | Any] | Any: ...
@overload  # +complex (fallback)
def trapezoid(
    y: onp.ToComplexND, x: onp.ToFloatND | None = None, dx: float = 1.0, axis: int = -1
) -> onp.ArrayND[np.complex128 | Any] | Any: ...

# TODO(@jorenham): improve like trapezoid
@overload
def simpson(
    y: onp.ToFloatND, x: onp.ToFloatND | None = None, *, dx: onp.ToFloat = 1.0, axis: op.CanIndex = -1
) -> npc.floating | onp.ArrayND[npc.floating]: ...
@overload
def simpson(
    y: onp.ToComplexND, x: onp.ToFloatND | None = None, *, dx: onp.ToFloat = 1.0, axis: op.CanIndex = -1
) -> npc.inexact | onp.ArrayND[npc.inexact]: ...

# TODO(@jorenham): improve like trapezoid
@overload
def romb(
    y: onp.ToFloatND, dx: onp.ToFloat = 1.0, axis: op.CanIndex = -1, show: bool = False
) -> npc.floating | onp.ArrayND[npc.floating]: ...
@overload
def romb(
    y: onp.ToComplexND, dx: onp.ToFloat = 1.0, axis: op.CanIndex = -1, show: bool = False
) -> npc.inexact | onp.ArrayND[npc.inexact]: ...

# sample-based cumulative integration

# TODO(@jorenham): improve
@overload
def cumulative_trapezoid(
    y: onp.ToFloatND,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: Literal[0] | None = None,
) -> onp.ArrayND[npc.floating]: ...
@overload
def cumulative_trapezoid(
    y: onp.ToComplexND,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: Literal[0] | None = None,
) -> onp.ArrayND[npc.inexact]: ...

# TODO(@jorenham): improve
@overload
def cumulative_simpson(
    y: onp.ToFloatND,
    *,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: onp.ToFloatND | None = None,
) -> onp.ArrayND[npc.floating]: ...
@overload
def cumulative_simpson(
    y: onp.ToComplexND,
    *,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: onp.ToComplexND | None = None,
) -> onp.ArrayND[npc.inexact]: ...

# function-based
@overload
def fixed_quad(
    func: Callable[[onp.Array1D[np.float64]], _NDT_f], a: onp.ToFloat, b: onp.ToFloat, args: tuple[()] = (), n: op.CanIndex = 5
) -> _NDT_f: ...
@overload
def fixed_quad(
    func: Callable[Concatenate[onp.Array1D[np.float64], ...], _NDT_f],
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[object, ...],
    n: op.CanIndex = 5,
) -> _NDT_f: ...

#
def qmc_quad(
    func: Callable[[onp.Array2D[np.float64]], onp.ArrayND[npc.floating]],
    a: onp.ToFloat1D,
    b: onp.ToFloat1D,
    *,
    n_estimates: int = 8,
    n_points: int = 1024,
    qrng: QMCEngine | None = None,
    log: bool = False,
) -> QMCQuadResult: ...

# low-level
def newton_cotes(rn: int, equal: int = 0) -> tuple[onp.Array1D[np.float64], float]: ...
