from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Literal, NamedTuple, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.stats.qmc import QMCEngine

__all__ = ["cumulative_simpson", "cumulative_trapezoid", "fixed_quad", "newton_cotes", "qmc_quad", "romb", "simpson", "trapezoid"]

_NDT_f = TypeVar("_NDT_f", bound=_QuadFuncOut)
_QuadFuncOut: TypeAlias = onp.ArrayND[np.floating[Any]] | Sequence[float | int | bool]

###

class QMCQuadResult(NamedTuple):
    integral: float | int | bool
    standard_error: float | int | bool

# sample-based integration
@overload
def trapezoid(
    y: onp.ToFloatND,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
) -> np.floating[Any] | onp.ArrayND[np.floating[Any]]: ...
@overload
def trapezoid(
    y: onp.ToComplexND,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
) -> np.inexact[Any] | onp.ArrayND[np.inexact[Any]]: ...

#
@overload
def simpson(
    y: onp.ToFloatND,
    x: onp.ToFloatND | None = None,
    *,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
) -> np.floating[Any] | onp.ArrayND[np.floating[Any]]: ...
@overload
def simpson(
    y: onp.ToComplexND,
    x: onp.ToFloatND | None = None,
    *,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
) -> np.inexact[Any] | onp.ArrayND[np.inexact[Any]]: ...

#
@overload
def romb(
    y: onp.ToFloatND,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    show: bool = False,
) -> np.floating[Any] | onp.ArrayND[np.floating[Any]]: ...
@overload
def romb(
    y: onp.ToComplexND,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    show: bool = False,
) -> np.inexact[Any] | onp.ArrayND[np.inexact[Any]]: ...

# sample-based cumulative integration
@overload
def cumulative_trapezoid(
    y: onp.ToFloatND,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: Literal[0] | None = None,
) -> onp.ArrayND[np.floating[Any]]: ...
@overload
def cumulative_trapezoid(
    y: onp.ToComplexND,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: Literal[0] | None = None,
) -> onp.ArrayND[np.inexact[Any]]: ...

#
@overload
def cumulative_simpson(
    y: onp.ToFloatND,
    *,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: onp.ToFloatND | None = None,
) -> onp.ArrayND[np.floating[Any]]: ...
@overload
def cumulative_simpson(
    y: onp.ToComplexND,
    *,
    x: onp.ToFloatND | None = None,
    dx: onp.ToFloat = 1.0,
    axis: op.CanIndex = -1,
    initial: onp.ToComplexND | None = None,
) -> onp.ArrayND[np.inexact[Any]]: ...

# function-based
@overload
def fixed_quad(
    func: Callable[[onp.Array1D[np.float64]], _NDT_f],
    a: onp.ToFloat,
    b: onp.ToFloat,
    args: tuple[()] = (),
    n: op.CanIndex = 5,
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
    func: Callable[[onp.Array2D[np.float64]], onp.ArrayND[np.floating[Any]]],
    a: onp.ToFloat1D,
    b: onp.ToFloat1D,
    *,
    n_estimates: int | bool = 8,
    n_points: int | bool = 1024,
    qrng: QMCEngine | None = None,
    log: bool = False,
) -> QMCQuadResult: ...

# low-level
def newton_cotes(rn: int | bool, equal: int | bool = 0) -> tuple[onp.Array1D[np.float64], float | int | bool]: ...
