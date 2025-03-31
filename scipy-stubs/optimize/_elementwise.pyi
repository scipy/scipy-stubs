from collections.abc import Callable
from typing import Any, Concatenate, Final, Generic, TypeAlias, TypedDict, final, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy._lib._util import _RichResult

_T = TypeVar("_T")
_ShapeT = TypeVar("_ShapeT", bound=tuple[int | bool, ...], default=tuple[int | bool, ...])
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int | bool, ...], default=tuple[int | bool, ...], covariant=True)

_Bracket: TypeAlias = tuple[_T, _T]

@type_check_only
class _Tolerances(TypedDict, total=False):
    xatol: onp.ToFloat
    xrtol: onp.ToFloat
    fatol: onp.ToFloat
    frtol: onp.ToFloat

@type_check_only
@final
class _FindResult(_RichResult, Generic[_ShapeT_co]):
    success: onp.ArrayND[np.bool_, _ShapeT_co]
    status: onp.ArrayND[np.int32, _ShapeT_co]
    x: onp.ArrayND[np.float64, _ShapeT_co]
    f_x: onp.ArrayND[np.float64, _ShapeT_co]
    nfev: onp.ArrayND[np.int32, _ShapeT_co]
    nit: onp.ArrayND[np.int32, _ShapeT_co]
    bracket: _Bracket[onp.ArrayND[np.float64, _ShapeT_co]]
    f_bracket: _Bracket[onp.ArrayND[np.float64, _ShapeT_co]]
    _order_keys: Final = ["success", "status", "x", "f_x", "nfev", "nit", "bracket", "f_bracket"]

@type_check_only
@final
class _BracketResult(_RichResult, Generic[_ShapeT_co]):
    success: onp.ArrayND[np.bool_, _ShapeT_co]
    status: onp.ArrayND[np.int32, _ShapeT_co]
    bracket: _Bracket[onp.ArrayND[np.float64, _ShapeT_co]]
    f_bracket: _Bracket[onp.ArrayND[np.float64, _ShapeT_co]]
    nfev: onp.ArrayND[np.int32, _ShapeT_co]
    nit: onp.ArrayND[np.int32, _ShapeT_co]

###

#
@overload
def find_root(
    f: Callable[[onp.ArrayND[np.float64, _ShapeT]], onp.ArrayND[np.floating[Any]]],
    init: _Bracket[onp.ToFloat] | _Bracket[onp.ToFloatND],
    /,
    *,
    args: tuple[()] = (),
    tolerances: _Tolerances | None = None,
    maxiter: int | bool | None = None,
    callback: Callable[[_FindResult[_ShapeT]], None] | None = None,
) -> _FindResult[_ShapeT]: ...
@overload
def find_root(
    f: Callable[Concatenate[onp.ArrayND[np.float64, _ShapeT], ...], onp.ArrayND[np.floating[Any]]],
    init: _Bracket[onp.ToFloat] | _Bracket[onp.ToFloatND],
    /,
    *,
    args: tuple[object, ...],
    tolerances: _Tolerances | None = None,
    maxiter: int | bool | None = None,
    callback: Callable[[_FindResult[_ShapeT]], None] | None = None,
) -> _FindResult[_ShapeT]: ...

#
@overload
def find_minimum(
    f: Callable[[onp.ArrayND[np.float64, _ShapeT]], onp.ArrayND[np.floating[Any]]],
    init: tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat] | tuple[onp.ToFloatND, onp.ToFloatND, onp.ToFloatND],
    /,
    *,
    args: tuple[()] = (),
    tolerances: _Tolerances | None = None,
    maxiter: int | bool = 100,
    callback: Callable[[_FindResult[_ShapeT]], None] | None = None,
) -> _FindResult[_ShapeT]: ...
@overload
def find_minimum(
    f: Callable[Concatenate[onp.ArrayND[np.float64, _ShapeT], ...], onp.ArrayND[np.floating[Any]]],
    init: tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat] | tuple[onp.ToFloatND, onp.ToFloatND, onp.ToFloatND],
    /,
    *,
    args: tuple[object, ...],
    tolerances: _Tolerances | None = None,
    maxiter: int | bool = 100,
    callback: Callable[[_FindResult[_ShapeT]], None] | None = None,
) -> _FindResult[_ShapeT]: ...

#
@overload
def bracket_root(
    f: Callable[[onp.ArrayND[np.float64, _ShapeT]], onp.ArrayND[np.floating[Any]]],
    xl0: onp.ToFloat | onp.ToFloatND,
    xr0: onp.ToFloat | onp.ToFloatND | None = None,
    *,
    xmin: onp.ToFloat | onp.ToFloatND | None = None,
    xmax: onp.ToFloat | onp.ToFloatND | None = None,
    factor: onp.ToFloat | onp.ToFloatND | None = None,
    args: tuple[()] = (),
    maxiter: int | bool = 1_000,
) -> _BracketResult[_ShapeT]: ...
@overload
def bracket_root(
    f: Callable[Concatenate[onp.ArrayND[np.float64, _ShapeT], ...], onp.ArrayND[np.floating[Any]]],
    xl0: onp.ToFloat | onp.ToFloatND,
    xr0: onp.ToFloat | onp.ToFloatND | None = None,
    *,
    xmin: onp.ToFloat | onp.ToFloatND | None = None,
    xmax: onp.ToFloat | onp.ToFloatND | None = None,
    factor: onp.ToFloat | onp.ToFloatND | None = None,
    args: tuple[object, ...],
    maxiter: int | bool = 1_000,
) -> _BracketResult[_ShapeT]: ...

#
@overload
def bracket_minimum(
    f: Callable[[onp.ArrayND[np.float64, _ShapeT]], onp.ArrayND[np.floating[Any]]],
    xm0: onp.ToFloat | onp.ToFloatND,
    *,
    xl0: onp.ToFloat | onp.ToFloatND | None = None,
    xr0: onp.ToFloat | onp.ToFloatND | None = None,
    xmin: onp.ToFloat | onp.ToFloatND | None = None,
    xmax: onp.ToFloat | onp.ToFloatND | None = None,
    factor: onp.ToFloat | onp.ToFloatND | None = None,
    args: tuple[()] = (),
    maxiter: int | bool = 1_000,
) -> _BracketResult[_ShapeT]: ...
@overload
def bracket_minimum(
    f: Callable[Concatenate[onp.ArrayND[np.float64, _ShapeT], ...], onp.ArrayND[np.floating[Any]]],
    xm0: onp.ToFloat | onp.ToFloatND,
    *,
    xl0: onp.ToFloat | onp.ToFloatND | None = None,
    xr0: onp.ToFloat | onp.ToFloatND | None = None,
    xmin: onp.ToFloat | onp.ToFloatND | None = None,
    xmax: onp.ToFloat | onp.ToFloatND | None = None,
    factor: onp.ToFloat | onp.ToFloatND | None = None,
    args: tuple[object, ...],
    maxiter: int | bool = 1_000,
) -> _BracketResult[_ShapeT]: ...
