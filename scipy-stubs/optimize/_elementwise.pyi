from collections.abc import Callable, Mapping
from typing import Concatenate, Final, final, overload, type_check_only
from typing_extensions import TypedDict

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._lib._util import _RichResult

type _Tuple2[T] = tuple[T, T]
type _Tuple3[T] = tuple[T, T, T]

@type_check_only
class _Tolerances(TypedDict, total=False, closed=True):
    xatol: onp.ToFloat
    xrtol: onp.ToFloat
    fatol: onp.ToFloat
    frtol: onp.ToFloat

@type_check_only
class _ResultBase[ShapeT: tuple[int, ...], BrackT](_RichResult):
    success: onp.ArrayND[np.bool, ShapeT]
    status: onp.ArrayND[np.int32, ShapeT]
    nfev: onp.ArrayND[np.int32, ShapeT]
    nit: onp.ArrayND[np.int32, ShapeT]
    bracket: BrackT
    f_bracket: BrackT

@type_check_only
class _FindResultBase[ShapeT: tuple[int, ...], BrackT](_ResultBase[ShapeT, BrackT]):
    x: onp.ArrayND[np.float64, ShapeT]
    f_x: onp.ArrayND[np.float64, ShapeT]
    _order_keys: Final = ["success", "status", "x", "f_x", "nfev", "nit", "bracket", "f_bracket"]

@type_check_only
@final
class _FindRootResult[ShapeT: tuple[int, ...]](_FindResultBase[ShapeT, _Tuple2[onp.ArrayND[np.float64, ShapeT]]]): ...

@type_check_only
@final
class _FindMinResult[ShapeT: tuple[int, ...]](_FindResultBase[ShapeT, _Tuple3[onp.ArrayND[np.float64, ShapeT]]]): ...

@type_check_only
@final
class _BracketRootResult[ShapeT: tuple[int, ...]](_ResultBase[ShapeT, _Tuple2[onp.ArrayND[np.float64, ShapeT]]]): ...

@type_check_only
@final
class _BracketMinResult[ShapeT: tuple[int, ...]](_ResultBase[ShapeT, _Tuple3[onp.ArrayND[np.float64, ShapeT]]]): ...

###

# TODO(@jorenham): array-api support
@overload
def find_root[ShapeT: tuple[int, ...]](
    f: Callable[[onp.ArrayND[np.float64, ShapeT]], onp.ArrayND[npc.floating]],
    init: _Tuple2[onp.ToFloat] | _Tuple2[onp.ToFloatND],
    /,
    *,
    args: tuple[()] = (),
    kwargs: None = None,
    tolerances: _Tolerances | None = None,
    maxiter: int | None = None,
    callback: Callable[[_FindRootResult[ShapeT]], None] | None = None,
) -> _FindRootResult[ShapeT]: ...
@overload
def find_root[ShapeT: tuple[int, ...]](
    f: Callable[Concatenate[onp.ArrayND[np.float64, ShapeT], ...], onp.ArrayND[npc.floating]],
    init: _Tuple2[onp.ToFloat] | _Tuple2[onp.ToFloatND],
    /,
    *,
    args: tuple[object, ...],
    kwargs: Mapping[str, object] | None = None,
    tolerances: _Tolerances | None = None,
    maxiter: int | None = None,
    callback: Callable[[_FindRootResult[ShapeT]], None] | None = None,
) -> _FindRootResult[ShapeT]: ...
@overload
def find_root[ShapeT: tuple[int, ...]](
    f: Callable[Concatenate[onp.ArrayND[np.float64, ShapeT], ...], onp.ArrayND[npc.floating]],
    init: _Tuple2[onp.ToFloat] | _Tuple2[onp.ToFloatND],
    /,
    *,
    args: tuple[object, ...] = (),
    kwargs: Mapping[str, object],
    tolerances: _Tolerances | None = None,
    maxiter: int | None = None,
    callback: Callable[[_FindRootResult[ShapeT]], None] | None = None,
) -> _FindRootResult[ShapeT]: ...

# TODO(@jorenham): array-api support
@overload
def find_minimum[ShapeT: tuple[int, ...]](
    f: Callable[[onp.ArrayND[np.float64, ShapeT]], onp.ArrayND[npc.floating]],
    init: tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat] | tuple[onp.ToFloatND, onp.ToFloatND, onp.ToFloatND],
    /,
    *,
    args: tuple[()] = (),
    kwargs: None = None,
    tolerances: _Tolerances | None = None,
    maxiter: int = 100,
    callback: Callable[[_FindMinResult[ShapeT]], None] | None = None,
) -> _FindMinResult[ShapeT]: ...
@overload
def find_minimum[ShapeT: tuple[int, ...]](
    f: Callable[Concatenate[onp.ArrayND[np.float64, ShapeT], ...], onp.ArrayND[npc.floating]],
    init: tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat] | tuple[onp.ToFloatND, onp.ToFloatND, onp.ToFloatND],
    /,
    *,
    args: tuple[object, ...],
    kwargs: Mapping[str, object] | None = None,
    tolerances: _Tolerances | None = None,
    maxiter: int = 100,
    callback: Callable[[_FindMinResult[ShapeT]], None] | None = None,
) -> _FindMinResult[ShapeT]: ...
@overload
def find_minimum[ShapeT: tuple[int, ...]](
    f: Callable[Concatenate[onp.ArrayND[np.float64, ShapeT], ...], onp.ArrayND[npc.floating]],
    init: tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat] | tuple[onp.ToFloatND, onp.ToFloatND, onp.ToFloatND],
    /,
    *,
    args: tuple[object, ...] = (),
    kwargs: Mapping[str, object],
    tolerances: _Tolerances | None = None,
    maxiter: int = 100,
    callback: Callable[[_FindMinResult[ShapeT]], None] | None = None,
) -> _FindMinResult[ShapeT]: ...

# TODO(@jorenham): array-api support
@overload
def bracket_root[ShapeT: tuple[int, ...]](
    f: Callable[[onp.ArrayND[np.float64, ShapeT]], onp.ArrayND[npc.floating]],
    xl0: onp.ToFloat | onp.ToFloatND,
    xr0: onp.ToFloat | onp.ToFloatND | None = None,
    *,
    xmin: onp.ToFloat | onp.ToFloatND | None = None,
    xmax: onp.ToFloat | onp.ToFloatND | None = None,
    factor: onp.ToFloat | onp.ToFloatND | None = None,
    args: tuple[()] = (),
    kwargs: None = None,
    maxiter: int = 1_000,
) -> _BracketRootResult[ShapeT]: ...
@overload
def bracket_root[ShapeT: tuple[int, ...]](
    f: Callable[Concatenate[onp.ArrayND[np.float64, ShapeT], ...], onp.ArrayND[npc.floating]],
    xl0: onp.ToFloat | onp.ToFloatND,
    xr0: onp.ToFloat | onp.ToFloatND | None = None,
    *,
    xmin: onp.ToFloat | onp.ToFloatND | None = None,
    xmax: onp.ToFloat | onp.ToFloatND | None = None,
    factor: onp.ToFloat | onp.ToFloatND | None = None,
    args: tuple[object, ...],
    kwargs: Mapping[str, object] | None = None,
    maxiter: int = 1_000,
) -> _BracketRootResult[ShapeT]: ...
@overload
def bracket_root[ShapeT: tuple[int, ...]](
    f: Callable[Concatenate[onp.ArrayND[np.float64, ShapeT], ...], onp.ArrayND[npc.floating]],
    xl0: onp.ToFloat | onp.ToFloatND,
    xr0: onp.ToFloat | onp.ToFloatND | None = None,
    *,
    xmin: onp.ToFloat | onp.ToFloatND | None = None,
    xmax: onp.ToFloat | onp.ToFloatND | None = None,
    factor: onp.ToFloat | onp.ToFloatND | None = None,
    args: tuple[object, ...] = (),
    kwargs: Mapping[str, object],
    maxiter: int = 1_000,
) -> _BracketRootResult[ShapeT]: ...

# TODO(@jorenham): array-api support
@overload
def bracket_minimum[ShapeT: tuple[int, ...]](
    f: Callable[[onp.ArrayND[np.float64, ShapeT]], onp.ArrayND[npc.floating]],
    xm0: onp.ToFloat | onp.ToFloatND,
    *,
    xl0: onp.ToFloat | onp.ToFloatND | None = None,
    xr0: onp.ToFloat | onp.ToFloatND | None = None,
    xmin: onp.ToFloat | onp.ToFloatND | None = None,
    xmax: onp.ToFloat | onp.ToFloatND | None = None,
    factor: onp.ToFloat | onp.ToFloatND | None = None,
    args: tuple[()] = (),
    kwargs: None = None,
    maxiter: int = 1_000,
) -> _BracketMinResult[ShapeT]: ...
@overload
def bracket_minimum[ShapeT: tuple[int, ...]](
    f: Callable[Concatenate[onp.ArrayND[np.float64, ShapeT], ...], onp.ArrayND[npc.floating]],
    xm0: onp.ToFloat | onp.ToFloatND,
    *,
    xl0: onp.ToFloat | onp.ToFloatND | None = None,
    xr0: onp.ToFloat | onp.ToFloatND | None = None,
    xmin: onp.ToFloat | onp.ToFloatND | None = None,
    xmax: onp.ToFloat | onp.ToFloatND | None = None,
    factor: onp.ToFloat | onp.ToFloatND | None = None,
    args: tuple[object, ...],
    kwargs: Mapping[str, object] | None = None,
    maxiter: int = 1_000,
) -> _BracketMinResult[ShapeT]: ...
@overload
def bracket_minimum[ShapeT: tuple[int, ...]](
    f: Callable[Concatenate[onp.ArrayND[np.float64, ShapeT], ...], onp.ArrayND[npc.floating]],
    xm0: onp.ToFloat | onp.ToFloatND,
    *,
    xl0: onp.ToFloat | onp.ToFloatND | None = None,
    xr0: onp.ToFloat | onp.ToFloatND | None = None,
    xmin: onp.ToFloat | onp.ToFloatND | None = None,
    xmax: onp.ToFloat | onp.ToFloatND | None = None,
    factor: onp.ToFloat | onp.ToFloatND | None = None,
    args: tuple[object, ...] = (),
    kwargs: Mapping[str, object],
    maxiter: int = 1_000,
) -> _BracketMinResult[ShapeT]: ...
