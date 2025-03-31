from collections.abc import Callable, Iterable, Mapping, Sequence
from types import ModuleType
from typing import Any, Concatenate, Final, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Falsy
from ._util import _RichResult

###

_FloatT = TypeVar("_FloatT", bound=np.floating[Any], default=np.floating[Any])
_ShapeT = TypeVar("_ShapeT", bound=onp.AtLeast1D, default=onp.AtLeast1D)
_FuncRealT = TypeVar("_FuncRealT", bound=Callable[Concatenate[onp.ArrayND[np.float64], ...], object])
_ModuleT = TypeVar("_ModuleT", bound=ModuleType, default=ModuleType)
_WorkT = TypeVar("_WorkT", bound=Mapping[str, Any])
_ResT = TypeVar("_ResT", bound=_RichResult, default=_RichResult)
_ToShapeT = TypeVar("_ToShapeT", bound=op.CanIndex | tuple[op.CanIndex, ...], default=tuple[int | bool, ...])

_Ignored: TypeAlias = _ResT

###

_ESIGNERR: Final = -1
_ECONVERR: Final = -2
_EVALUEERR: Final = -3
_ECALLBACK: Final = -4
_EINPUTERR: Final = -5
_ECONVERGED: Final = 0
_EINPROGRESS: Final = 1

# TODO: complex | float | int | bool
def _initialize(
    func: _FuncRealT,
    xs: Sequence[onp.ToFloat1D],
    args: tuple[onp.ToFloat1D, ...],
    complex_ok: Falsy = False,
    preserve_shape: bool | None = None,
    xp: _ModuleT | None = None,
) -> tuple[
    _FuncRealT,  # func
    list[onp.Array1D[_FloatT]],  # xs
    list[onp.Array1D[_FloatT]],  # fs
    list[onp.Array1D[np.floating[Any]]],  # args
    onp.AtLeast1D,  # shape
    _FloatT,  # xfat
    _ModuleT,  # xp
]: ...

# TODO: `_RichResult` subtype
def _loop(
    work: _ResT,
    callback: Callable[[_ResT], _Ignored],
    shape: Sequence[op.CanIndex],
    maxiter: int | bool,
    func: Callable[[onp.Array[_ShapeT, _FloatT]], onp.ToComplexND],
    args: tuple[onp.ArrayND[np.floating[Any]], ...],
    dtype: np.inexact[Any],
    pre_func_eval: Callable[[_ResT], onp.Array[_ShapeT, _FloatT]],
    post_func_eval: Callable[[onp.Array[_ShapeT, _FloatT], onp.Array[_ShapeT, np.floating[Any]], _ResT], _Ignored],
    check_termination: Callable[[_ResT], onp.Array[_ShapeT, np.bool_]],
    post_termination_check: Callable[[_ResT], _Ignored],
    customize_result: Callable[[_ResT, _ToShapeT], tuple[int | bool, ...]],
    res_work_pairs: Iterable[tuple[str, str]],
    xp: ModuleType,
    preserve_shape: bool | None = False,
) -> _ResT: ...

#
def _check_termination(
    work: _WorkT,
    res: Mapping[str, onp.Array[_ShapeT, _FloatT]],
    res_work_pairs: Iterable[tuple[str, str]],
    active: onp.Array[_ShapeT, np.integer[Any]],
    check_termination: Callable[[_WorkT], onp.Array[_ShapeT, np.bool_]],
    preserve_shape: bool | None,
    xp: ModuleType,
) -> onp.Array1D[np.intp]: ...

#
def _update_active(
    work: Mapping[str, onp.Array[_ShapeT, _FloatT]],
    res: Mapping[str, onp.Array[_ShapeT, _FloatT]],
    res_work_pairs: Iterable[tuple[str, str]],
    active: onp.Array[_ShapeT, np.integer[Any]],
    mask: onp.Array[_ShapeT, np.bool_] | None,
    preserve_shape: bool | None,
    xp: ModuleType,
) -> None: ...

#
def _prepare_result(
    work: Mapping[str, onp.Array[_ShapeT, _FloatT]],
    res: _ResT,
    res_work_pairs: Iterable[tuple[str, str]],
    active: onp.Array[_ShapeT, np.integer[Any]],
    shape: _ToShapeT,
    customize_result: Callable[[_ResT, _ToShapeT], tuple[int | bool, ...]],
    preserve_shape: bool | None,
    xp: ModuleType,
) -> _ResT: ...
