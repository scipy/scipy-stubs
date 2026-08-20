from _typeshed import ConvertibleToInt, Unused
from collections.abc import Callable, Mapping
from typing import Any, Concatenate, Literal, Never, overload, type_check_only
from typing_extensions import TypedDict

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._lib._util import _RichResult

###

type _Function00[FloatT: npc.floating] = Callable[Concatenate[FloatT, ...], onp.ToFloat]
type _Function11[FloatT: npc.floating] = Callable[Concatenate[onp.Array1D[FloatT], ...], onp.ToFloat1D]
type _FunctionNN[FloatT: npc.floating] = Callable[Concatenate[onp.ArrayND[FloatT], ...], onp.ToFloatND]

type _AsF64ND = onp.ToArrayND[float, np.float64 | npc.integer | np.bool]

type _ToArgsND = tuple[onp.ToScalar | onp.ToArrayND, ...]
type _ToKwargsND = Mapping[str, onp.ToScalar | onp.ToArrayND]

# workaround for https://github.com/microsoft/pyright/issues/10232
type _JustAnyShape = tuple[Never, Never, Never, Never]

@type_check_only
class _Tolerances(TypedDict, total=False, closed=True):
    rtol: onp.ToFloat
    atol: onp.ToFloat

@type_check_only
class _DerivativeResult0D[FloatT: npc.floating](_RichResult[FloatT | np.int32 | np.bool]):
    success: np.bool
    status: np.int32
    nfev: np.int32
    nit: np.int32
    x: FloatT
    df: FloatT
    error: FloatT

@type_check_only
class _DerivativeResultND[FloatT: npc.floating, ShapeT: tuple[int, ...]](
    _RichResult[onp.ArrayND[FloatT | np.int32 | np.bool, ShapeT]]
):
    success: onp.ArrayND[np.bool, ShapeT]
    status: onp.ArrayND[np.int32, ShapeT]
    nfev: onp.ArrayND[np.int32, ShapeT]
    nit: onp.ArrayND[np.int32, ShapeT]
    x: onp.ArrayND[FloatT, ShapeT]
    df: onp.ArrayND[FloatT, ShapeT]
    error: onp.ArrayND[FloatT, ShapeT]

@type_check_only
class _JacobianResult[FloatT: npc.floating, ShapeT: tuple[int, ...]](
    _RichResult[onp.ArrayND[FloatT | np.int32 | np.bool, ShapeT]]
):
    status: onp.ArrayND[np.int32, ShapeT]
    df: onp.ArrayND[FloatT, ShapeT]
    error: onp.ArrayND[FloatT, ShapeT]
    nit: onp.ArrayND[np.int32, ShapeT]
    nfev: onp.ArrayND[np.int32, ShapeT]
    success: onp.ArrayND[np.bool, ShapeT]

@type_check_only
class _HessianResult[FloatT: npc.floating, ShapeT: tuple[int, ...]](
    _RichResult[onp.ArrayND[FloatT | np.int32 | np.bool, ShapeT]]
):
    status: onp.ArrayND[np.int32, ShapeT]
    error: onp.ArrayND[FloatT, ShapeT]
    nfev: onp.ArrayND[np.int64, ShapeT]
    success: onp.ArrayND[np.bool, ShapeT]
    ddf: onp.ArrayND[FloatT, ShapeT]

###

@overload  # ?d f64
def derivative(
    f: _FunctionNN[np.float64],
    x: onp.ArrayND[np.float64 | npc.integer | np.bool, _JustAnyShape],
    *,
    args: _ToArgsND = (),
    kwargs: _ToKwargsND | None = None,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat | onp.ToFloatND = 0.5,
    step_factor: onp.ToFloat = 2.0,
    step_direction: onp.ToJustInt | onp.ToJustIntND = 0,
    preserve_shape: bool = False,
    callback: Callable[[_DerivativeResultND[np.float64, tuple[Any, ...]]], Unused] | None = None,
) -> _DerivativeResultND[np.float64, tuple[Any, ...]]: ...
@overload  # ?d <known>
def derivative[FloatT: npc.floating](
    f: _FunctionNN[FloatT],
    x: onp.ArrayND[FloatT, _JustAnyShape],
    *,
    args: _ToArgsND = (),
    kwargs: _ToKwargsND | None = None,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat | onp.ToFloatND = 0.5,
    step_factor: onp.ToFloat = 2.0,
    step_direction: onp.ToJustInt | onp.ToJustIntND = 0,
    preserve_shape: bool = False,
    callback: Callable[[_DerivativeResultND[FloatT, tuple[Any, ...]]], Unused] | None = None,
) -> _DerivativeResultND[FloatT, tuple[Any, ...]]: ...
@overload  # 0d f64
def derivative(
    f: _Function00[np.float64],
    x: float | np.float64 | npc.integer | onp.Array0D[np.float64 | npc.integer],
    *,
    args: tuple[onp.ToScalar, ...] = (),
    kwargs: Mapping[str, onp.ToScalar] | None = None,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat = 0.5,
    step_factor: onp.ToFloat = 2.0,
    step_direction: onp.ToJustInt = 0,
    preserve_shape: Literal[False] = False,
    callback: Callable[[_DerivativeResult0D[np.float64]], Unused] | None = None,
) -> _DerivativeResult0D[np.float64]: ...
@overload  # 0d <known>
def derivative[FloatT: npc.floating](
    f: _Function00[FloatT],
    x: FloatT | onp.Array0D[FloatT | npc.integer],
    *,
    args: tuple[onp.ToScalar, ...] = (),
    kwargs: Mapping[str, onp.ToScalar] | None = None,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat = 0.5,
    step_factor: onp.ToFloat = 2.0,
    step_direction: onp.ToJustInt = 0,
    preserve_shape: Literal[False] = False,
    callback: Callable[[_DerivativeResult0D[FloatT]], Unused] | None = None,
) -> _DerivativeResult0D[FloatT]: ...
@overload  # 1d f64
def derivative(
    f: _Function11[np.float64],
    x: onp.ToArrayStrict1D[float, np.float64 | npc.integer | np.bool],
    *,
    args: tuple[onp.ToScalar | onp.ToArrayStrict1D, ...] = (),
    kwargs: Mapping[str, onp.ToScalar | onp.ToArrayStrict1D] | None = None,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat | onp.ToFloatStrict1D = 0.5,
    step_factor: onp.ToFloat = 2.0,
    step_direction: onp.ToJustInt | onp.ToJustIntStrict1D = 0,
    preserve_shape: Literal[False] = False,
    callback: Callable[[_DerivativeResultND[np.float64, tuple[int]]], Unused] | None = None,
) -> _DerivativeResultND[np.float64, tuple[int]]: ...
@overload  # 1d <known>
def derivative[FloatT: npc.floating](
    f: _Function11[FloatT],
    x: onp.Array1D[FloatT],
    *,
    args: tuple[onp.ToScalar | onp.ToArrayStrict1D, ...] = (),
    kwargs: Mapping[str, onp.ToScalar | onp.ToArrayStrict1D] | None = None,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat | onp.ToFloatStrict1D = 0.5,
    step_factor: onp.ToFloat = 2.0,
    step_direction: onp.ToJustInt | onp.ToJustIntStrict1D = 0,
    preserve_shape: Literal[False] = False,
    callback: Callable[[_DerivativeResultND[FloatT, tuple[int]]], Unused] | None = None,
) -> _DerivativeResultND[FloatT, tuple[int]]: ...
@overload  # Nd f64
def derivative(
    f: _FunctionNN[np.float64],
    x: _AsF64ND,
    *,
    args: _ToArgsND = (),
    kwargs: _ToKwargsND | None = None,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat | onp.ToFloatND = 0.5,
    step_factor: onp.ToFloat = 2.0,
    step_direction: onp.ToJustInt | onp.ToJustIntND = 0,
    preserve_shape: bool = False,
    callback: Callable[[_DerivativeResultND[np.float64, tuple[Any, ...]]], Unused] | None = None,
) -> _DerivativeResultND[np.float64, tuple[Any, ...]]: ...
@overload  # Nd <known>
def derivative[FloatT: npc.floating](
    f: _FunctionNN[FloatT],
    x: FloatT | onp.CanArrayND[FloatT],
    *,
    args: _ToArgsND = (),
    kwargs: _ToKwargsND | None = None,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat | onp.ToFloatND = 0.5,
    step_factor: onp.ToFloat = 2.0,
    step_direction: onp.ToJustInt | onp.ToJustIntND = 0,
    preserve_shape: bool = False,
    callback: Callable[[_DerivativeResultND[FloatT, tuple[Any, ...]]], Unused] | None = None,
) -> _DerivativeResultND[FloatT, tuple[Any, ...]]: ...

#
@overload  # f64
def jacobian(
    f: Callable[[onp.ArrayND[np.float64]], onp.ToFloat | onp.ToFloatND],
    x: _AsF64ND,
    *,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat | onp.ToFloatND = 0.5,
    step_factor: onp.ToFloat = 2.0,
    step_direction: onp.ToJustInt | onp.ToJustIntND = 0,
) -> _JacobianResult[np.float64, tuple[int, *tuple[Any, ...]]]: ...
@overload  # <known>
def jacobian[FloatT: npc.floating](
    f: Callable[[onp.ArrayND[FloatT]], onp.ToFloat | onp.ToFloatND],
    x: onp.CanArrayND[FloatT],
    *,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat | onp.ToFloatND = 0.5,
    step_factor: onp.ToFloat = 2.0,
    step_direction: onp.ToJustInt | onp.ToJustIntND = 0,
) -> _JacobianResult[FloatT, tuple[int, *tuple[Any, ...]]]: ...

#
@overload  # f64
def hessian(
    f: Callable[[onp.ArrayND[np.float64]], onp.ToFloat | onp.ToFloatND],
    x: _AsF64ND,
    *,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat | onp.ToFloatND = 0.5,
    step_factor: onp.ToFloat = 2.0,
) -> _HessianResult[np.float64, tuple[int, int, *tuple[Any, ...]]]: ...
@overload  # <known>
def hessian[FloatT: npc.floating](
    f: Callable[[onp.ArrayND[FloatT]], onp.ToFloat | onp.ToFloatND],
    x: onp.CanArrayND[FloatT],
    *,
    tolerances: _Tolerances | None = None,
    maxiter: ConvertibleToInt = 10,
    order: ConvertibleToInt = 8,
    initial_step: onp.ToFloat | onp.ToFloatND = 0.5,
    step_factor: onp.ToFloat = 2.0,
) -> _HessianResult[FloatT, tuple[int, int, *tuple[Any, ...]]]: ...
