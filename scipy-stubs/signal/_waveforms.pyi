from collections.abc import Iterable
from typing import Any, Literal, SupportsIndex, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._typing import AnyShape

__all__ = ["chirp", "gausspulse", "sawtooth", "square", "sweep_poly", "unit_impulse"]

###

# workaround for a strange bug in pyright's overlapping overload detection with `numpy<2.1`
type _WorkaroundForPyright = tuple[int] | tuple[Any, ...]

type _Tuple2[T] = tuple[T, T]
type _Tuple3[T] = tuple[T, T, T]

type _ToFloat = npc.floating | npc.integer | np.bool
type _ToFloat0ND = onp.ToFloat | onp.ToFloatND

type _ToIdx = SupportsIndex | Iterable[SupportsIndex] | Literal["mid"]
type _ChirpMethod = Literal["linear", "quadratic", "logarithmic", "hyperbolic"]

###

#
@overload  # 0d T
def sawtooth[FloatT: npc.floating](t: FloatT, width: float = 1.0) -> onp.Array0D[FloatT]: ...
@overload  # Nd T
def sawtooth[FloatT: npc.floating, ShapeT: tuple[int, ...]](
    t: onp.ArrayND[FloatT, ShapeT], width: float = 1.0
) -> onp.ArrayND[FloatT, ShapeT]: ...
@overload  # T
def sawtooth[FloatT: npc.floating](
    t: FloatT | onp.ArrayND[FloatT], width: float = 1.0
) -> onp.ArrayND[FloatT, _WorkaroundForPyright]: ...
@overload  # 0d +int
def sawtooth(t: float | onp.ToInt, width: float = 1.0) -> onp.Array0D[np.float64]: ...
@overload  # Nd +int
def sawtooth[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[npc.integer | np.bool, ShapeT], width: float = 1.0
) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # nd +int
def sawtooth(t: onp.SequenceND[float] | onp.ToIntND, width: float = 1.0) -> onp.ArrayND[np.float64, _WorkaroundForPyright]: ...
@overload  # +f64  (fallback)
def sawtooth(t: _ToFloat0ND, width: _ToFloat0ND = 1.0) -> onp.ArrayND[np.float64 | Any, _WorkaroundForPyright]: ...

#
@overload  # 0d T
def square[FloatT: npc.floating](t: FloatT, duty: float = 0.5) -> onp.Array0D[FloatT]: ...
@overload  # Nd T
def square[FloatT: npc.floating, ShapeT: tuple[int, ...]](
    t: onp.ArrayND[FloatT, ShapeT], duty: float = 0.5
) -> onp.ArrayND[FloatT, ShapeT]: ...
@overload  # T
def square[FloatT: npc.floating](
    t: FloatT | onp.ArrayND[FloatT], duty: float = 0.5
) -> onp.ArrayND[FloatT, _WorkaroundForPyright]: ...
@overload  # 0d +int
def square(t: float | onp.ToInt, duty: float = 0.5) -> onp.Array0D[np.float64]: ...
@overload  # Nd +int
def square[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[npc.integer | np.bool, ShapeT], duty: float = 0.5
) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # nd +int
def square(t: onp.SequenceND[float] | onp.ToIntND, duty: float = 0.5) -> onp.ArrayND[np.float64, _WorkaroundForPyright]: ...
@overload  # fallback
def square(t: _ToFloat0ND, duty: _ToFloat0ND = 0.5) -> onp.ArrayND[np.float64 | Any, _WorkaroundForPyright]: ...

#
@overload  # +f64 0d, complex: False
def chirp(
    t: onp.ToFloat64,
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: Literal[False] = False,
) -> np.float64: ...
@overload  # +f64 Nd, complex: False
def chirp[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[npc.floating64 | npc.floating32 | npc.floating16 | npc.integer | np.bool, ShapeT],
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: Literal[False] = False,
) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # +f64 nd, complex: False
def chirp(
    t: onp.ToFloat64_ND,
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: Literal[False] = False,
) -> onp.ArrayND[np.float64, _WorkaroundForPyright]: ...
@overload  # +c128 0d, complex: True
def chirp(
    t: onp.ToComplex128,
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: Literal[True],
) -> np.complex128: ...
@overload  # +c128 Nd, complex: True
def chirp[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[npc.number64 | npc.number32 | npc.number16 | npc.integer | np.bool, ShapeT],
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: Literal[True],
) -> onp.ArrayND[np.complex128, ShapeT]: ...
@overload  # +c128 nd, complex: True
def chirp(
    t: onp.ToComplex128_ND,
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: Literal[True],
) -> onp.ArrayND[np.complex128, _WorkaroundForPyright]: ...
@overload  # ~c128 0d
def chirp(
    t: onp.ToJustComplex128 | np.complex64,
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: bool = False,
) -> np.complex128: ...
@overload  # ~c128 Nd
def chirp[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[npc.complexfloating128 | npc.complexfloating64, ShapeT],
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: bool = False,
) -> onp.ArrayND[np.complex128, ShapeT]: ...
@overload  # ~c128 nd
def chirp(
    t: onp.ToJustComplex128_ND | onp.ToJustComplex64_ND,
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload  # fallback (known shape)
def chirp[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[npc.number | np.bool, ShapeT],
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: bool = False,
) -> onp.ArrayND[Any, ShapeT]: ...
@overload  # fallback (unknown shape)
def chirp(
    t: onp.ToComplexND,
    f0: float,
    t1: float,
    f1: float,
    method: _ChirpMethod = "linear",
    phi: float = 0,
    vertex_zero: bool = True,
    *,
    complex: bool = False,
) -> onp.ArrayND[Any, _WorkaroundForPyright]: ...

#
@overload  # 0d
def sweep_poly(t: onp.ToFloat, poly: onp.ToFloatND | np.poly1d, phi: onp.ToFloat = 0) -> np.float64: ...
@overload  # Nd
def sweep_poly[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[_ToFloat, ShapeT], poly: onp.ToFloatND | np.poly1d, phi: onp.ToFloat = 0
) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # nd
def sweep_poly(
    t: onp.ToFloatND, poly: onp.ToFloatND | np.poly1d, phi: onp.ToFloat = 0
) -> onp.ArrayND[np.float64, _WorkaroundForPyright]: ...

#
@overload  # shape: 1d, dtype is not given
def unit_impulse(shape: onp.ToInt, idx: _ToIdx | None = None, dtype: type[float] = ...) -> onp.Array1D[np.float64]: ...
@overload  # shape: T, dtype is not given
def unit_impulse[ShapeT: tuple[int, ...]](
    shape: ShapeT, idx: _ToIdx | None = None, dtype: type[float] = ...
) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # shape: nd, dtype is not given
def unit_impulse(
    shape: AnyShape, idx: _ToIdx | None = None, dtype: type[float] = ...
) -> onp.ArrayND[np.float64, _WorkaroundForPyright]: ...
@overload  # shape: 1d, dtype is given
def unit_impulse[ScalarT: np.generic](
    shape: onp.ToInt, idx: _ToIdx | None, dtype: onp.ToDType[ScalarT]
) -> onp.Array1D[ScalarT]: ...
@overload  # shape: T, dtype is given
def unit_impulse[ScalarT: np.generic, ShapeT: tuple[int, ...]](
    shape: ShapeT, idx: _ToIdx | None, dtype: onp.ToDType[ScalarT]
) -> onp.ArrayND[ScalarT, ShapeT]: ...
@overload  # shape: nd, dtype is given
def unit_impulse[ScalarT: np.generic](
    shape: AnyShape, idx: _ToIdx | None, dtype: onp.ToDType[ScalarT]
) -> onp.ArrayND[ScalarT, _WorkaroundForPyright]: ...

#
@overload  # "cutoff"
def gausspulse(
    t: Literal["cutoff"],
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: bool = False,
    retenv: bool = False,
) -> np.float64: ...
@overload  # 0d
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: onp.ToFalse = False,
    retenv: onp.ToFalse = False,
) -> np.float64: ...
@overload  # 0d, retenv
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: onp.ToFalse = False,
    *,
    retenv: onp.ToTrue,
) -> tuple[np.float64, np.float64]: ...
@overload  # 0d, retquad
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    *,
    retquad: onp.ToTrue,
    retenv: onp.ToFalse = False,
) -> tuple[np.float64, np.float64]: ...
@overload  # 0d, retquad, retenv
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    *,
    retquad: onp.ToTrue,
    retenv: onp.ToTrue,
) -> tuple[np.float64, np.float64, np.float64]: ...
@overload  # Nd
def gausspulse[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[_ToFloat, ShapeT],
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: onp.ToFalse = False,
    retenv: onp.ToFalse = False,
) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd, retenv
def gausspulse[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[_ToFloat, ShapeT],
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: onp.ToFalse = False,
    *,
    retenv: onp.ToTrue,
) -> _Tuple2[onp.ArrayND[np.float64, ShapeT]]: ...
@overload  # Nd, retquad
def gausspulse[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[_ToFloat, ShapeT],
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    *,
    retquad: onp.ToTrue,
    retenv: onp.ToFalse = False,
) -> _Tuple2[onp.ArrayND[np.float64, ShapeT]]: ...
@overload  # Nd, retquad, retenv
def gausspulse[ShapeT: tuple[int, ...]](
    t: onp.ArrayND[_ToFloat, ShapeT],
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    *,
    retquad: onp.ToTrue,
    retenv: onp.ToTrue,
) -> _Tuple3[onp.ArrayND[np.float64, ShapeT]]: ...
@overload  # nd
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: onp.ToFalse = False,
    retenv: onp.ToFalse = False,
) -> onp.ArrayND[np.float64, _WorkaroundForPyright]: ...
@overload  # nd, retenv
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: onp.ToFalse = False,
    *,
    retenv: onp.ToTrue,
) -> _Tuple2[onp.ArrayND[np.float64, _WorkaroundForPyright]]: ...
@overload  # nd, retquad
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    *,
    retquad: onp.ToTrue,
    retenv: onp.ToFalse = False,
) -> _Tuple2[onp.ArrayND[np.float64, _WorkaroundForPyright]]: ...
@overload  # nd, retquad, retenv
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    *,
    retquad: onp.ToTrue,
    retenv: onp.ToTrue,
) -> _Tuple3[onp.ArrayND[np.float64, _WorkaroundForPyright]]: ...
