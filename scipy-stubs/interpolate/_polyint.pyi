from collections.abc import Callable, Sequence
from typing import ClassVar, TypeAlias, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import ToRNG

__all__ = [
    "BarycentricInterpolator",
    "KroghInterpolator",
    "approximate_taylor_polynomial",
    "barycentric_interpolate",
    "krogh_interpolate",
]

_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]
_Complex2D: TypeAlias = onp.Array2D[np.complex128]
_ComplexND: TypeAlias = onp.ArrayND[np.complex128]

###

class _Interpolator1D:  # undocumented
    __slots__: ClassVar[tuple[str, ...]] = "_y_axis", "_y_extra_shape", "dtype"

    _y_axis: int | bool | None
    _y_extra_shape: tuple[int | bool, ...] | None
    dtype: type[np.float64 | np.complex128]

    def __init__(
        self,
        /,
        xi: onp.ToFloatND | None = None,
        yi: onp.ToComplexND | None = None,
        axis: int | bool | None = None,
    ) -> None: ...
    def __call__(self, /, x: onp.ToFloat | onp.ToFloatND) -> _FloatND | _ComplexND: ...

class _Interpolator1DWithDerivatives(_Interpolator1D):  # undocumented
    def derivatives(self, /, x: onp.ToFloatND, der: int | bool | Sequence[int | bool] | None = None) -> _FloatND | _ComplexND: ...
    def derivative(self, /, x: onp.ToFloatND, der: int | bool = 1) -> _FloatND | _ComplexND: ...

class KroghInterpolator(_Interpolator1DWithDerivatives):
    xi: _Float1D
    yi: _FloatND | _ComplexND
    c: _Float2D | _Complex2D

    def __init__(self, /, xi: onp.ToFloatND, yi: onp.ToComplexND, axis: int | bool = 0) -> None: ...

class BarycentricInterpolator(_Interpolator1DWithDerivatives):
    n: int | bool
    xi: _Float1D
    yi: _FloatND | _ComplexND
    wi: _Float1D

    def __init__(
        self,
        /,
        xi: onp.ToFloat1D,
        yi: onp.ToComplexND | None = None,
        axis: int | bool = 0,
        *,
        wi: onp.ToFloatND | None = None,
        rng: ToRNG = None,
    ) -> None: ...
    def set_yi(self, /, yi: onp.ToComplexND, axis: int | bool | None = None) -> None: ...
    def add_xi(self, /, xi: onp.ToFloat1D, yi: onp.ToComplexND | None = None) -> None: ...

#
@overload
def krogh_interpolate(
    xi: onp.ToFloat1D,
    yi: onp.ToFloatND,
    x: onp.ToFloat | onp.ToFloat1D,
    der: onp.ToJustInt | onp.ToJustInt1D = 0,
    axis: int | bool = 0,
) -> _FloatND: ...
@overload
def krogh_interpolate(
    xi: onp.ToFloat1D,
    yi: onp.ToJustComplexND,
    x: onp.ToFloat | onp.ToFloat1D,
    der: onp.ToJustInt | onp.ToJustInt1D = 0,
    axis: int | bool = 0,
) -> _ComplexND: ...
@overload
def krogh_interpolate(
    xi: onp.ToFloat1D,
    yi: onp.ToComplexND,
    x: onp.ToFloat | onp.ToFloat1D,
    der: onp.ToJustInt | onp.ToJustInt1D = 0,
    axis: int | bool = 0,
) -> _FloatND | _ComplexND: ...

#
def approximate_taylor_polynomial(
    f: Callable[[_Float1D], onp.ToComplexND],
    x: onp.ToFloat,
    degree: onp.ToJustInt,
    scale: onp.ToFloat,
    order: onp.ToJustInt | None = None,
) -> np.poly1d: ...

#
@overload
def barycentric_interpolate(
    xi: onp.ToFloat1D,
    yi: onp.ToFloatND,
    x: onp.ToFloat,
    axis: int | bool = 0,
    *,
    der: onp.ToJustInt | onp.ToJustInt1D = 0,
    rng: ToRNG = None,
) -> np.float64: ...
@overload
def barycentric_interpolate(
    xi: onp.ToFloat1D,
    yi: onp.ToFloatND,
    x: onp.ToFloat1D,
    axis: int | bool = 0,
    *,
    der: onp.ToJustInt | onp.ToJustInt1D = 0,
    rng: ToRNG = None,
) -> _FloatND: ...
@overload
def barycentric_interpolate(
    xi: onp.ToFloat1D,
    yi: onp.ToJustComplexND,
    x: onp.ToFloat,
    axis: int | bool = 0,
    *,
    der: onp.ToJustInt | onp.ToJustInt1D = 0,
    rng: ToRNG = None,
) -> np.complex128: ...
@overload
def barycentric_interpolate(
    xi: onp.ToFloat1D,
    yi: onp.ToJustComplexND,
    x: onp.ToFloat1D,
    axis: int | bool = 0,
    *,
    der: onp.ToJustInt | onp.ToJustInt1D = 0,
    rng: ToRNG = None,
) -> _ComplexND: ...
@overload
def barycentric_interpolate(
    xi: onp.ToFloat1D,
    yi: onp.ToComplexND,
    x: onp.ToFloat,
    axis: int | bool = 0,
    *,
    der: onp.ToJustInt | onp.ToJustInt1D = 0,
    rng: ToRNG = None,
) -> np.float64 | np.complex128: ...
@overload
def barycentric_interpolate(
    xi: onp.ToFloat1D,
    yi: onp.ToComplexND,
    x: onp.ToFloat1D,
    axis: int | bool = 0,
    *,
    der: onp.ToJustInt | onp.ToJustInt1D = 0,
    rng: ToRNG = None,
) -> _FloatND | _ComplexND: ...
