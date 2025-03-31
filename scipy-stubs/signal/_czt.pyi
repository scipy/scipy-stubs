from typing import Final, TypeAlias, overload

import numpy as np
import optype as op
import optype.numpy as onp

__all__ = ["CZT", "ZoomFFT", "czt", "czt_points", "zoom_fft"]

_Complex: TypeAlias = np.complex128 | np.clongdouble

###

class CZT:
    w: Final[complex | float | int | bool | np.complex128]
    a: Final[complex | float | int | bool | np.complex128]
    m: Final[int | bool]
    n: Final[int | bool]

    def __init__(
        self,
        /,
        n: int | bool,
        m: int | bool | None = None,
        w: complex | float | int | bool | np.complex128 | None = None,
        a: complex | float | int | bool | np.complex128 = 1 + 0j,
    ) -> None: ...
    @overload
    def __call__(self, /, x: onp.ToComplexStrict1D, *, axis: op.CanIndex = -1) -> onp.Array1D[_Complex]: ...
    @overload
    def __call__(self, /, x: onp.ToComplexStrict2D, *, axis: op.CanIndex = -1) -> onp.Array2D[_Complex]: ...
    @overload
    def __call__(self, /, x: onp.ToComplexStrict3D, *, axis: op.CanIndex = -1) -> onp.Array3D[_Complex]: ...
    @overload
    def __call__(self, /, x: onp.ToComplexND, *, axis: op.CanIndex = -1) -> onp.ArrayND[_Complex]: ...
    def points(self, /) -> onp.Array1D[np.complex128]: ...

class ZoomFFT(CZT):
    f1: onp.ToFloat
    f2: onp.ToFloat
    fs: float | int | bool | np.float64

    def __init__(
        self,
        /,
        n: int | bool,
        fn: float | int | bool | np.float64 | onp.ToFloat1D,
        m: int | bool | None = None,
        *,
        fs: float | int | bool | np.float64 = 2,
        endpoint: onp.ToBool = False,
    ) -> None: ...

#
def _validate_sizes(n: int | bool, m: int | bool | None) -> int | bool: ...

#
def czt_points(
    m: int | bool,
    w: complex | float | int | bool | np.complex128 | None = None,
    a: complex | float | int | bool | np.complex128 = 1 + 0j,
) -> onp.Array1D[np.complex128]: ...

#
@overload
def czt(
    x: onp.ToComplexStrict1D,
    m: int | bool | None = None,
    w: complex | float | int | bool | np.complex128 | None = None,
    a: complex | float | int | bool | np.complex128 = 1 + 0j,
    *,
    axis: op.CanIndex = -1,
) -> onp.Array1D[_Complex]: ...
@overload
def czt(
    x: onp.ToComplexStrict2D,
    m: int | bool | None = None,
    w: complex | float | int | bool | np.complex128 | None = None,
    a: complex | float | int | bool | np.complex128 = 1 + 0j,
    *,
    axis: op.CanIndex = -1,
) -> onp.Array2D[_Complex]: ...
@overload
def czt(
    x: onp.ToComplexStrict3D,
    m: int | bool | None = None,
    w: complex | float | int | bool | np.complex128 | None = None,
    a: complex | float | int | bool | np.complex128 = 1 + 0j,
    *,
    axis: op.CanIndex = -1,
) -> onp.Array3D[_Complex]: ...
@overload
def czt(
    x: onp.ToComplexND,
    m: int | bool | None = None,
    w: complex | float | int | bool | np.complex128 | None = None,
    a: complex | float | int | bool | np.complex128 = 1 + 0j,
    *,
    axis: op.CanIndex = -1,
) -> onp.ArrayND[_Complex]: ...

#
@overload
def zoom_fft(
    x: onp.ToComplexStrict1D,
    fn: float | int | bool | np.float64 | onp.ToFloat1D,
    m: int | bool | None = None,
    *,
    fs: float | int | bool | np.float64 = 2,
    endpoint: onp.ToBool = False,
    axis: op.CanIndex = -1,
) -> onp.Array1D[_Complex]: ...
@overload
def zoom_fft(
    x: onp.ToComplexStrict2D,
    fn: float | int | bool | np.float64 | onp.ToFloat1D,
    m: int | bool | None = None,
    *,
    fs: float | int | bool | np.float64 = 2,
    endpoint: onp.ToBool = False,
    axis: op.CanIndex = -1,
) -> onp.Array2D[_Complex]: ...
@overload
def zoom_fft(
    x: onp.ToComplexStrict3D,
    fn: float | int | bool | np.float64 | onp.ToFloat1D,
    m: int | bool | None = None,
    *,
    fs: float | int | bool | np.float64 = 2,
    endpoint: onp.ToBool = False,
    axis: op.CanIndex = -1,
) -> onp.Array3D[_Complex]: ...
@overload
def zoom_fft(
    x: onp.ToComplexND,
    fn: float | int | bool | np.float64 | onp.ToFloat1D,
    m: int | bool | None = None,
    *,
    fs: float | int | bool | np.float64 = 2,
    endpoint: onp.ToBool = False,
    axis: op.CanIndex = -1,
) -> onp.ArrayND[_Complex]: ...
