from typing import Any, Final, Never, SupportsIndex, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["CZT", "ZoomFFT", "czt", "czt_points", "zoom_fft"]

###

type _ToInexact80ND = onp.ToArrayND[npc.inexact80, npc.inexact80]

# workaround for non-overload-spec-compliant type-checkers
type _JustAnyShape = tuple[Never, Never, Never, Never]
type _ToComplex128StrictND = onp.ArrayND[npc.number64 | npc.number32 | npc.number16 | npc.integer | np.bool, _JustAnyShape]

###

class CZT:
    w: Final[complex]
    a: Final[complex]
    m: Final[int]
    n: Final[int]

    def __init__(self, /, n: int, m: int | None = None, w: complex | None = None, a: complex = 1 + 0j) -> None: ...

    #
    @overload
    def __call__(self, /, x: _ToComplex128StrictND, *, axis: SupportsIndex = -1) -> onp.ArrayND[np.complex128]: ...
    @overload
    def __call__(self, /, x: onp.ToComplex128Strict1D, *, axis: SupportsIndex = -1) -> onp.Array1D[np.complex128]: ...
    @overload
    def __call__(self, /, x: onp.ToComplex128Strict2D, *, axis: SupportsIndex = -1) -> onp.Array2D[np.complex128]: ...
    @overload
    def __call__(self, /, x: onp.ToComplex128Strict3D, *, axis: SupportsIndex = -1) -> onp.Array3D[np.complex128]: ...
    @overload
    def __call__(self, /, x: _ToInexact80ND, *, axis: SupportsIndex = -1) -> onp.ArrayND[np.clongdouble]: ...
    @overload
    def __call__(self, /, x: onp.ToComplexND, *, axis: SupportsIndex = -1) -> onp.ArrayND[np.complex128 | Any]: ...

    #
    def points(self, /) -> onp.Array1D[np.complex128]: ...

class ZoomFFT(CZT):
    f1: float | Any
    f2: float | Any
    fs: float

    def __init__(
        self, /, n: int, fn: float | onp.ToFloat1D, m: int | None = None, *, fs: float = 2, endpoint: bool = False
    ) -> None: ...

#
def _validate_sizes(n: int, m: int | None) -> int: ...

#
def czt_points(m: int, w: complex | None = None, a: complex = 1 + 0j) -> onp.Array1D[np.complex128]: ...

#
@overload
def czt(
    x: _ToComplex128StrictND, m: int | None = None, w: complex | None = None, a: complex = 1 + 0j, *, axis: SupportsIndex = -1
) -> onp.ArrayND[np.complex128]: ...
@overload
def czt(
    x: onp.ToComplex128Strict1D, m: int | None = None, w: complex | None = None, a: complex = 1 + 0j, *, axis: SupportsIndex = -1
) -> onp.Array1D[np.complex128]: ...
@overload
def czt(
    x: onp.ToComplex128Strict2D, m: int | None = None, w: complex | None = None, a: complex = 1 + 0j, *, axis: SupportsIndex = -1
) -> onp.Array2D[np.complex128]: ...
@overload
def czt(
    x: onp.ToComplex128Strict3D, m: int | None = None, w: complex | None = None, a: complex = 1 + 0j, *, axis: SupportsIndex = -1
) -> onp.Array3D[np.complex128]: ...
@overload
def czt(
    x: _ToInexact80ND, m: int | None = None, w: complex | None = None, a: complex = 1 + 0j, *, axis: SupportsIndex = -1
) -> onp.ArrayND[np.clongdouble]: ...
@overload
def czt(
    x: onp.ToComplexND, m: int | None = None, w: complex | None = None, a: complex = 1 + 0j, *, axis: SupportsIndex = -1
) -> onp.ArrayND[np.complex128 | Any]: ...

#
@overload
def zoom_fft(
    x: _ToComplex128StrictND,
    fn: float | onp.ToFloat1D,
    m: int | None = None,
    *,
    fs: float = 2,
    endpoint: bool = False,
    axis: SupportsIndex = -1,
) -> onp.ArrayND[np.complex128]: ...
@overload
def zoom_fft(
    x: onp.ToComplex128Strict1D,
    fn: float | onp.ToFloat1D,
    m: int | None = None,
    *,
    fs: float = 2,
    endpoint: bool = False,
    axis: SupportsIndex = -1,
) -> onp.Array1D[np.complex128]: ...
@overload
def zoom_fft(
    x: onp.ToComplex128Strict2D,
    fn: float | onp.ToFloat1D,
    m: int | None = None,
    *,
    fs: float = 2,
    endpoint: bool = False,
    axis: SupportsIndex = -1,
) -> onp.Array2D[np.complex128]: ...
@overload
def zoom_fft(
    x: onp.ToComplex128Strict3D,
    fn: float | onp.ToFloat1D,
    m: int | None = None,
    *,
    fs: float = 2,
    endpoint: bool = False,
    axis: SupportsIndex = -1,
) -> onp.Array3D[np.complex128]: ...
@overload
def zoom_fft(
    x: _ToInexact80ND,
    fn: float | onp.ToFloat1D,
    m: int | None = None,
    *,
    fs: float = 2,
    endpoint: bool = False,
    axis: SupportsIndex = -1,
) -> onp.ArrayND[np.clongdouble]: ...
@overload
def zoom_fft(
    x: onp.ToComplexND,
    fn: float | onp.ToFloat1D,
    m: int | None = None,
    *,
    fs: float = 2,
    endpoint: bool = False,
    axis: SupportsIndex = -1,
) -> onp.ArrayND[np.complex128 | Any]: ...
