from collections.abc import Sequence
from typing import Any, Literal, Never, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._typing import AnyShape

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

_Norm: TypeAlias = Literal["backward", "ortho", "forward"]
_Unused: TypeAlias = Never  # not used by scipy

_FloatingND: TypeAlias = onp.ArrayND[np.float32 | np.float64 | np.longdouble]
_ComplexND: TypeAlias = onp.ArrayND[npc.complexfloating]

_AsFloat32: TypeAlias = onp.CanArray[_ShapeT, np.dtype[npc.floating32]]
_AsFloat64: TypeAlias = onp.CanArray[_ShapeT, np.dtype[np.bool_ | npc.integer | npc.floating64]]
_AsFloat80: TypeAlias = onp.CanArray[_ShapeT, np.dtype[np.longdouble]]

_AsComplex64: TypeAlias = onp.CanArray[_ShapeT, np.dtype[npc.inexact32]]
_AsComplex128: TypeAlias = onp.CanArray[_ShapeT, np.dtype[np.bool_ | npc.integer | npc.inexact64]]
_AsComplex160: TypeAlias = onp.CanArray[_ShapeT, np.dtype[np.longdouble | np.clongdouble]]

_ToFloat64_ND: TypeAlias = onp.ToArrayND[float, np.bool_ | npc.integer | npc.floating64]
_ToComplex128_ND: TypeAlias = onp.ToArrayND[complex, np.bool_ | npc.integer | npc.inexact64]

###
# 1-D

# keep in sync with `ifft`
@overload
def fft(  # type: ignore[overload-overlap]
    x: _AsComplex64[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex64]: ...
@overload
def fft(
    x: _AsComplex128[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex128]: ...
@overload
def fft(
    x: _AsComplex160[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.clongdouble]: ...
@overload
def fft(
    x: Sequence[complex],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array1D[np.complex128]: ...
@overload
def fft(
    x: _ToComplex128_ND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128]: ...
@overload
def fft(
    x: onp.ToComplexND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128 | Any]: ...

# keep in sync with `fft`
@overload
def ifft(  # type: ignore[overload-overlap]
    x: _AsComplex64[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex64]: ...
@overload
def ifft(
    x: _AsComplex128[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex128]: ...
@overload
def ifft(
    x: _AsComplex160[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.clongdouble]: ...
@overload
def ifft(
    x: Sequence[complex],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array1D[np.complex128]: ...
@overload
def ifft(
    x: _ToComplex128_ND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128]: ...
@overload
def ifft(
    x: onp.ToComplexND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128 | Any]: ...

# keep in sync with `ihfft`
@overload
def rfft(  # type: ignore[overload-overlap]
    x: _AsFloat32[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex64]: ...
@overload
def rfft(
    x: _AsFloat64[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex128]: ...
@overload
def rfft(
    x: _AsFloat80[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.clongdouble]: ...
@overload
def rfft(
    x: Sequence[float],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array1D[np.complex128]: ...
@overload
def rfft(
    x: _ToFloat64_ND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128]: ...
@overload
def rfft(
    x: onp.ToFloatND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128 | Any]: ...

# keep in sync with `hfft`
@overload
def irfft(  # type: ignore[overload-overlap]
    x: _AsComplex64[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def irfft(
    x: _AsComplex128[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def irfft(
    x: _AsComplex160[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.longdouble]: ...
@overload
def irfft(
    x: Sequence[complex],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array1D[np.float64]: ...
@overload
def irfft(
    x: _ToComplex128_ND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def irfft(
    x: onp.ToComplexND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.float64 | Any]: ...

# keep in sync with `irfft`
@overload
def hfft(  # type: ignore[overload-overlap]
    x: _AsComplex64[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def hfft(
    x: _AsComplex128[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def hfft(
    x: _AsComplex160[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.longdouble]: ...
@overload
def hfft(
    x: Sequence[complex],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array1D[np.float64]: ...
@overload
def hfft(
    x: _ToComplex128_ND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def hfft(
    x: onp.ToComplexND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.float64 | Any]: ...

#
# keep in sync with `rfft`
@overload
def ihfft(  # type: ignore[overload-overlap]
    x: _AsFloat32[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex64]: ...
@overload
def ihfft(
    x: _AsFloat64[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex128]: ...
@overload
def ihfft(
    x: _AsFloat80[_ShapeT],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.clongdouble]: ...
@overload
def ihfft(
    x: Sequence[float],
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array1D[np.complex128]: ...
@overload
def ihfft(
    x: _ToFloat64_ND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128]: ...
@overload
def ihfft(
    x: onp.ToFloatND,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128 | Any]: ...

###
# 2-D

# keep in sync with `ifft2`
@overload
def fft2(  # type: ignore[overload-overlap]
    x: _AsComplex64[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex64]: ...
@overload
def fft2(
    x: _AsComplex128[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex128]: ...
@overload
def fft2(
    x: _AsComplex160[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.clongdouble]: ...
@overload
def fft2(
    x: Sequence[Sequence[complex]],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array2D[np.complex128]: ...
@overload
def fft2(
    x: _ToComplex128_ND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128]: ...
@overload
def fft2(
    x: onp.ToComplexND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128 | Any]: ...

# keep in sync with `fft2`
@overload
def ifft2(  # type: ignore[overload-overlap]
    x: _AsComplex64[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex64]: ...
@overload
def ifft2(
    x: _AsComplex128[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex128]: ...
@overload
def ifft2(
    x: _AsComplex160[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.clongdouble]: ...
@overload
def ifft2(
    x: Sequence[Sequence[complex]],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array2D[np.complex128]: ...
@overload
def ifft2(
    x: _ToComplex128_ND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128]: ...
@overload
def ifft2(
    x: onp.ToComplexND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128 | Any]: ...

# keep in sync with `ihfft2`
@overload
def rfft2(  # type: ignore[overload-overlap]
    x: _AsFloat32[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex64]: ...
@overload
def rfft2(
    x: _AsFloat64[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex128]: ...
@overload
def rfft2(
    x: _AsFloat80[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.clongdouble]: ...
@overload
def rfft2(
    x: Sequence[Sequence[float]],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array2D[np.complex128]: ...
@overload
def rfft2(
    x: _ToFloat64_ND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128]: ...
@overload
def rfft2(
    x: onp.ToFloatND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128 | Any]: ...

# keep in sync with `hfft2`
@overload
def irfft2(  # type: ignore[overload-overlap]
    x: _AsComplex64[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def irfft2(
    x: _AsComplex128[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def irfft2(
    x: _AsComplex160[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.longdouble]: ...
@overload
def irfft2(
    x: Sequence[Sequence[complex]],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array2D[np.float64]: ...
@overload
def irfft2(
    x: _ToComplex128_ND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def irfft2(
    x: onp.ToComplexND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.float64 | Any]: ...

# keep in sync with `irfft2`
@overload
def hfft2(  # type: ignore[overload-overlap]
    x: _AsComplex64[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.float32]: ...
@overload
def hfft2(
    x: _AsComplex128[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def hfft2(
    x: _AsComplex160[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.longdouble]: ...
@overload
def hfft2(
    x: Sequence[Sequence[complex]],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array2D[np.float64]: ...
@overload
def hfft2(
    x: _ToComplex128_ND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload
def hfft2(
    x: onp.ToComplexND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.float64 | Any]: ...

# keep in sync with `rfft2`
@overload
def ihfft2(  # type: ignore[overload-overlap]
    x: _AsFloat32[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex64]: ...
@overload
def ihfft2(
    x: _AsFloat64[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.complex128]: ...
@overload
def ihfft2(
    x: _AsFloat80[_ShapeT],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array[_ShapeT, np.clongdouble]: ...
@overload
def ihfft2(
    x: Sequence[Sequence[float]],
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.Array2D[np.complex128]: ...
@overload
def ihfft2(
    x: _ToFloat64_ND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128]: ...
@overload
def ihfft2(
    x: onp.ToFloatND,
    s: onp.ToJustInt1D | None = None,
    axes: AnyShape = (-2, -1),
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> onp.ArrayND[np.complex128 | Any]: ...

###
# N-D

def fftn(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> _ComplexND: ...
def ifftn(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> _ComplexND: ...
def rfftn(
    x: onp.ToFloatND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> _ComplexND: ...
def irfftn(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> _FloatingND: ...
def hfftn(
    x: onp.ToComplexND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> _FloatingND: ...
def ihfftn(
    x: onp.ToFloatND,
    s: onp.ToIntND | None = None,
    axes: AnyShape | None = None,
    norm: _Norm | None = None,
    overwrite_x: onp.ToBool = False,
    workers: int | None = None,
    *,
    plan: _Unused | None = None,
) -> _ComplexND: ...
