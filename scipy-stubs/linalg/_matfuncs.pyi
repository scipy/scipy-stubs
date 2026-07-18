from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, Never, overload
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._expm_frechet import expm_cond, expm_frechet

__all__ = [
    "coshm",
    "cosm",
    "expm",
    "expm_cond",
    "expm_frechet",
    "fractional_matrix_power",
    "funm",
    "khatri_rao",
    "logm",
    "signm",
    "sinhm",
    "sinm",
    "sqrtm",
    "tanhm",
    "tanm",
]

###

# always called with the 1-d diagonal of the (complex) schur form; the return value is unsafely cast back to its dtype
type _Func1D[ComplexT: npc.complexfloating] = Callable[[onp.Array1D[ComplexT]], onp.ToComplexND]

type _ToPosInt = npc.unsignedinteger | Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]

type _Float64ND = onp.ArrayND[np.float64]
type _FloatND = onp.ArrayND[npc.floating]
type _Complex128ND = onp.ArrayND[np.complex128]
type _ComplexND = onp.ArrayND[npc.complexfloating]

# workaround for Pyrefly (1.1.1) that doesn't want to assign `tuple[Any, ...]` to `T: onp.AtLeast2D`.
type _AtLeast2D_ish = tuple[int, int, *tuple[int, ...]] | tuple[Any, ...]
# workaround for https://github.com/microsoft/pyright/issues/10232 (only matches gradually-shaped arrays)
type _JustAnyShape = tuple[Never, Never, Never, Never]
# workaround for Pyright (1.1.411) false positive reportIncompatibleOverload on `numpy<2.1`
type _AnyShapeOrTriviallyMaybeAlso2D = tuple[int, int] | tuple[Any, ...]

###

eps: Final[np.float64] = ...  # undocumented
feps: Final[np.float32] = ...  # undocumented
_array_precision: Final[dict[Literal["i", "l", "f", "d", "F", "D"], Literal[0, 1]]] = ...  # undocumented

def _asarray_square[InexactT: npc.inexact](A: onp.ToArray2D[InexactT]) -> onp.Array2D[InexactT]: ...  # undocumented

#
@overload
def _maybe_real[ShapeT: tuple[int, ...]](
    A: onp.ArrayND[npc.inexact], B: onp.ArrayND[npc.inexact64, ShapeT], tol: float | None = None
) -> onp.ArrayND[np.float64, ShapeT]: ...  # undocumented
@overload
def _maybe_real[ShapeT: tuple[int, ...]](
    A: onp.ArrayND[npc.inexact], B: onp.ArrayND[npc.inexact32, ShapeT], tol: float | None = None
) -> onp.ArrayND[np.float32, ShapeT]: ...  # undocumented

# NOTE: with fractional `t`, real input can have either real or complex output depending on the sign of the eigenvalues
@overload  # Nd T, +int t
def fractional_matrix_power[NumberT: npc.inexact64 | npc.inexact32 | npc.integer, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[NumberT, ShapeT], t: _ToPosInt
) -> onp.ArrayND[NumberT, ShapeT]: ...
@overload  # Nd T@inexact, int t
def fractional_matrix_power[InexactT: npc.inexact64 | npc.inexact32, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[InexactT, ShapeT], t: onp.ToInt
) -> onp.ArrayND[InexactT, ShapeT]: ...
@overload  # Nd ~int, int t
def fractional_matrix_power[IntT: npc.integer, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[IntT, ShapeT], t: onp.ToInt
) -> onp.ArrayND[IntT | np.float64, ShapeT]: ...
@overload  # Nd +f64, ~float t
def fractional_matrix_power[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.floating64 | npc.floating32 | npc.integer, ShapeT], t: onp.ToJustFloat
) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd c128 | c64, ~float t
def fractional_matrix_power[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.complexfloating128 | npc.complexfloating64, ShapeT], t: onp.ToJustFloat
) -> onp.ArrayND[np.complex128, ShapeT]: ...
@overload  # Nd T, +int t
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 1.20")
def fractional_matrix_power[NumberT: npc.inexact80 | np.float16 | np.bool, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[NumberT, ShapeT], t: _ToPosInt
) -> onp.ArrayND[NumberT, ShapeT]: ...
@overload  # Nd bool, int t
@deprecated("bool input will no longer be supported in SciPy 1.20")
def fractional_matrix_power[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[np.bool, ShapeT], t: onp.ToInt
) -> onp.ArrayND[np.bool | np.float64, ShapeT]: ...
@overload  # Nd bool, ~float t
@deprecated("bool input will no longer be supported in SciPy 1.20")
def fractional_matrix_power[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[np.bool, ShapeT], t: onp.ToJustFloat
) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd f16, ~float t
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def fractional_matrix_power[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.floating16, ShapeT], t: onp.ToJustFloat
) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd f80, ~float t
@deprecated("longdouble input will no longer be supported in SciPy 1.20")
def fractional_matrix_power[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.floating80, ShapeT], t: onp.ToJustFloat
) -> onp.ArrayND[np.longdouble | np.clongdouble, ShapeT]: ...
@overload  # Nd c160, ~float t
@deprecated("clongdouble input will no longer be supported in SciPy 1.20")
def fractional_matrix_power[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.complexfloating160, ShapeT], t: onp.ToJustFloat
) -> onp.ArrayND[np.clongdouble, ShapeT]: ...
@overload  # 2d +float, int t
def fractional_matrix_power(A: Sequence[Sequence[float]], t: onp.ToInt) -> onp.Array2D[np.float64]: ...
@overload  # 2d +float, ~float t
def fractional_matrix_power(A: Sequence[Sequence[float]], t: onp.ToJustFloat) -> onp.Array2D[np.float64 | np.complex128]: ...
@overload  # 2d ~complex
def fractional_matrix_power(A: Sequence[list[complex]], t: onp.ToFloat) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float, int t
def fractional_matrix_power(A: Sequence[Sequence[Sequence[float]]], t: onp.ToInt) -> onp.Array3D[np.float64]: ...
@overload  # 3d +float, ~float t
def fractional_matrix_power(
    A: Sequence[Sequence[Sequence[float]]], t: onp.ToJustFloat
) -> onp.Array3D[np.float64 | np.complex128]: ...
@overload  # 3d ~complex
def fractional_matrix_power(A: Sequence[Sequence[list[complex]]], t: onp.ToFloat) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def fractional_matrix_power(A: onp.ToComplexND, t: onp.ToFloat) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

# mypy reports two false positive `overload-overlap` errors on `numpy>=2.2` for `sqrtm`, so we're forced to ignore it module-wide
# mypy: disable-error-code="overload-overlap"

# NOTE: real input can have either real or complex output depending on the sign of the values
@overload  # Nd ~f64
def sqrtm[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT],
) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd ~f32
def sqrtm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating32, ShapeT]) -> onp.ArrayND[np.float32 | np.complex64, ShapeT]: ...
@overload  # Nd T@(c128|c64)
def sqrtm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT],
) -> onp.ArrayND[ComplexT, ShapeT]: ...
@overload  # Nd bool
@deprecated("bool input will no longer be supported in SciPy 1.20")
def sqrtm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[np.bool, ShapeT]) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd f80
@deprecated("longdouble input will no longer be supported in SciPy 1.20")
def sqrtm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating80, ShapeT]) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd f16
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def sqrtm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating16, ShapeT]) -> onp.ArrayND[np.float32 | np.complex64, ShapeT]: ...
@overload  # Nd c160
@deprecated("clongdouble input will no longer be supported in SciPy 1.20")
def sqrtm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.complexfloating160, ShapeT]) -> onp.ArrayND[np.complex128, ShapeT]: ...
@overload  # 2d +float
def sqrtm(A: Sequence[Sequence[float]]) -> onp.Array2D[np.float64 | np.complex128]: ...
@overload  # 2d ~complex
def sqrtm(A: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def sqrtm(A: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64 | np.complex128]: ...
@overload  # 3d ~complex
def sqrtm(A: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def sqrtm(A: onp.ToComplexND) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

# NOTE: real input can have either real or complex output depending on the sign of the values
@overload  # Nd +f64 (except bool | f16)
def logm[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.floating64 | npc.floating32 | npc.integer, ShapeT],
) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd c128 | c64
def logm[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.complexfloating128 | npc.complexfloating64, ShapeT],
) -> onp.ArrayND[np.complex128, ShapeT]: ...
@overload  # Nd bool
@deprecated("bool input will no longer be supported in SciPy 1.20")
def logm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[np.bool, ShapeT]) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd f80
@deprecated("longdouble input will no longer be supported in SciPy 1.20")
def logm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating80, ShapeT]) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd f16
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def logm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating16, ShapeT]) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd c160
@deprecated("clongdouble input will no longer be supported in SciPy 1.20")
def logm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.complexfloating160, ShapeT]) -> onp.ArrayND[np.complex128, ShapeT]: ...
@overload  # 2d +float
def logm(A: Sequence[Sequence[float]]) -> onp.Array2D[np.float64 | np.complex128]: ...
@overload  # 2d ~complex
def logm(A: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def logm(A: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64 | np.complex128]: ...
@overload  # 3d ~complex
def logm(A: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def logm(A: onp.ToComplexND) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

# keep in sync with `cosm`, `sinm`, and `tanm`
@overload  # Nd ~f64
def expm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd ~f32
def expm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating32, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd T@(c128|c64)
def expm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT],
) -> onp.ArrayND[ComplexT, ShapeT]: ...
@overload  # Nd bool
@deprecated("bool input will no longer be supported in SciPy 1.20")
def expm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[np.bool, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd f16
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def expm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating16, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # 2d +float
def expm(A: Sequence[Sequence[float]]) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~complex
def expm(A: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def expm(A: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64]: ...
@overload  # 3d ~complex
def expm(A: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def expm(A: onp.ToComplexND) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

#
def _exp_sinch[ComplexT: npc.complexfloating, ShapeT: tuple[int, ...]](
    x: onp.ArrayND[ComplexT, ShapeT],
) -> onp.ArrayND[ComplexT, ShapeT]: ...  # undocumented

# keep in sync with `expm`
@overload  # Nd ~f64
def cosm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd ~f32
def cosm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating32, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd T@(c128|c64)
def cosm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT],
) -> onp.ArrayND[ComplexT, ShapeT]: ...
@overload  # Nd bool
@deprecated("bool input will no longer be supported in SciPy 1.20")
def cosm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[np.bool, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd f16
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def cosm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating16, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # 2d +float
def cosm(A: Sequence[Sequence[float]]) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~complex
def cosm(A: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def cosm(A: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64]: ...
@overload  # 3d ~complex
def cosm(A: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def cosm(A: onp.ToComplexND) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

# keep in sync with `expm`
@overload  # Nd ~f64
def sinm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd ~f32
def sinm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating32, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd T@(c128|c64)
def sinm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT],
) -> onp.ArrayND[ComplexT, ShapeT]: ...
@overload  # Nd bool
@deprecated("bool input will no longer be supported in SciPy 1.20")
def sinm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[np.bool, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd f16
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def sinm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating16, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # 2d +float
def sinm(A: Sequence[Sequence[float]]) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~complex
def sinm(A: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def sinm(A: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64]: ...
@overload  # 3d ~complex
def sinm(A: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def sinm(A: onp.ToComplexND) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

# keep in sync with `expm`
@overload  # Nd ~f64
def tanm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd ~f32
def tanm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating32, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd T@(c128|c64)
def tanm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT],
) -> onp.ArrayND[ComplexT, ShapeT]: ...
@overload  # Nd bool
@deprecated("bool input will no longer be supported in SciPy 1.20")
def tanm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[np.bool, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd f16
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def tanm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating16, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # 2d +float
def tanm(A: Sequence[Sequence[float]]) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~complex
def tanm(A: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def tanm(A: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64]: ...
@overload  # 3d ~complex
def tanm(A: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def tanm(A: onp.ToComplexND) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

# keep in sync with `sinhm` and `tanhm` (and `expm`, minus the `bool` overload)
@overload  # Nd ~f64
def coshm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd ~f32
def coshm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating32, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd T@(c128|c64)
def coshm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT],
) -> onp.ArrayND[ComplexT, ShapeT]: ...
@overload  # Nd f16
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def coshm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating16, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # 2d +float
def coshm(A: Sequence[Sequence[float]]) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~complex
def coshm(A: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def coshm(A: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64]: ...
@overload  # 3d ~complex
def coshm(A: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def coshm(A: onp.ToComplexND) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

# keep in sync with `coshm`
@overload  # Nd ~f64
def sinhm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd ~f32
def sinhm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating32, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd T@(c128|c64)
def sinhm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT],
) -> onp.ArrayND[ComplexT, ShapeT]: ...
@overload  # Nd f16
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def sinhm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating16, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # 2d +float
def sinhm(A: Sequence[Sequence[float]]) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~complex
def sinhm(A: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def sinhm(A: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64]: ...
@overload  # 3d ~complex
def sinhm(A: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def sinhm(A: onp.ToComplexND) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

# keep in sync with `coshm`
@overload  # Nd ~f64
def tanhm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd ~f32
def tanhm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating32, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd T@(c128|c64)
def tanhm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT],
) -> onp.ArrayND[ComplexT, ShapeT]: ...
@overload  # Nd f16
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def tanhm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating16, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # 2d +float
def tanhm(A: Sequence[Sequence[float]]) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~complex
def tanhm(A: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def tanhm(A: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64]: ...
@overload  # 3d ~complex
def tanhm(A: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def tanhm(A: onp.ToComplexND) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

# NOTE: real input can have either real or complex output, depending on `func` and the input values
@overload  # Nd +f64
def funm[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT], func: _Func1D[np.complex128], disp: onp.ToTrue = True
) -> onp.ArrayND[np.float64 | np.complex128, ShapeT]: ...
@overload  # Nd +f64, disp=False
def funm[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT], func: _Func1D[np.complex128], disp: onp.ToFalse
) -> tuple[onp.ArrayND[np.float64 | np.complex128, ShapeT], float]: ...
@overload  # Nd ~f32
def funm[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.floating32, ShapeT], func: _Func1D[np.complex64], disp: onp.ToTrue = True
) -> onp.ArrayND[np.float32 | np.complex64, ShapeT]: ...
@overload  # Nd ~f32, disp=False
def funm[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[npc.floating32, ShapeT], func: _Func1D[np.complex64], disp: onp.ToFalse
) -> tuple[onp.ArrayND[np.float32 | np.complex64, ShapeT], float]: ...
@overload  # Nd T@(c128|c64)
def funm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT], func: _Func1D[ComplexT], disp: onp.ToTrue = True
) -> onp.ArrayND[ComplexT, ShapeT]: ...
@overload  # Nd T@(c128|c64), disp=False
def funm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT], func: _Func1D[ComplexT], disp: onp.ToFalse
) -> tuple[onp.ArrayND[ComplexT, ShapeT], float]: ...
@overload  # Nd bool | f16 | f80 | c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 1.20")
def funm[ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[np.bool | npc.floating16 | npc.inexact80, ShapeT], func: _Func1D[Any], disp: onp.ToTrue = True
) -> onp.ArrayND[npc.inexact64 | npc.inexact32, ShapeT]: ...
@overload  # 2d +float
def funm(
    A: Sequence[Sequence[float]], func: _Func1D[np.complex128], disp: onp.ToTrue = True
) -> onp.Array2D[np.float64 | np.complex128]: ...
@overload  # 2d ~complex
def funm(A: Sequence[list[complex]], func: _Func1D[np.complex128], disp: onp.ToTrue = True) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def funm(
    A: Sequence[Sequence[Sequence[float]]], func: _Func1D[np.complex128], disp: onp.ToTrue = True
) -> onp.Array3D[np.float64 | np.complex128]: ...
@overload  # 3d ~complex
def funm(
    A: Sequence[Sequence[list[complex]]], func: _Func1D[np.complex128], disp: onp.ToTrue = True
) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def funm(
    A: onp.ToComplexND, func: _Func1D[Any], disp: onp.ToTrue = True
) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...
@overload  # Nd +c, disp=False  (fallback)
def funm(
    A: onp.ToComplexND, func: _Func1D[Any], disp: onp.ToFalse
) -> tuple[onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D], float]: ...

# NOTE: at runtime the out dtype is value-dependent, which we ignore here, because it seems to be unintentional.
# https://github.com/scipy/scipy/issues/25657
@overload  # Nd ~f64
def signm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating64 | npc.integer, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd ~f32
def signm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating32, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd T@(c128|c64)
def signm[ComplexT: np.complex128 | np.complex64, ShapeT: _AtLeast2D_ish](
    A: onp.ArrayND[ComplexT, ShapeT],
) -> onp.ArrayND[ComplexT, ShapeT]: ...
@overload  # Nd bool
@deprecated("bool input will no longer be supported in SciPy 1.20")
def signm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[np.bool, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd f80
@deprecated("longdouble input will no longer be supported in SciPy 1.20")
def signm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating80, ShapeT]) -> onp.ArrayND[np.float64, ShapeT]: ...
@overload  # Nd f16
@deprecated("float16 input will no longer be supported in SciPy 1.20")
def signm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.floating16, ShapeT]) -> onp.ArrayND[np.float32, ShapeT]: ...
@overload  # Nd c160
@deprecated("clongdouble input will no longer be supported in SciPy 1.20")
def signm[ShapeT: _AtLeast2D_ish](A: onp.ArrayND[npc.complexfloating160, ShapeT]) -> onp.ArrayND[np.complex128, ShapeT]: ...
@overload  # 2d +float
def signm(A: Sequence[Sequence[float]]) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~complex
def signm(A: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float
def signm(A: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64]: ...
@overload  # 3d ~complex
def signm(A: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +c  (fallback)
def signm(A: onp.ToComplexND) -> onp.ArrayND[Any, _AnyShapeOrTriviallyMaybeAlso2D]: ...

#
@overload  # ?d T, Nd T
def khatri_rao[ScalarT: (npc.signedinteger, npc.unsignedinteger, np.float32, np.float64, np.complex64, np.complex128)](
    a: onp.ArrayND[ScalarT, _JustAnyShape], b: onp.ArrayND[ScalarT]
) -> onp.ArrayND[ScalarT]: ...
@overload  # Nd T, ?d T
def khatri_rao[ScalarT: (npc.signedinteger, npc.unsignedinteger, np.float32, np.float64, np.complex64, np.complex128)](
    a: onp.ArrayND[ScalarT, tuple[int, ...]], b: onp.ArrayND[ScalarT, _JustAnyShape]
) -> onp.ArrayND[ScalarT]: ...
@overload  # 2d T, 2d T
def khatri_rao[ScalarT: (npc.signedinteger, npc.unsignedinteger, np.float32, np.float64, np.complex64, np.complex128)](
    a: onp.Array2D[ScalarT], b: onp.Array2D[ScalarT]
) -> onp.Array2D[ScalarT]: ...
@overload  # <3d T, 3d T
def khatri_rao[ScalarT: (npc.signedinteger, npc.unsignedinteger, np.float32, np.float64, np.complex64, np.complex128)](
    a: onp.Array2D[ScalarT] | onp.Array3D[ScalarT], b: onp.Array3D[ScalarT]
) -> onp.Array3D[ScalarT]: ...
@overload  # 3d T, 2d T
def khatri_rao[ScalarT: (npc.signedinteger, npc.unsignedinteger, np.float32, np.float64, np.complex64, np.complex128)](
    a: onp.Array3D[ScalarT], b: onp.Array2D[ScalarT]
) -> onp.Array3D[ScalarT]: ...
@overload  # Nd bool | f16 | f80 | c160, Nd +complex
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 1.20")
def khatri_rao(a: onp.ArrayND[np.bool | npc.floating16 | npc.inexact80], b: onp.ToComplexND) -> onp.ArrayND[Any]: ...
@overload  # Nd +complex, Nd bool | f16 | f80 | c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 1.20")
def khatri_rao(a: onp.ToComplexND, b: onp.ArrayND[np.bool | npc.floating16 | npc.inexact80]) -> onp.ArrayND[Any]: ...
@overload  # 2d +float, 2d +float
def khatri_rao(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~complex, 2d ~complex
def khatri_rao(a: Sequence[list[complex]], b: Sequence[list[complex]]) -> onp.Array2D[np.complex128]: ...
@overload  # 3d +float, 3d +float
def khatri_rao(a: Sequence[Sequence[Sequence[float]]], b: Sequence[Sequence[Sequence[float]]]) -> onp.Array3D[np.float64]: ...
@overload  # 3d ~complex, 3d ~complex
def khatri_rao(a: Sequence[Sequence[list[complex]]], b: Sequence[Sequence[list[complex]]]) -> onp.Array3D[np.complex128]: ...
@overload  # Nd +int, +int  (signed x unsigned can promote to f64)
def khatri_rao(a: onp.ToIntND, b: onp.ToIntND) -> onp.ArrayND[npc.integer | np.float64]: ...
@overload  # Nd +f64, ~f64
def khatri_rao(a: onp.ToFloat64_ND, b: onp.ToJustFloat64_ND) -> _Float64ND: ...
@overload  # Nd ~f64, +f64
def khatri_rao(a: onp.ToJustFloat64_ND, b: onp.ToFloat64_ND) -> _Float64ND: ...
@overload  # Nd +f32, ~f32
def khatri_rao(a: onp.ToFloat32_ND, b: onp.ToJustFloat32_ND) -> onp.ArrayND[np.float32]: ...
@overload  # Nd ~f32, +f32
def khatri_rao(a: onp.ToJustFloat32_ND, b: onp.ToFloat32_ND) -> onp.ArrayND[np.float32]: ...
@overload  # Nd +float, ~float
def khatri_rao(a: onp.ToFloatND, b: onp.ToJustFloatND) -> _FloatND: ...
@overload  # Nd ~float, +float
def khatri_rao(a: onp.ToJustFloatND, b: onp.ToFloatND) -> _FloatND: ...
@overload  # Nd +c64, ~c64
def khatri_rao(a: onp.ToComplex64_ND, b: onp.ToJustComplex64_ND) -> onp.ArrayND[np.complex64]: ...
@overload  # Nd ~c64, +c64
def khatri_rao(a: onp.ToJustComplex64_ND, b: onp.ToComplex64_ND) -> onp.ArrayND[np.complex64]: ...
@overload  # Nd +c128, ~c128
def khatri_rao(a: onp.ToComplex128_ND, b: onp.ToJustComplex128_ND) -> _Complex128ND: ...
@overload  # Nd ~c128, +c128
def khatri_rao(a: onp.ToJustComplex128_ND, b: onp.ToComplex128_ND) -> _Complex128ND: ...
@overload  # Nd +complex, ~complex
def khatri_rao(a: onp.ToComplexND, b: onp.ToJustComplexND) -> _ComplexND: ...
@overload  # Nd ~complex, +complex
def khatri_rao(a: onp.ToJustComplexND, b: onp.ToComplexND) -> _ComplexND: ...
@overload  # Nd +complex, +complex  (fallback)
def khatri_rao(a: onp.ToComplexND, b: onp.ToComplexND) -> onp.ArrayND[Any]: ...
