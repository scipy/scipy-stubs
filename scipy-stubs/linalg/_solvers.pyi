from _typeshed import Incomplete
from typing import Any, Final, Literal, Never, overload
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = [
    "solve_continuous_are",
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_discrete_lyapunov",
    "solve_lyapunov",
    "solve_sylvester",
]

###

type _Inexact64_2D = onp.Array2D[np.float64 | np.complex128]

type _DiscreteMethod = Literal["direct", "bilinear"]

type _ToDeprecatedND = onp.ToArrayND[Never, np.bool | np.float16 | npc.inexact80]

type _ToF64ND = onp.ToArrayND[float, npc.floating64 | npc.integer]
type _ToInexact32ND = onp.ToJustFloat32_ND | onp.ToJustComplex64_ND

type _AsF32ND = onp.ToArrayND[Never, npc.floating32 | npc.integer16 | npc.integer8]
type _AsC64ND = onp.ToArrayND[Never, npc.inexact32 | npc.integer16 | npc.integer8]
type _AsF64ND = onp.ToArrayND[float, npc.floating64 | npc.integer32 | npc.integer64]

###

# NOTE: mypy incorrectly sees disjoint dtypes like `npc.floating32` and `npc.floating64` as overlapping
# mypy: disable-error-code=overload-overlap

@overload  # ~bool | ~f16 | ~f80 | ~c160, +complex, +complex
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_sylvester(a: _ToDeprecatedND, b: onp.ToComplexND, q: onp.ToComplexND) -> onp.ArrayND[Any]: ...
@overload  # +complex, ~bool | ~f16 | ~f80 | ~c160, +complex
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_sylvester(a: onp.ToComplexND, b: _ToDeprecatedND, q: onp.ToComplexND) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, ~bool | ~f16 | ~f80 | ~c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_sylvester(a: onp.ToComplexND, b: onp.ToComplexND, q: _ToDeprecatedND) -> onp.ArrayND[Any]: ...
@overload  # ~f32, ~f32, +f32
def solve_sylvester(a: onp.ToJustFloat32_ND, b: onp.ToJustFloat32_ND, q: _AsF32ND) -> onp.ArrayND[np.float32]: ...
@overload  # ~c64, ~f32 | ~c64, +c64
def solve_sylvester(a: onp.ToJustComplex64_ND, b: _ToInexact32ND, q: _AsC64ND) -> onp.ArrayND[np.complex64]: ...
@overload  # ~f32, ~c64, +c64
def solve_sylvester(a: onp.ToJustFloat32_ND, b: onp.ToJustComplex64_ND, q: _AsC64ND) -> onp.ArrayND[np.complex64]: ...
@overload  # ~f32, ~f32, ~c64
def solve_sylvester(a: onp.ToJustFloat32_ND, b: onp.ToJustFloat32_ND, q: onp.ToJustComplex64_ND) -> onp.ArrayND[np.complex64]: ...
@overload  # +f64, +float, +float
def solve_sylvester(a: _ToF64ND, b: onp.ToFloatND, q: onp.ToFloatND) -> onp.ArrayND[np.float64]: ...
@overload  # ~f32, +f64, +float
def solve_sylvester(a: onp.ToJustFloat32_ND, b: _ToF64ND, q: onp.ToFloatND) -> onp.ArrayND[np.float64]: ...
@overload  # ~f32, ~f32, +f64
def solve_sylvester(a: onp.ToJustFloat32_ND, b: onp.ToJustFloat32_ND, q: _AsF64ND) -> onp.ArrayND[np.float64]: ...
@overload  # catch-all
def solve_sylvester(a: onp.ToFloatND, b: onp.ToFloatND, q: onp.ToFloatND) -> onp.ArrayND[np.float64 | Any]: ...
@overload  # ~c128, +complex, +complex
def solve_sylvester(a: onp.ToJustComplex128_ND, b: onp.ToComplexND, q: onp.ToComplexND) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, ~c128, +complex
def solve_sylvester(a: onp.ToComplexND, b: onp.ToJustComplex128_ND, q: onp.ToComplexND) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, ~c128
def solve_sylvester(a: onp.ToComplexND, b: onp.ToComplexND, q: onp.ToJustComplex128_ND) -> onp.ArrayND[np.complex128]: ...
@overload  # ~c64, +f64, +complex
def solve_sylvester(a: onp.ToJustComplex64_ND, b: _ToF64ND, q: onp.ToComplexND) -> onp.ArrayND[np.complex128]: ...
@overload  # ~c64, +complex, +f64
def solve_sylvester(a: onp.ToJustComplex64_ND, b: onp.ToComplexND, q: _AsF64ND) -> onp.ArrayND[np.complex128]: ...
@overload  # +f64, ~c64, +complex
def solve_sylvester(a: _ToF64ND, b: onp.ToJustComplex64_ND, q: onp.ToComplexND) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, ~c64, +f64
def solve_sylvester(a: onp.ToComplexND, b: onp.ToJustComplex64_ND, q: _AsF64ND) -> onp.ArrayND[np.complex128]: ...
@overload  # +f64, +complex, ~c64
def solve_sylvester(a: _ToF64ND, b: onp.ToComplexND, q: onp.ToJustComplex64_ND) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +f64, ~c64
def solve_sylvester(a: onp.ToComplexND, b: _ToF64ND, q: onp.ToJustComplex64_ND) -> onp.ArrayND[np.complex128]: ...
@overload  # catch-all
def solve_sylvester(a: onp.ToComplexND, b: onp.ToComplexND, q: onp.ToComplexND) -> onp.ArrayND[np.complex128 | Any]: ...

#
@overload  # ~bool | ~f16 | ~f80 | ~c160, +complex
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_continuous_lyapunov(a: _ToDeprecatedND, q: onp.ToComplexND) -> onp.ArrayND[Any]: ...
@overload  # +complex, ~bool | ~f16 | ~f80 | ~c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_continuous_lyapunov(a: onp.ToComplexND, q: _ToDeprecatedND) -> onp.ArrayND[Any]: ...
@overload  # ~f32, +f32
def solve_continuous_lyapunov(a: onp.ToJustFloat32_ND, q: _AsF32ND) -> onp.ArrayND[np.float32]: ...
@overload  # ~c64, +c64
def solve_continuous_lyapunov(a: onp.ToJustComplex64_ND, q: _AsC64ND) -> onp.ArrayND[np.complex64]: ...
@overload  # ~f32, ~c64
def solve_continuous_lyapunov(a: onp.ToJustFloat32_ND, q: onp.ToJustComplex64_ND) -> onp.ArrayND[np.complex64]: ...
@overload  # +f64, +float
def solve_continuous_lyapunov(a: _ToF64ND, q: onp.ToFloatND) -> onp.ArrayND[np.float64]: ...
@overload  # ~f32, +f64
def solve_continuous_lyapunov(a: onp.ToJustFloat32_ND, q: _AsF64ND) -> onp.ArrayND[np.float64]: ...
@overload  # catch-all
def solve_continuous_lyapunov(a: onp.ToFloatND, q: onp.ToFloatND) -> onp.ArrayND[np.float64 | Any]: ...
@overload  # ~c128, +complex
def solve_continuous_lyapunov(a: onp.ToJustComplex128_ND, q: onp.ToComplexND) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, ~c128
def solve_continuous_lyapunov(a: onp.ToComplexND, q: onp.ToJustComplex128_ND) -> onp.ArrayND[np.complex128]: ...
@overload  # ~c64, +f64
def solve_continuous_lyapunov(a: onp.ToJustComplex64_ND, q: _AsF64ND) -> onp.ArrayND[np.complex128]: ...
@overload  # +f64, ~c64
def solve_continuous_lyapunov(a: _ToF64ND, q: onp.ToJustComplex64_ND) -> onp.ArrayND[np.complex128]: ...
@overload  # catch-all
def solve_continuous_lyapunov(a: onp.ToComplexND, q: onp.ToComplexND) -> onp.ArrayND[np.complex128 | Any]: ...

#
solve_lyapunov: Final = solve_continuous_lyapunov

#
# NOTE: Both solvers construct a `float64` identity matrix, so single precision input is never preserved.
def _solve_discrete_lyapunov_direct(a: onp.Array2D[npc.number], q: onp.Array2D[npc.number]) -> _Inexact64_2D: ...
def _solve_discrete_lyapunov_bilinear(a: onp.Array2D[npc.number], q: onp.Array2D[npc.number]) -> _Inexact64_2D: ...

#
@overload  # ~bool | ~f16 | ~f80 | ~c160, +complex
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_discrete_lyapunov(
    a: _ToDeprecatedND, q: onp.ToComplexND, method: _DiscreteMethod | None = None
) -> onp.ArrayND[Any]: ...
@overload  # +complex, ~bool | ~f16 | ~f80 | ~c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_discrete_lyapunov(
    a: onp.ToComplexND, q: _ToDeprecatedND, method: _DiscreteMethod | None = None
) -> onp.ArrayND[Any]: ...
@overload  # +float, +float
def solve_discrete_lyapunov(
    a: onp.ToFloatND, q: onp.ToFloatND, method: _DiscreteMethod | None = None
) -> onp.ArrayND[np.float64]: ...
@overload  # ~complex, +complex
def solve_discrete_lyapunov(
    a: onp.ToJustComplexND, q: onp.ToComplexND, method: _DiscreteMethod | None = None
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, ~complex
def solve_discrete_lyapunov(
    a: onp.ToComplexND, q: onp.ToJustComplexND, method: _DiscreteMethod | None = None
) -> onp.ArrayND[np.complex128]: ...
@overload  # catch-all
def solve_discrete_lyapunov(
    a: onp.ToComplexND, q: onp.ToComplexND, method: _DiscreteMethod | None = None
) -> onp.ArrayND[np.complex128 | Any]: ...

#
@overload  # ~bool | ~f16 | ~f80 | ~c160, +complex, +complex, +complex, +complex?, +complex?
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_continuous_are(
    a: _ToDeprecatedND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, ~bool | ~f16 | ~f80 | ~c160, +complex, +complex, +complex?, +complex?
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_continuous_are(
    a: onp.ToComplexND,
    b: _ToDeprecatedND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, ~bool | ~f16 | ~f80 | ~c160, +complex, +complex?, +complex?
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: _ToDeprecatedND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, +complex, ~bool | ~f16 | ~f80 | ~c160, +complex?, +complex?
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: _ToDeprecatedND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, +complex, +complex, ~bool | ~f16 | ~f80 | ~c160, +complex?
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: _ToDeprecatedND,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, +complex, +complex, +complex, ~bool | ~f16 | ~f80 | ~c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None,
    s: _ToDeprecatedND,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, +complex, +complex, +complex?, *, ~bool | ~f16 | ~f80 | ~c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    *,
    s: _ToDeprecatedND,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # real
def solve_continuous_are(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    q: onp.ToFloatND,
    r: onp.ToFloatND,
    e: onp.ToFloatND | None = None,
    s: onp.ToFloatND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # ~complex, +complex, +complex, +complex, +complex?, +complex?
def solve_continuous_are(
    a: onp.ToJustComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, ~complex, +complex, +complex, +complex?, +complex?
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToJustComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, ~complex, +complex, +complex?, +complex?
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToJustComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, +complex, ~complex, +complex?, +complex?
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToJustComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, +complex, +complex, ~complex, +complex?
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToJustComplexND,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, +complex, +complex, +complex, ~complex
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None,
    s: onp.ToJustComplexND,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, +complex, +complex, +complex?, *, ~complex
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    *,
    s: onp.ToJustComplexND,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # catch-all
def solve_continuous_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128 | Any]: ...

#
@overload  # ~bool | ~f16 | ~f80 | ~c160, +complex, +complex, +complex, +complex?, +complex?
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_discrete_are(
    a: _ToDeprecatedND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, ~bool | ~f16 | ~f80 | ~c160, +complex, +complex, +complex?, +complex?
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_discrete_are(
    a: onp.ToComplexND,
    b: _ToDeprecatedND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, ~bool | ~f16 | ~f80 | ~c160, +complex, +complex?, +complex?
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: _ToDeprecatedND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, +complex, ~bool | ~f16 | ~f80 | ~c160, +complex?, +complex?
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: _ToDeprecatedND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, +complex, +complex, ~bool | ~f16 | ~f80 | ~c160, +complex?
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: _ToDeprecatedND,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, +complex, +complex, +complex, ~bool | ~f16 | ~f80 | ~c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None,
    s: _ToDeprecatedND,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # +complex, +complex, +complex, +complex, +complex?, *, ~bool | ~f16 | ~f80 | ~c160
@deprecated("bool, float16, longdouble, and clongdouble input will no longer be supported in SciPy 2.1")
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    *,
    s: _ToDeprecatedND,
    balanced: bool = True,
) -> onp.ArrayND[Any]: ...
@overload  # real
def solve_discrete_are(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    q: onp.ToFloatND,
    r: onp.ToFloatND,
    e: onp.ToFloatND | None = None,
    s: onp.ToFloatND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # ~complex, +complex, +complex, +complex, +complex?, +complex?
def solve_discrete_are(
    a: onp.ToJustComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, ~complex, +complex, +complex, +complex?, +complex?
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToJustComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, ~complex, +complex, +complex?, +complex?
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToJustComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, +complex, ~complex, +complex?, +complex?
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToJustComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, +complex, +complex, ~complex, +complex?
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToJustComplexND,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, +complex, +complex, +complex, ~complex
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None,
    s: onp.ToJustComplexND,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +complex, +complex, +complex, +complex?, *, ~complex
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    *,
    s: onp.ToJustComplexND,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # catch-all
def solve_discrete_are(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None = None,
    s: onp.ToComplexND | None = None,
    balanced: bool = True,
) -> onp.ArrayND[np.complex128 | Any]: ...

#
def _are_validate_args(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    q: onp.ToComplexND,
    r: onp.ToComplexND,
    e: onp.ToComplexND | None,
    s: onp.ToComplexND | None,
    eq_type: Literal["care", "dare"] = "care",
) -> tuple[
    onp.ArrayND[Incomplete],  # a
    onp.ArrayND[Incomplete],  # b
    onp.ArrayND[Incomplete],  # q
    onp.ArrayND[Incomplete],  # r
    onp.ArrayND[Incomplete],  # e
    onp.ArrayND[Incomplete],  # s
    int,  # m
    int,  # n
    type[float | complex],  # r_or_c
    bool,  # gen_or_not
]: ...
