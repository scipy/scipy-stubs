from collections.abc import Callable
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
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

_ToPosInt: TypeAlias = np.unsignedinteger[Any] | Literal[0, 1, 2, 4, 5, 6, 7, 8]

_IntND: TypeAlias = onp.ArrayND[np.integer[Any]]
_Float64ND: TypeAlias = onp.ArrayND[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.floating[Any]]
_Complex128ND: TypeAlias = onp.ArrayND[np.complex128]
_ComplexND: TypeAlias = onp.ArrayND[np.complexfloating[Any, Any]]
_RealND: TypeAlias = onp.ArrayND[np.floating[Any] | np.integer[Any]]
_InexactND: TypeAlias = onp.ArrayND[np.inexact[Any]]

_FloatFunc: TypeAlias = Callable[[onp.Array1D[np.float64]], onp.ToFloat1D]
_ComplexFunc: TypeAlias = Callable[[onp.Array1D[np.complex128]], onp.ToComplex1D]

###

@overload  # int, positive int
def fractional_matrix_power(A: onp.ToIntND, t: _ToPosInt) -> _IntND: ...
@overload  # real, int
def fractional_matrix_power(A: onp.ToFloatND, t: onp.ToInt) -> _RealND: ...
@overload  # complex, int
def fractional_matrix_power(A: onp.ToComplexND, t: onp.ToInt) -> onp.ArrayND[Any]: ...
@overload  # complex, float
def fractional_matrix_power(A: onp.ToComplexND, t: onp.ToJustFloat) -> _ComplexND: ...

#
@overload
def sqrtm(A: onp.ToIntND | onp.ToJustFloat64_ND) -> _Float64ND: ...
@overload
def sqrtm(A: onp.ToJustFloatND) -> _FloatND: ...
@overload
def sqrtm(A: onp.ToJustComplex128_ND) -> _Complex128ND: ...
@overload
def sqrtm(A: onp.ToJustComplexND) -> _ComplexND: ...
@overload
def sqrtm(A: onp.ToComplexND) -> _InexactND: ...
@overload
@deprecated("The `disp` argument is deprecated and will be removed in SciPy 1.18.0.")
def sqrtm(A: onp.ToComplexND, disp: Truthy) -> _InexactND: ...
@overload
@deprecated("The `disp` argument is deprecated and will be removed in SciPy 1.18.0.")
def sqrtm(A: onp.ToComplexND, disp: Falsy) -> tuple[_InexactND, np.float64]: ...
@overload
@deprecated("The `blocksize` argument is deprecated and will be removed in SciPy 1.18.0.")
def sqrtm(A: onp.ToComplexND, *, blocksize: int) -> _InexactND: ...
@overload
@deprecated("The `blocksize` argument is deprecated and will be removed in SciPy 1.18.0.")
@deprecated("The `disp` argument is deprecated and will be removed in SciPy 1.18.0.")
def sqrtm(A: onp.ToComplexND, disp: Truthy, blocksize: int) -> _InexactND: ...
@overload
@deprecated("The `blocksize` argument is deprecated and will be removed in SciPy 1.18.0.")
@deprecated("The `disp` argument is deprecated and will be removed in SciPy 1.18.0.")
def sqrtm(A: onp.ToComplexND, disp: Falsy, blocksize: int) -> tuple[_InexactND, np.float64]: ...

# NOTE: return dtype depends on the sign of the values
@overload  # complex 2d+ array-like
def logm(A: onp.ToComplexND) -> _InexactND: ...
@overload  # complex 2d+ array-like, disp=True  (deprecated)
@deprecated("The `disp` argument is deprecated and will be removed in SciPy 1.18.0.")
def logm(A: onp.ToComplexND, disp: Truthy) -> _InexactND: ...
@overload  # complex 2d+ array-like, disp=False  (deprecated)
@deprecated(
    "The `disp` argument is deprecated and will be removed in SciPy 1.18.0. "
    "The previously returned error estimate can be computed as `norm(expm(logm(A)) - A, 1) / norm(A, 1)`."
)
def logm(A: onp.ToComplexND, disp: Falsy) -> tuple[_InexactND, float]: ...

#
@overload  # real
def expm(A: onp.ToFloatND) -> _FloatND: ...
@overload  # complex
def expm(A: onp.ToComplexND) -> _InexactND: ...

#
@overload  # real
def cosm(A: onp.ToFloatND) -> _FloatND: ...
@overload  # complex
def cosm(A: onp.ToComplexND) -> _InexactND: ...

#
@overload  # real
def sinm(A: onp.ToFloatND) -> _FloatND: ...
@overload  # complex
def sinm(A: onp.ToComplexND) -> _InexactND: ...

#
@overload  # real
def tanm(A: onp.ToFloatND) -> _FloatND: ...
@overload  # complex
def tanm(A: onp.ToComplexND) -> _InexactND: ...

#
@overload  # real
def coshm(A: onp.ToFloatND) -> _FloatND: ...
@overload  # complex
def coshm(A: onp.ToComplexND) -> _InexactND: ...

#
@overload  # real
def sinhm(A: onp.ToFloatND) -> _FloatND: ...
@overload  # complex
def sinhm(A: onp.ToComplexND) -> _InexactND: ...

#
@overload  # real
def tanhm(A: onp.ToFloatND) -> _FloatND: ...
@overload  # complex
def tanhm(A: onp.ToComplexND) -> _InexactND: ...

#
@overload  # real, disp: True = ...
def funm(A: onp.ToFloatND, func: _FloatFunc, disp: Truthy = True) -> _FloatND: ...
@overload  # real, disp: False
def funm(A: onp.ToFloatND, func: _FloatFunc, disp: Falsy) -> _ComplexND: ...
@overload  # complex, disp: True = ...
def funm(A: onp.ToComplexND, func: _ComplexFunc, disp: Truthy = True) -> _ComplexND: ...
@overload  # complex, disp: False
def funm(A: onp.ToComplexND, func: _ComplexFunc, disp: Falsy) -> tuple[_ComplexND, np.float64]: ...

#
@overload  # real 2d+ array-like
def signm(A: onp.ToFloatND) -> _FloatND: ...
@overload  # real 2d+ array-like, disp: True  (deprecated)
@deprecated("The `disp` argument is deprecated and will be removed in SciPy 1.18.0.")
def signm(A: onp.ToFloatND, disp: Truthy) -> _FloatND: ...
@overload  # real 2d+ array-like, disp: False  (deprecated)
@deprecated(
    "The `disp` argument is deprecated and will be removed in SciPy 1.18.0. "
    "The previously returned error estimate can be computed as `norm(signm @ signm - signm, 1)`."
)
def signm(A: onp.ToFloatND, disp: Falsy) -> tuple[_FloatND, np.float64]: ...
@overload  # complex 2d+ array-like
def signm(A: onp.ToComplexND) -> _InexactND: ...
@overload  # complex 2d+ array-like, disp: True  (deprecated)
@deprecated("The `disp` argument is deprecated and will be removed in SciPy 1.18.0.")
def signm(A: onp.ToComplexND, disp: Truthy) -> _InexactND: ...
@overload  # complex 2d+ array-like, disp: False  (deprecated)
@deprecated(
    "The `disp` argument is deprecated and will be removed in SciPy 1.18.0. "
    "The previously returned error estimate can be computed as `norm(signm @ signm - signm, 1)`."
)
def signm(A: onp.ToComplexND, disp: Falsy) -> tuple[_InexactND, np.float64]: ...

#
@overload  # +integer, +integer
def khatri_rao(a: onp.ToIntND, b: onp.ToIntND) -> _IntND: ...
@overload  # +float64, ~float64
def khatri_rao(a: onp.ToFloat64_ND, b: onp.ToJustFloat64_ND) -> _Float64ND: ...
@overload  # ~float64, +float64
def khatri_rao(a: onp.ToJustFloat64_ND, b: onp.ToFloat64_ND) -> _Float64ND: ...
@overload  # +floating, ~floating
def khatri_rao(a: onp.ToFloatND, b: onp.ToJustFloatND) -> _FloatND: ...
@overload  # ~floating, +floating
def khatri_rao(a: onp.ToJustFloatND, b: onp.ToFloatND) -> _FloatND: ...
@overload  # +complex128, ~complex128
def khatri_rao(a: onp.ToComplex128_ND, b: onp.ToJustComplex128_ND) -> _Complex128ND: ...
@overload  # ~complex128, +complex128
def khatri_rao(a: onp.ToJustComplex128_ND, b: onp.ToComplex128_ND) -> _Complex128ND: ...
@overload  # +complexfloating, ~complexfloating
def khatri_rao(a: onp.ToComplexND, b: onp.ToJustComplexND) -> _ComplexND: ...
@overload  # ~complexfloating, +complexfloating
def khatri_rao(a: onp.ToJustComplexND, b: onp.ToComplexND) -> _ComplexND: ...
@overload  # fallback
def khatri_rao(a: onp.ToComplexND, b: onp.ToComplexND) -> onp.ArrayND[Any]: ...
