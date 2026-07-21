from typing import overload

import numpy as np
import optype.numpy as onp

__all__ = ["ldl"]

type _ISize1D = onp.Array1D[np.intp]
type _ISizeND = onp.ArrayND[np.intp]

type _Float32_2D = onp.Array2D[np.float32]
type _Float32ND = onp.ArrayND[np.float32]
type _Float64_2D = onp.Array2D[np.float64]
type _Float64ND = onp.ArrayND[np.float64]
type _Float2D = onp.Array2D[np.float32 | np.float64]
type _FloatND = onp.ArrayND[np.float32 | np.float64]

type _Complex64_2D = onp.Array2D[np.complex64]
type _Complex64ND = onp.ArrayND[np.complex64]
type _Complex128_2D = onp.Array2D[np.complex128]
type _Complex128ND = onp.ArrayND[np.complex128]
type _Complex2D = onp.Array2D[np.complex64 | np.complex128]
type _ComplexND = onp.ArrayND[np.complex64 | np.complex128]

type _InexactND = onp.ArrayND[np.float32 | np.float64 | np.complex64 | np.complex128]

###

@overload  # 2d: float32 -> float32
def ldl(  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    A: onp.ToArrayStrict2D[np.float32, np.float32],
    lower: bool = True,
    hermitian: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[_Float32_2D, _Float32_2D, _ISize1D]: ...
@overload  # 2d: -> float64
def ldl(
    A: onp.ToFloat64Strict2D, lower: bool = True, hermitian: bool = True, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[_Float64_2D, _Float64_2D, _ISize1D]: ...
@overload  # 2d: real -> float32 | float64
def ldl(
    A: onp.ToFloatStrict2D, lower: bool = True, hermitian: bool = True, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[_Float2D, _Float2D, _ISize1D]: ...
@overload  # nd: float32 -> float32
def ldl(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    A: onp.ToArrayND[np.float32, np.float32],
    lower: bool = True,
    hermitian: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[_Float32ND, _Float32ND, _ISizeND]: ...
@overload  # nd: -> float64
def ldl(
    A: onp.ToFloat64_ND, lower: bool = True, hermitian: bool = True, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[_Float64ND, _Float64ND, _ISizeND]: ...
@overload  # nd: real -> float32 | float64
def ldl(
    A: onp.ToFloatND, lower: bool = True, hermitian: bool = True, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[_FloatND, _FloatND, _ISizeND]: ...
@overload  # 2d: complex64 -> complex64
def ldl(  # type: ignore[overload-overlap]
    A: onp.ToArrayStrict2D[np.complex64, np.complex64],
    lower: bool = True,
    hermitian: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[_Complex64_2D, _Complex64_2D, _ISize1D]: ...
@overload  # 2d: -> complex128
def ldl(
    A: onp.ToJustComplex128Strict2D,
    lower: bool = True,
    hermitian: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[_Complex128_2D, _Complex128_2D, _ISize1D]: ...
@overload  # 2d:complex -> complex64 | complex128
def ldl(
    A: onp.ToJustComplexStrict2D, lower: bool = True, hermitian: bool = True, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[_Complex2D, _Complex2D, _ISize1D]: ...
@overload  # nd: complex64 -> complex64
def ldl(  # type: ignore[overload-overlap]
    A: onp.ToArrayND[np.complex64, np.complex64],
    lower: bool = True,
    hermitian: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[_Complex64ND, _Complex64ND, _ISizeND]: ...
@overload  # nd: -> complex128
def ldl(
    A: onp.ToJustComplex128_ND, lower: bool = True, hermitian: bool = True, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[_Complex128ND, _Complex128ND, _ISizeND]: ...
@overload  # nd: complex -> complex64 | complex128
def ldl(
    A: onp.ToJustComplexND, lower: bool = True, hermitian: bool = True, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[_ComplexND, _ComplexND, _ISizeND]: ...
@overload  # nd: -> f32 | f64 | c64 | c128
def ldl(
    A: onp.ToComplexND, lower: bool = True, hermitian: bool = True, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[_InexactND, _InexactND, _ISizeND]: ...
