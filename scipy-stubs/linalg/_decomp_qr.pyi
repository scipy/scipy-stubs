from typing import Any, Literal, Never, overload
from typing_extensions import deprecated

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["qr", "qr_multiply", "rq"]

type _Tuple2[T] = tuple[T, T]

type _AsF64Strict2D = onp.ToArrayStrict2D[float, npc.floating64 | npc.integer64 | npc.integer32]

type _AsF16ND = onp.ToArrayND[Never, np.float16 | np.bool]
type _AsF32ND = onp.ToArrayND[Never, npc.floating32 | npc.integer16 | npc.integer8]
type _AsF64ND = onp.ToArrayND[float, npc.floating64 | npc.integer64 | npc.integer32]
type _AsC80ND = onp.ToArrayND[Never, npc.inexact80]

type _Side = Literal["left", "right"]
type _ModeFullEcon = Literal["full", "economic"]
type _ModeR = Literal["r"]
type _ModeRaw = Literal["raw"]

type _NoValueType = op.JustObject

###

# NOTE: mypy incorrectly sees disjoint dtypes like `np.complex64` and `np.complex128` as overlapping
# mypy: disable-error-code=overload-overlap

@overload  # +f64, mode: {full, economic}
def qr(
    a: _AsF64ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float64]]: ...
@overload  # +f64, mode: {full, economic}, pivoting: True
def qr(
    a: _AsF64ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    *,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.int32]]: ...
@overload  # +f64, mode: r
def qr(
    a: _AsF64ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float64]]: ...
@overload  # +f64, mode: r, pivoting: True
def qr(
    a: _AsF64ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]]: ...
@overload  # +f64, mode: raw
def qr(
    a: _AsF64ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.float64]], onp.ArrayND[np.float64]]: ...
@overload  # +f64, mode: raw, pivoting: True
def qr(
    a: _AsF64ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.float64]], onp.ArrayND[np.float64], onp.ArrayND[np.int32]]: ...
@overload  # +f32, mode: {full, economic}
def qr(
    a: _AsF32ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float32]]: ...
@overload  # +f32, mode: {full, economic}, pivoting: True
def qr(
    a: _AsF32ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    *,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], onp.ArrayND[np.int32]]: ...
@overload  # +f32, mode: r
def qr(
    a: _AsF32ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float32]]: ...
@overload  # +f32, mode: r, pivoting: True
def qr(
    a: _AsF32ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float32], onp.ArrayND[np.int32]]: ...
@overload  # +f32, mode: raw
def qr(
    a: _AsF32ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.float32]], onp.ArrayND[np.float32]]: ...
@overload  # +f32, mode: raw, pivoting: True
def qr(
    a: _AsF32ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.float32]], onp.ArrayND[np.float32], onp.ArrayND[np.int32]]: ...
@overload  # ~c64, mode: {full, economic}
def qr(
    a: onp.ToJustComplex64_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex64]]: ...
@overload  # ~c64, mode: {full, economic}, pivoting: True
def qr(
    a: onp.ToJustComplex64_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    *,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64], onp.ArrayND[np.int32]]: ...
@overload  # ~c64, mode: r
def qr(
    a: onp.ToJustComplex64_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.complex64]]: ...
@overload  # ~c64, mode: r, pivoting: True
def qr(
    a: onp.ToJustComplex64_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.int32]]: ...
@overload  # ~c64, mode: raw
def qr(
    a: onp.ToJustComplex64_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.complex64]], onp.ArrayND[np.complex64]]: ...
@overload  # ~c64, mode: raw, pivoting: True
def qr(
    a: onp.ToJustComplex64_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.complex64]], onp.ArrayND[np.complex64], onp.ArrayND[np.int32]]: ...
@overload  # ~c128, mode: {full, economic}
def qr(
    a: onp.ToJustComplex128_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex128]]: ...
@overload  # ~c128, mode: {full, economic}, pivoting: True
def qr(
    a: onp.ToJustComplex128_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    *,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128], onp.ArrayND[np.int32]]: ...
@overload  # ~c128, mode: r
def qr(
    a: onp.ToJustComplex128_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.complex128]]: ...
@overload  # ~c128, mode: r, pivoting: True
def qr(
    a: onp.ToJustComplex128_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.int32]]: ...
@overload  # ~c128, mode: raw
def qr(
    a: onp.ToJustComplex128_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.complex128]], onp.ArrayND[np.complex128]]: ...
@overload  # ~c128, mode: raw, pivoting: True
def qr(
    a: onp.ToJustComplex128_ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.complex128]], onp.ArrayND[np.complex128], onp.ArrayND[np.int32]]: ...
@overload  # ~bool | ~f16 | ~f80 | ~c160, mode: {full, economic}
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def qr(
    a: _AsF16ND | _AsC80ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float64 | Any]]: ...
@overload  # ~bool | ~f16 | ~f80 | ~c160, mode: {full, economic}, pivoting: True
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def qr(
    a: _AsF16ND | _AsC80ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    *,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any], onp.ArrayND[np.int32]]: ...
@overload  # ~bool | ~f16 | ~f80 | ~c160, mode: r
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def qr(
    a: _AsF16ND | _AsC80ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float64 | Any]]: ...
@overload  # ~bool | ~f16 | ~f80 | ~c160, mode: r, pivoting: True
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def qr(
    a: _AsF16ND | _AsC80ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.int32]]: ...
@overload  # ~bool | ~f16 | ~f80 | ~c160, mode: raw
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def qr(
    a: _AsF16ND | _AsC80ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.float64 | Any]], onp.ArrayND[np.float64 | Any]]: ...
@overload  # ~bool | ~f16 | ~f80 | ~c160, mode: raw, pivoting: True
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def qr(
    a: _AsF16ND | _AsC80ND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.float64 | Any]], onp.ArrayND[np.float64 | Any], onp.ArrayND[np.int32]]: ...
@overload  # catch-all, mode: {full, economic}
def qr(
    a: onp.ToComplexND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float64 | Any]]: ...
@overload  # catch-all, mode: {full, economic}, pivoting: True
def qr(
    a: onp.ToComplexND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    mode: _ModeFullEcon = "full",
    *,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any], onp.ArrayND[np.int32]]: ...
@overload  # catch-all, mode: r
def qr(
    a: onp.ToComplexND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float64 | Any]]: ...
@overload  # catch-all, mode: r, pivoting: True
def qr(
    a: onp.ToComplexND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeR,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.int32]]: ...
@overload  # catch-all, mode: raw
def qr(
    a: onp.ToComplexND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToFalse = False,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.float64 | Any]], onp.ArrayND[np.float64 | Any]]: ...
@overload  # catch-all, mode: raw, pivoting: True
def qr(
    a: onp.ToComplexND,
    overwrite_a: bool = False,
    lwork: _NoValueType = ...,
    *,
    mode: _ModeRaw,
    pivoting: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_Tuple2[onp.ArrayND[np.float64 | Any]], onp.ArrayND[np.float64 | Any], onp.ArrayND[np.int32]]: ...

#
@overload  # 2d +f32, c: 1d
def qr_multiply(
    a: onp.ToFloat32Strict2D,
    c: onp.ToFloatStrict1D,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.Array1D[np.float32], onp.Array2D[np.float32]]: ...
@overload  # 2d +f64, c: 1d
def qr_multiply(
    a: _AsF64Strict2D,
    c: onp.ToFloatStrict1D,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.Array1D[np.float64], onp.Array2D[np.float64]]: ...
@overload  # 2d +f32, c: 2d
def qr_multiply(
    a: onp.ToFloat32Strict2D,
    c: onp.ToFloatStrict2D,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> _Tuple2[onp.Array2D[np.float32]]: ...
@overload  # 2d +f64, c: 2d
def qr_multiply(
    a: _AsF64Strict2D,
    c: onp.ToFloatStrict2D,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> _Tuple2[onp.Array2D[np.float64]]: ...
@overload  # Nd +f32, c: Nd
def qr_multiply(
    a: _AsF32ND,
    c: onp.ToFloatND,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> _Tuple2[onp.ArrayND[np.float32]]: ...
@overload  # Nd +f64, c: Nd
def qr_multiply(
    a: _AsF64ND,
    c: onp.ToFloatND,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> _Tuple2[onp.ArrayND[np.float64]]: ...
@overload  # 2d +f32, c: 1d | 2d, pivoting: True
def qr_multiply(
    a: onp.ToFloat32Strict2D,
    c: onp.ToFloatStrict1D | onp.ToFloatStrict2D,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.float32], onp.Array2D[np.float32], onp.Array1D[np.int32]]: ...
@overload  # 2d +f64, c: 1d | 2d, pivoting: True
def qr_multiply(
    a: _AsF64Strict2D,
    c: onp.ToFloatStrict1D | onp.ToFloatStrict2D,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.float64], onp.Array2D[np.float64], onp.Array1D[np.int32]]: ...
@overload  # Nd +f32, c: Nd, pivoting: True
def qr_multiply(
    a: _AsF32ND,
    c: onp.ToFloatND,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], onp.ArrayND[np.int32]]: ...
@overload  # Nd +f64, c: Nd, pivoting: True
def qr_multiply(
    a: _AsF64ND,
    c: onp.ToFloatND,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.int32]]: ...
@overload  # 2d ~c64, c: 1d | 2d
def qr_multiply(
    a: onp.ToJustComplex64Strict2D,
    c: onp.ToComplexStrict1D | onp.ToComplexStrict2D,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.complex64], onp.Array2D[np.complex64]]: ...
@overload  # 2d ~c128, c: 1d | 2d
def qr_multiply(
    a: onp.ToJustComplex128Strict2D,
    c: onp.ToComplexStrict1D | onp.ToComplexStrict2D,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.complex128], onp.Array2D[np.complex128]]: ...
@overload  # Nd ~c64, c: Nd
def qr_multiply(
    a: onp.ToJustComplex64_ND,
    c: onp.ToComplexND,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> _Tuple2[onp.ArrayND[np.complex64]]: ...
@overload  # Nd ~c128, c: Nd
def qr_multiply(
    a: onp.ToJustComplex128_ND,
    c: onp.ToComplexND,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> _Tuple2[onp.ArrayND[np.complex128]]: ...
@overload  # 2d ~c64, c: 1d | 2d, pivoting: True
def qr_multiply(
    a: onp.ToJustComplex64Strict2D,
    c: onp.ToComplexStrict1D | onp.ToComplexStrict2D,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.complex64], onp.Array2D[np.complex64], onp.Array1D[np.int32]]: ...
@overload  # 2d ~c128, c: 1d | 2d, pivoting: True
def qr_multiply(
    a: onp.ToJustComplex128Strict2D,
    c: onp.ToComplexStrict1D | onp.ToComplexStrict2D,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.complex128], onp.Array2D[np.complex128], onp.Array1D[np.int32]]: ...
@overload  # Nd ~c64, c: Nd, pivoting: True
def qr_multiply(
    a: onp.ToJustComplex64_ND,
    c: onp.ToComplexND,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64], onp.ArrayND[np.int32]]: ...
@overload  # Nd ~c128, c: Nd, pivoting: True
def qr_multiply(
    a: onp.ToJustComplex128_ND,
    c: onp.ToComplexND,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128], onp.ArrayND[np.int32]]: ...
@overload  # 2d +float, c: 1d
def qr_multiply(
    a: onp.ToFloatStrict2D,
    c: onp.ToFloatStrict1D,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.Array1D[np.float64 | Any], onp.Array2D[np.float64 | Any]]: ...
@overload  # 2d +float, c: 2d
def qr_multiply(
    a: onp.ToFloatStrict2D,
    c: onp.ToFloatStrict2D,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> _Tuple2[onp.Array2D[np.float64 | Any]]: ...
@overload  # 2d +float, c: 1d | 2d
def qr_multiply(
    a: onp.ToFloatStrict2D,
    c: onp.ToFloatStrict1D | onp.ToFloatStrict2D,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.Array2D[np.float64 | Any]]: ...
@overload  # Nd +float, c: Nd
def qr_multiply(
    a: onp.ToFloatND,
    c: onp.ToFloatND,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> _Tuple2[onp.ArrayND[np.float64 | Any]]: ...
@overload  # 2d +float, c: 1d | 2d, pivoting: True
def qr_multiply(
    a: onp.ToFloatStrict2D,
    c: onp.ToFloatStrict1D | onp.ToFloatStrict2D,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.Array2D[np.float64 | Any], onp.Array1D[np.int32]]: ...
@overload  # Nd +float, c: Nd, pivoting: True
def qr_multiply(
    a: onp.ToFloatND,
    c: onp.ToFloatND,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any], onp.ArrayND[np.int32]]: ...
@overload  # 2d catch-all, c: 1d | 2d
def qr_multiply(
    a: onp.ToComplexStrict2D,
    c: onp.ToComplexStrict1D | onp.ToComplexStrict2D,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.Array2D[np.float64 | Any]]: ...
@overload  # Nd catch-all, c: Nd
def qr_multiply(
    a: onp.ToComplexND,
    c: onp.ToComplexND,
    mode: _Side = "right",
    pivoting: onp.ToFalse = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> _Tuple2[onp.ArrayND[np.float64 | Any]]: ...
@overload  # 2d catch-all, c: 1d | 2d, pivoting: True
def qr_multiply(
    a: onp.ToComplexStrict2D,
    c: onp.ToComplexStrict1D | onp.ToComplexStrict2D,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.Array2D[np.float64 | Any], onp.Array1D[np.int32]]: ...
@overload  # Nd catch-all, c: Nd, pivoting: True
def qr_multiply(
    a: onp.ToComplexND,
    c: onp.ToComplexND,
    mode: _Side = "right",
    *,
    pivoting: onp.ToTrue,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any], onp.ArrayND[np.int32]]: ...

#
@overload  # 2d +f64, mode: {full, economic}
def rq(
    a: _AsF64Strict2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: bool = True,
) -> _Tuple2[onp.Array2D[np.float64]]: ...
@overload  # Nd +f64, mode: {full, economic}
def rq(
    a: _AsF64ND, overwrite_a: bool = False, lwork: int | None = None, mode: _ModeFullEcon = "full", check_finite: bool = True
) -> _Tuple2[onp.ArrayND[np.float64]]: ...
@overload  # 2d +f64, mode: r
def rq(
    a: _AsF64Strict2D, overwrite_a: bool = False, lwork: int | None = None, *, mode: _ModeR, check_finite: bool = True
) -> onp.Array2D[np.float64]: ...
@overload  # Nd +f64, mode: r
def rq(
    a: _AsF64ND, overwrite_a: bool = False, lwork: int | None = None, *, mode: _ModeR, check_finite: bool = True
) -> onp.ArrayND[np.float64]: ...
@overload  # 2d +f32, mode: {full, economic}
def rq(
    a: onp.ToArrayStrict2D[Never, npc.floating32 | npc.integer16 | npc.integer8],
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: bool = True,
) -> _Tuple2[onp.Array2D[np.float32]]: ...
@overload  # Nd +f32, mode: {full, economic}
def rq(
    a: _AsF32ND, overwrite_a: bool = False, lwork: int | None = None, mode: _ModeFullEcon = "full", check_finite: bool = True
) -> _Tuple2[onp.ArrayND[np.float32]]: ...
@overload  # 2d +f32, mode: r
def rq(
    a: onp.ToArrayStrict2D[Never, npc.floating32 | npc.integer16 | npc.integer8],
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: _ModeR,
    check_finite: bool = True,
) -> onp.Array2D[np.float32]: ...
@overload  # Nd +f32, mode: r
def rq(
    a: _AsF32ND, overwrite_a: bool = False, lwork: int | None = None, *, mode: _ModeR, check_finite: bool = True
) -> onp.ArrayND[np.float32]: ...
@overload  # 2d ~c64, mode: {full, economic}
def rq(
    a: onp.ToJustComplex64Strict2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: bool = True,
) -> _Tuple2[onp.Array2D[np.complex64]]: ...
@overload  # Nd ~c64, mode: {full, economic}
def rq(
    a: onp.ToJustComplex64_ND,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex64]]: ...
@overload  # 2d ~c64, mode: r
def rq(
    a: onp.ToJustComplex64Strict2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: _ModeR,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd ~c64, mode: r
def rq(
    a: onp.ToJustComplex64_ND, overwrite_a: bool = False, lwork: int | None = None, *, mode: _ModeR, check_finite: bool = True
) -> onp.ArrayND[np.complex64]: ...
@overload  # 2d ~c128, mode: {full, economic}
def rq(
    a: onp.ToJustComplex128Strict2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: bool = True,
) -> _Tuple2[onp.Array2D[np.complex128]]: ...
@overload  # Nd ~c128, mode: {full, economic}
def rq(
    a: onp.ToJustComplex128_ND,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex128]]: ...
@overload  # 2d ~c128, mode: r
def rq(
    a: onp.ToJustComplex128Strict2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: _ModeR,
    check_finite: bool = True,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd ~c128, mode: r
def rq(
    a: onp.ToJustComplex128_ND, overwrite_a: bool = False, lwork: int | None = None, *, mode: _ModeR, check_finite: bool = True
) -> onp.ArrayND[np.complex128]: ...
@overload  # 2d ~bool | ~f16 | ~f80 | ~c160, mode: {full, economic}
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def rq(
    a: onp.ToArrayStrict2D[Never, np.bool | np.float16 | npc.inexact80],
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: bool = True,
) -> _Tuple2[onp.Array2D[np.float64 | Any]]: ...
@overload  # Nd ~bool | ~f16 | ~f80 | ~c160, mode: {full, economic}
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def rq(
    a: _AsF16ND | _AsC80ND,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float64 | Any]]: ...
@overload  # 2d ~bool | ~f16 | ~f80 | ~c160, mode: r
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def rq(
    a: onp.ToArrayStrict2D[Never, np.bool | np.float16 | npc.inexact80],
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: _ModeR,
    check_finite: bool = True,
) -> onp.Array2D[np.float64 | Any]: ...
@overload  # Nd ~bool | ~f16 | ~f80 | ~c160, mode: r
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def rq(
    a: _AsF16ND | _AsC80ND, overwrite_a: bool = False, lwork: int | None = None, *, mode: _ModeR, check_finite: bool = True
) -> onp.ArrayND[np.float64 | Any]: ...
@overload  # 2d catch-all, mode: {full, economic}
def rq(
    a: onp.ToComplexStrict2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: bool = True,
) -> _Tuple2[onp.Array2D[np.float64 | Any]]: ...
@overload  # Nd catch-all, mode: {full, economic}
def rq(
    a: onp.ToComplexND,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float64 | Any]]: ...
@overload  # 2d catch-all, mode: r
def rq(
    a: onp.ToComplexStrict2D, overwrite_a: bool = False, lwork: int | None = None, *, mode: _ModeR, check_finite: bool = True
) -> onp.Array2D[np.float64 | Any]: ...
@overload  # Nd catch-all, mode: r
def rq(
    a: onp.ToComplexND, overwrite_a: bool = False, lwork: int | None = None, *, mode: _ModeR, check_finite: bool = True
) -> onp.ArrayND[np.float64 | Any]: ...
