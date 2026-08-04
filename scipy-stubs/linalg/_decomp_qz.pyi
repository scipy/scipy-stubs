from collections.abc import Callable
from typing import Any, Literal, Never, overload
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["ordqz", "qz"]

###

type _FloatND = onp.ArrayND[npc.floating]
type _ComplexND = onp.ArrayND[npc.complexfloating]
type _InexactND = onp.ArrayND[npc.inexact]

type _Tuple4[T] = tuple[T, T, T, T]
type _Tuple2C12[T2, T1] = tuple[T2, T2, _ComplexND, T1, T2, T2]

type _AsF32ND = onp.ToArrayND[Never, npc.floating32 | npc.integer16 | npc.integer8]
type _AsF64ND = onp.ToArrayND[float, npc.floating64 | npc.integer64 | npc.integer32]
type _AsC64ND = onp.ToArrayND[Never, npc.inexact32 | npc.integer16 | npc.integer8]
type _AsC128ND = onp.ToArrayND[complex, npc.inexact64 | np.int64 | np.int32]
type _AsInexactND = onp.ToArrayND[complex, npc.inexact64 | npc.inexact32 | npc.integer]

type _OutputReal = Literal["real", "r"]
type _OutputComplex = Literal["complex", "c"]
type _Output = Literal[_OutputReal, _OutputComplex]

type _Sort = Literal["lhp", "rhp", "iuc", "ouc"] | Callable[[float, float], bool]

###

# NOTE: mypy incorrectly sees disjoint dtypes like `np.complex64` and `np.complex128` as overlapping
# mypy: disable-error-code=overload-overlap

# NOTE: `sort` will raise `ValueError` if not `None`.
@overload  # ~bool | ~f16 | ~f80 | ~c160, +complex
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def qz(
    A: onp.ToArrayND[Never, np.bool | np.float16 | npc.inexact80],
    B: onp.ToComplexND,
    output: _Output = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.float64 | Any]]: ...
@overload  # +complex, ~bool | ~f16 | ~f80 | ~c160
@deprecated("bool, float16, and longdouble input will no longer be supported in SciPy 2.1")
def qz(
    A: onp.ToComplexND,
    B: onp.ToArrayND[Never, np.bool | np.float16 | npc.inexact80],
    output: _Output = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.float64 | Any]]: ...
@overload  # +f32, +f32
def qz(
    A: _AsF32ND,
    B: _AsF32ND,
    output: _OutputReal = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.float32]]: ...
@overload  # +f64, +f32 | +f64
def qz(
    A: _AsF64ND,
    B: onp.ToArrayND[float, npc.floating64 | npc.floating32 | npc.integer],
    output: _OutputReal = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.float64]]: ...
@overload  # +f32, +f64
def qz(
    A: _AsF32ND,
    B: _AsF64ND,
    output: _OutputReal = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.float64]]: ...
@overload  # ~c64, +f32 | ~c64
def qz(
    A: onp.ToJustComplex64_ND,
    B: onp.ToArrayND[Never, npc.inexact32 | npc.integer16 | npc.integer8],
    output: _OutputReal = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.complex64]]: ...
@overload  # +f32, ~c64
def qz(
    A: _AsF32ND,
    B: onp.ToJustComplex64_ND,
    output: _OutputReal = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.complex64]]: ...
@overload  # ~c64, +f64
def qz(
    A: onp.ToJustComplex64_ND,
    B: _AsF64ND,
    output: _OutputReal = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.complex128]]: ...
@overload  # +f64, ~c64
def qz(
    A: _AsF64ND,
    B: onp.ToJustComplex64_ND,
    output: _OutputReal = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.complex128]]: ...
@overload  # ~c128, +inexact
def qz(
    A: onp.ToJustComplex128_ND,
    B: _AsInexactND,
    output: _OutputReal = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.complex128]]: ...
@overload  # +inexact, ~c128
def qz(
    A: _AsInexactND,
    B: onp.ToJustComplex128_ND,
    output: _OutputReal = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.complex128]]: ...
@overload  # +c64, +c64, output: complex
def qz(
    A: _AsC64ND,
    B: _AsC64ND,
    output: _OutputComplex,
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.complex64]]: ...
@overload  # +c128, +inexact, output: complex
def qz(
    A: _AsC128ND,
    B: _AsInexactND,
    output: _OutputComplex,
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.complex128]]: ...
@overload  # +inexact, +c128, output: complex
def qz(
    A: _AsInexactND,
    B: _AsC128ND,
    output: _OutputComplex,
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.complex128]]: ...
@overload  # catch-all
def qz(
    A: onp.ToComplexND,
    B: onp.ToComplexND,
    output: _OutputReal = "real",
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.float64 | Any]]: ...
@overload  # catch-all, output: complex
def qz(
    A: onp.ToComplexND,
    B: onp.ToComplexND,
    output: _OutputComplex,
    lwork: int | None = None,
    sort: None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple4[onp.ArrayND[np.complex128 | Any]]: ...

#
@overload  # float, {"real"}
def ordqz(
    A: onp.ToFloatND,
    B: onp.ToFloatND,
    sort: _Sort = "lhp",
    output: _OutputReal = "real",
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple2C12[_FloatND, _FloatND]: ...
@overload  # complex, {"real"}
def ordqz(
    A: onp.ToComplexND,
    B: onp.ToComplexND,
    sort: _Sort = "lhp",
    output: _OutputReal = "real",
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple2C12[_InexactND, _InexactND]: ...
@overload  # complex, {"complex"} (positional)
def ordqz(
    A: onp.ToComplexND,
    B: onp.ToComplexND,
    sort: _Sort,
    output: _OutputComplex,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple2C12[_ComplexND, _ComplexND]: ...
@overload  # complex, {"complex"} (keyword)
def ordqz(
    A: onp.ToComplexND,
    B: onp.ToComplexND,
    sort: _Sort = "lhp",
    *,
    output: _OutputComplex,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Tuple2C12[_ComplexND, _ComplexND]: ...
