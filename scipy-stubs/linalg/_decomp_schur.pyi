from collections.abc import Callable
from typing import Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["rsf2csf", "schur"]

###

_T = TypeVar("_T")
_Shape2T = TypeVar("_Shape2T", bound=tuple[int, int, *tuple[int, ...]])

_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple2i: TypeAlias = tuple[_T, _T, int]

_OutputReal: TypeAlias = Literal["real", "r"]
_OutputComplex: TypeAlias = Literal["complex", "c"]
_Output: TypeAlias = Literal[_OutputReal, _OutputComplex]

_Sort: TypeAlias = Literal["lhp", "rhp", "iuc", "ouc"] | Callable[[float, float], bool]

_as_f32: TypeAlias = np.float32 | np.float16 | np.bool_  # noqa: PYI042
_as_f64: TypeAlias = npc.floating64 | npc.floating80 | npc.integer  # noqa: PYI042
_as_c128: TypeAlias = npc.complexfloating128 | npc.complexfloating160  # noqa: PYI042

###

# NOTE: On numpy<2.1, pyright reports 12 false positive incompatible overload errors here.
# pyright: reportOverlappingOverload=false

# NOTE: The ignored `overload-overlap` mypy errors are false positives

#
@overload  # Nd f64
def schur(  # type: ignore[overload-overlap]
    a: onp.ArrayND[_as_f64, _Shape2T],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float64, _Shape2T]]: ...
@overload  # Nd f64, sort=<given>
def schur(  # type: ignore[overload-overlap]
    a: onp.ArrayND[_as_f64, _Shape2T],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.float64, _Shape2T]]: ...
@overload  # Nd f64, output="complex"
def schur(  # type: ignore[overload-overlap]
    a: onp.ArrayND[_as_f64, _Shape2T],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex128, _Shape2T]]: ...
@overload  # Nd f64, output="complex", sort=<given>
def schur(  # type: ignore[overload-overlap]
    a: onp.ArrayND[_as_f64, _Shape2T],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex128, _Shape2T]]: ...
@overload  # Nd f32
def schur(
    a: onp.ArrayND[_as_f32, _Shape2T],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float32, _Shape2T]]: ...
@overload  # Nd f32, sort=<given>
def schur(
    a: onp.ArrayND[_as_f32, _Shape2T],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.float32, _Shape2T]]: ...
@overload  # Nd f32, output="complex"
def schur(
    a: onp.ArrayND[_as_f32, _Shape2T],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex64, _Shape2T]]: ...
@overload  # Nd f32, output="complex", sort=<given>
def schur(
    a: onp.ArrayND[_as_f32, _Shape2T],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex64, _Shape2T]]: ...
@overload  # Nd c128
def schur(
    a: onp.ArrayND[_as_c128, _Shape2T],
    output: _Output = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex128, _Shape2T]]: ...
@overload  # Nd c128, sort=<given>
def schur(
    a: onp.ArrayND[_as_c128, _Shape2T],
    output: _Output = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex128, _Shape2T]]: ...
@overload  # Nd c64
def schur(
    a: onp.ArrayND[np.complex64, _Shape2T],
    output: _Output = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex64, _Shape2T]]: ...
@overload  # Nd c64, sort=<given>
def schur(
    a: onp.ArrayND[np.complex64, _Shape2T],
    output: _Output = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex64, _Shape2T]]: ...
@overload  # ?d f64
def schur(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[float, _as_f64],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float64]]: ...
@overload  # ?d f64, sort=<given>
def schur(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[float, _as_f64],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.float64]]: ...
@overload  # ?d f64, output="complex"
def schur(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[float, _as_f64],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex128]]: ...
@overload  # ?d f64, output="complex", sort=<given>
def schur(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[float, _as_f64],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex128]]: ...
@overload  # ?d f32
def schur(
    a: onp.ToArrayND[np.float32, _as_f32],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float32]]: ...
@overload  # ?d f32, sort=<given>
def schur(
    a: onp.ToArrayND[np.float32, _as_f32],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.float32]]: ...
@overload  # ?d f32, output="complex"
def schur(
    a: onp.ToArrayND[np.float32, _as_f32],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex64]]: ...
@overload  # ?d f32, output="complex", sort=<given>
def schur(
    a: onp.ToArrayND[np.float32, _as_f32],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex64]]: ...
@overload  # ?d c128
def schur(
    a: onp.ToArrayND[op.JustComplex, _as_c128],
    output: _Output = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex128]]: ...
@overload  # ?d c128, sort=<given>
def schur(
    a: onp.ToArrayND[op.JustComplex, _as_c128],
    output: _Output = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex128]]: ...
@overload  # ?d c64
def schur(
    a: onp.ToJustComplex64_ND,
    output: _Output = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex64]]: ...
@overload  # ?d c64, sort=<given>
def schur(
    a: onp.ToJustComplex64_ND,
    output: _Output = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex64]]: ...

# will raise for dtypes that don't have character code in `ilfdFD`
@overload  # ?d c128|f64|i64|i32, ?d c128|f64|i64|i32
def rsf2csf(  # type: ignore[overload-overlap]
    T: onp.ToArrayND[complex, npc.inexact64 | npc.integer64 | npc.integer32],
    Z: onp.ToArrayND[complex, npc.inexact64 | npc.integer64 | npc.integer32],
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex128]]: ...
@overload  # ?d f32|c64, ?d f32|c64 -> c64
def rsf2csf(
    T: onp.ToArrayND[npc.inexact32, npc.inexact32], Z: onp.ToArrayND[npc.inexact32, npc.inexact32], check_finite: bool = True
) -> _Tuple2[onp.ArrayND[np.complex64]]: ...
