from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, Truthy

__all__ = ["qr", "qr_multiply", "rq"]

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]

_Int1D: TypeAlias = onp.Array1D[np.int32 | np.int64]
_Float1D: TypeAlias = onp.Array1D[np.floating[Any]]
_Float2D: TypeAlias = onp.Array2D[np.floating[Any]]
_Complex1D: TypeAlias = onp.Array1D[np.inexact[Any]]
_Complex2D: TypeAlias = onp.Array2D[np.inexact[Any]]

_Side: TypeAlias = Literal["left", "right"]
_ModeFullEcon: TypeAlias = Literal["full", "economic"]
_ModeR: TypeAlias = Literal["r"]
_ModeRaw: TypeAlias = Literal["raw"]

###

# 2 * (3 + 4 + 4) = 22 overloads (10/22 handle the positional cases of `mode`/`pivoting`)
@overload  # float | int | bool, mode: {full, economic}, pivoting: {False}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    mode: _ModeFullEcon = "full",
    pivoting: Falsy = False,
    check_finite: onp.ToBool = True,
) -> _Tuple2[_Float2D]: ...
@overload  # float | int | bool, mode: {full, economic}, pivoting: {True}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeFullEcon,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, _Float2D, _Int1D]: ...
@overload  # float | int | bool, mode: {full, economic}, *, pivoting: {True}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    mode: _ModeFullEcon = "full",
    *,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, _Float2D, _Int1D]: ...
@overload  # float | int | bool, mode: {r}, pivoting: {False}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeR,
    pivoting: Falsy = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D]: ...
@overload  # float | int | bool, mode: {r}, pivoting: {True}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeR,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, _Int1D]: ...
@overload  # float | int | bool, *, mode: {r}, pivoting: {False}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    *,
    mode: _ModeR,
    pivoting: Falsy = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D]: ...
@overload  # float | int | bool, *, mode: {r}, pivoting: {True}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    *,
    mode: _ModeR,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, _Int1D]: ...
@overload  # float | int | bool, mode: {raw}, pivoting: {False}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeRaw,
    pivoting: Falsy = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Tuple2[_Float2D], _Float2D]: ...
@overload  # float | int | bool, mode: {raw}, pivoting: {True}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeRaw,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Tuple2[_Float2D], _Float2D, _Int1D]: ...
@overload  # float | int | bool, *, mode: {raw}, pivoting: {False}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    *,
    mode: _ModeRaw,
    pivoting: Falsy = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Tuple2[_Float2D], _Float2D]: ...
@overload  # float | int | bool, *, mode: {raw}, pivoting: {True}
def qr(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    *,
    mode: _ModeRaw,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Tuple2[_Float2D], _Float2D, _Int1D]: ...
@overload  # complex | float | int | bool, mode: {full, economic}, pivoting: {False}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    mode: _ModeFullEcon = "full",
    pivoting: Falsy = False,
    check_finite: onp.ToBool = True,
) -> _Tuple2[_Complex2D]: ...
@overload  # complex | float | int | bool, mode: {full, economic}, pivoting: {True}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeFullEcon,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D, _Complex2D, _Int1D]: ...
@overload  # complex | float | int | bool, mode: {full, economic}, *, pivoting: {True}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    mode: _ModeFullEcon = "full",
    *,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D, _Complex2D, _Int1D]: ...
@overload  # complex | float | int | bool, mode: {r}, pivoting: {False}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeR,
    pivoting: Falsy = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D]: ...
@overload  # complex | float | int | bool, mode: {r}, pivoting: {True}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeR,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D, _Int1D]: ...
@overload  # complex | float | int | bool, *, mode: {r}, pivoting: {False}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    *,
    mode: _ModeR,
    pivoting: Falsy = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D]: ...
@overload  # complex | float | int | bool, *, mode: {r}, pivoting: {True}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    *,
    mode: _ModeR,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Complex2D, _Int1D]: ...
@overload  # complex | float | int | bool, mode: {raw}, pivoting: {False}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeRaw,
    pivoting: Falsy = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Tuple2[_Complex2D], _Complex2D]: ...
@overload  # complex | float | int | bool, mode: {raw}, pivoting: {True}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeRaw,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Tuple2[_Complex2D], _Complex2D, _Int1D]: ...
@overload  # complex | float | int | bool, *, mode: {raw}, pivoting: {False}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    *,
    mode: _ModeRaw,
    pivoting: Falsy = False,
    check_finite: onp.ToBool = True,
) -> tuple[_Tuple2[_Complex2D], _Complex2D]: ...
@overload  # complex | float | int | bool, *, mode: {raw}, pivoting: {True}
def qr(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    *,
    mode: _ModeRaw,
    pivoting: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_Tuple2[_Complex2D], _Complex2D, _Int1D]: ...

#
@overload  # (float | int | bool[:, :], float | int | bool[:], pivoting=False) -> (float | int | bool[:], float | int | bool[:, :])
def qr_multiply(
    a: onp.ToFloat2D,
    c: onp.ToFloatStrict1D,
    mode: _Side = "right",
    pivoting: Falsy = False,
    conjugate: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_c: onp.ToBool = False,
) -> tuple[_Float1D, _Complex2D]: ...
@overload  # (float | int | bool[:, :], float | int | bool[:, :], pivoting=False) -> (float | int | bool[:, :], float | int | bool[:, :])
def qr_multiply(
    a: onp.ToFloat2D,
    c: onp.ToFloatStrict2D,
    mode: _Side = "right",
    pivoting: Falsy = False,
    conjugate: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_c: onp.ToBool = False,
) -> tuple[_Float2D, _Complex2D]: ...
@overload  # (float | int | bool[:, :], float | int | bool[:, :?], pivoting=False) -> (float | int | bool[:, :?], float | int | bool[:, :])
def qr_multiply(
    a: onp.ToFloat2D,
    c: onp.ToFloat1D | onp.ToFloat2D,
    mode: _Side = "right",
    pivoting: Falsy = False,
    conjugate: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_c: onp.ToBool = False,
) -> tuple[_Float1D | _Float2D, _Complex2D]: ...
@overload  # (float | int | bool[:, :], float | int | bool[:, :?], pivoting=True, /) -> (float | int | bool[:, :?], float | int | bool[:, :], int | bool[:])
def qr_multiply(
    a: onp.ToFloat2D,
    c: onp.ToFloat1D | onp.ToFloat2D,
    mode: _Side,
    pivoting: Truthy,
    conjugate: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_c: onp.ToBool = False,
) -> tuple[_Float1D | _Float2D, _Float2D, _Int1D]: ...
@overload  # (float | int | bool[:, :], float | int | bool[:, :?], *, pivoting=True) -> (float | int | bool[:, :?], float | int | bool[:, :], int | bool[:])
def qr_multiply(
    a: onp.ToFloat2D,
    c: onp.ToFloat1D | onp.ToFloat2D,
    mode: _Side = "right",
    *,
    pivoting: Truthy,
    conjugate: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_c: onp.ToBool = False,
) -> tuple[_Float1D | _Float2D, _Float2D, _Int1D]: ...
@overload  # (complex | float | int | bool[:, :], complex | float | int | bool[:, :?], pivoting=False) -> (complex | float | int | bool[:, :?], complex | float | int | bool[:, :])
def qr_multiply(
    a: onp.ToComplex2D,
    c: onp.ToComplex1D | onp.ToComplex2D,
    mode: _Side = "right",
    pivoting: Falsy = False,
    conjugate: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_c: onp.ToBool = False,
) -> tuple[_Complex1D | _Complex2D, _Complex2D]: ...
@overload  # (complex | float | int | bool[:, :], complex | float | int | bool[:, :?], pivoting=True, /) -> (complex | float | int | bool[:, :?], complex | float | int | bool[:, :], int | bool[:])
def qr_multiply(
    a: onp.ToComplex2D,
    c: onp.ToComplex1D | onp.ToComplex2D,
    mode: _Side,
    pivoting: Truthy,
    conjugate: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_c: onp.ToBool = False,
) -> tuple[_Complex1D | _Complex2D, _Complex2D, _Int1D]: ...
@overload  # (complex | float | int | bool[:, :], complex | float | int | bool[:, :?], *, pivoting=True) -> (complex | float | int | bool[:, :?], complex | float | int | bool[:, :], int | bool[:])
def qr_multiply(
    a: onp.ToComplex2D,
    c: onp.ToComplex1D | onp.ToComplex2D,
    mode: _Side = "right",
    *,
    pivoting: Truthy,
    conjugate: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_c: onp.ToBool = False,
) -> tuple[_Complex1D | _Complex2D, _Complex2D, _Int1D]: ...

#
@overload  # (float | int | bool[:, :], mode: {"full", "economic"}) -> (float | int | bool[:, :], float | int | bool[:, :])
def rq(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: onp.ToBool = True,
) -> tuple[_Float2D, _Float2D]: ...
@overload  # (float | int | bool[:, :], mode: {"r"}, /) -> float | int | bool[:, :]
def rq(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeR,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload  # (float | int | bool[:, :], *, mode: {"r"}) -> float | int | bool[:, :]
def rq(
    a: onp.ToFloat2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    *,
    mode: _ModeR,
    check_finite: onp.ToBool = True,
) -> _Float2D: ...
@overload  # (complex | float | int | bool[:, :], mode: {"full", "economic"}) -> (complex | float | int | bool[:, :], complex | float | int | bool[:, :])
def rq(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    mode: _ModeFullEcon = "full",
    check_finite: onp.ToBool = True,
) -> _Tuple2[_Complex2D]: ...
@overload  # (complex | float | int | bool[:, :], mode: {"r"}, /) -> complex | float | int | bool[:, :]
def rq(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool,
    lwork: onp.ToJustInt | None,
    mode: _ModeR,
    check_finite: onp.ToBool = True,
) -> _Complex2D: ...
@overload  # (complex | float | int | bool[:, :], *, mode: {"r"}) -> complex | float | int | bool[:, :]
def rq(
    a: onp.ToComplex2D,
    overwrite_a: onp.ToBool = False,
    lwork: onp.ToJustInt | None = None,
    *,
    mode: _ModeR,
    check_finite: onp.ToBool = True,
) -> _Complex2D: ...
