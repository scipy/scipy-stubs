from collections.abc import Callable
from typing import Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["rsf2csf", "schur"]

###

_T = TypeVar("_T")

_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple2i: TypeAlias = tuple[_T, _T, int]

_ComplexND: TypeAlias = onp.ArrayND[npc.complexfloating]

_OutputReal: TypeAlias = Literal["real", "r"]
_OutputComplex: TypeAlias = Literal["complex", "c"]

_Sort: TypeAlias = Literal["lhp", "rhp", "iuc", "ouc"] | Callable[[float, float], bool]

_as_f32: TypeAlias = np.float32 | np.float16 | np.bool_  # noqa: PYI042
_as_f64: TypeAlias = npc.floating64 | npc.floating80 | npc.integer  # noqa: PYI042
_as_c128: TypeAlias = npc.complexfloating128 | npc.complexfloating160  # noqa: PYI042

###

@overload  # f64
def schur(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[float, _as_f64],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float64]]: ...
@overload  # f64, sort=<given>
def schur(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[float, _as_f64],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.float64]]: ...
@overload  # f32
def schur(
    a: onp.ToArrayND[np.float32, _as_f32],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.float32]]: ...
@overload  # f32, sort=<given>
def schur(
    a: onp.ToArrayND[np.float32, _as_f32],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.float32]]: ...
@overload  # c128
def schur(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[op.JustComplex, _as_c128],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex128]]: ...
@overload  # c128, sort=<given>
def schur(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[op.JustComplex, _as_c128],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex128]]: ...
@overload  # c128, output="complex"
def schur(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[complex, npc.inexact64 | npc.inexact80 | npc.integer],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex128]]: ...
@overload  # c128, output="complex", sort=<given>
def schur(  # type: ignore[overload-overlap]
    a: onp.ToArrayND[complex, npc.inexact64 | npc.inexact80 | npc.integer],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex128]]: ...
@overload  # c64
def schur(
    a: onp.ToArrayND[np.complex64, np.complex64],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex64]]: ...
@overload  # c64, sort=<given>
def schur(
    a: onp.ToArrayND[np.complex64, np.complex64],
    output: _OutputReal = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex64]]: ...
@overload  # c64, output="complex"
def schur(
    a: onp.ToArrayND[npc.inexact32, npc.inexact32 | np.float16 | np.bool_],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> _Tuple2[onp.ArrayND[np.complex64]]: ...
@overload  # c64, output="complex", sort=<given>
def schur(
    a: onp.ToArrayND[npc.inexact32, npc.inexact32 | np.float16 | np.bool_],
    output: _OutputComplex,
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: _Sort,
    check_finite: bool = True,
) -> _Tuple2i[onp.ArrayND[np.complex64]]: ...

# TODO(@jorenham)
def rsf2csf(T: onp.ToFloatND, Z: onp.ToComplexND, check_finite: bool = True) -> tuple[_ComplexND, _ComplexND]: ...
