from collections.abc import Callable
from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp

__all__ = ["rsf2csf", "schur"]

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple2i: TypeAlias = tuple[_T, _T, int]

_FloatND: TypeAlias = onp.ArrayND[np.floating[Any]]
_ComplexND: TypeAlias = onp.ArrayND[np.complexfloating[Any, Any]]
_InexactND: TypeAlias = onp.ArrayND[np.inexact[Any]]

_OutputReal: TypeAlias = Literal["real", "r"]
_OutputComplex: TypeAlias = Literal["complex", "c"]

_Sort: TypeAlias = Literal["lhp", "rhp", "iuc", "ouc"] | Callable[[float, float], onp.ToBool]

###

@overload  # float, output: {"real"}, sort: _Sort (positional)
def schur(
    a: onp.ToFloatND,
    output: _OutputReal = "real",
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    sort: None = None,
    check_finite: onp.ToBool = True,
) -> _Tuple2[_FloatND]: ...
@overload  # float, output: {"real"}, sort: _Sort (keyword)
def schur(
    a: onp.ToFloatND,
    output: _OutputReal = "real",
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    *,
    sort: _Sort,
    check_finite: onp.ToBool = True,
) -> _Tuple2i[_InexactND]: ...
@overload  # complex, output: {"real"}, sort: _Sort (positional)
def schur(
    a: onp.ToComplexND,
    output: _OutputReal = "real",
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    sort: None = None,
    check_finite: onp.ToBool = True,
) -> _Tuple2[_InexactND]: ...
@overload  # complex, output: {"real"}, sort: _Sort (keyword)
def schur(
    a: onp.ToComplexND,
    output: _OutputReal = "real",
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    *,
    sort: _Sort,
    check_finite: onp.ToBool = True,
) -> _Tuple2i[_InexactND]: ...
@overload  # complex, output: {"complex"}, sort: _Sort (positional)
def schur(
    a: onp.ToComplexND,
    output: _OutputComplex,
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    sort: None = None,
    check_finite: onp.ToBool = True,
) -> _Tuple2[_ComplexND]: ...
@overload  # complex, output: {"complex"}, sort: _Sort (keyword)
def schur(
    a: onp.ToComplexND,
    output: _OutputComplex,
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    *,
    sort: _Sort,
    check_finite: onp.ToBool = True,
) -> _Tuple2i[_ComplexND]: ...

#
def rsf2csf(T: onp.ToFloatND, Z: onp.ToComplexND, check_finite: onp.ToBool = True) -> tuple[_ComplexND, _ComplexND]: ...
