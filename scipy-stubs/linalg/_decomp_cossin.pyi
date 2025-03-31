from collections.abc import Iterable
from typing import Any, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import Falsy, Truthy

__all__ = ["cossin"]

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple3: TypeAlias = tuple[_T, _T, _T]

_Float1D: TypeAlias = onp.Array1D[np.floating[Any]]
_Float2D: TypeAlias = onp.Array2D[np.floating[Any]]
_Complex2D: TypeAlias = onp.Array2D[np.inexact[Any]]

@overload  # (float | int | bool[:, :], separate=False) -> float | int | bool[:, :]**3
def cossin(
    X: onp.ToFloat2D | Iterable[onp.ToFloat2D],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    separate: Falsy = False,
    swap_sign: onp.ToBool = False,
    compute_u: onp.ToBool = True,
    compute_vh: onp.ToBool = True,
) -> _Tuple3[_Float2D]: ...
@overload  # (float | int | bool[:, :], *, separate=True) -> (float | int | bool[:, :]**2, float | int | bool[:], float | int | bool[:, :]**2)
def cossin(
    X: onp.ToFloat2D | Iterable[onp.ToFloat2D],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    *,
    separate: Truthy,
    swap_sign: onp.ToBool = False,
    compute_u: onp.ToBool = True,
    compute_vh: onp.ToBool = True,
) -> tuple[_Tuple2[_Float2D], _Float1D, _Tuple2[_Float2D]]: ...
@overload  # (complex | float | int | bool[:, :], separate=False) -> complex | float | int | bool[:, :]**3
def cossin(
    X: onp.ToComplex2D | Iterable[onp.ToComplex2D],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    separate: Falsy = False,
    swap_sign: onp.ToBool = False,
    compute_u: onp.ToBool = True,
    compute_vh: onp.ToBool = True,
) -> _Tuple3[_Complex2D]: ...
@overload  # (complex | float | int | bool[:, :], separate=True) -> (complex | float | int | bool[:, :]**2, float | int | bool[:], complex | float | int | bool[:, :]**2)
def cossin(
    X: onp.ToComplex2D | Iterable[onp.ToComplex2D],
    p: opt.AnyInt | None = None,
    q: opt.AnyInt | None = None,
    *,
    separate: Truthy,
    swap_sign: onp.ToBool = False,
    compute_u: onp.ToBool = True,
    compute_vh: onp.ToBool = True,
) -> tuple[_Tuple2[_Complex2D], _Float1D, _Tuple2[_Complex2D]]: ...
