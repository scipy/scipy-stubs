from collections.abc import Callable
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Numeric
from scipy.sparse.linalg import LinearOperator

__all__ = ["bicg", "bicgstab", "cg", "cgs", "gmres", "qmr"]

_Float: TypeAlias = np.float32 | np.float64
_Complex: TypeAlias = np.complex64 | np.complex128

_ToInt: TypeAlias = np.integer[Any] | np.bool_
_ToFloat: TypeAlias = _Float | _ToInt
_ToComplex: TypeAlias = _Complex | _ToFloat

_FloatT = TypeVar("_FloatT", bound=_Float, default=np.float64)
_ComplexT = TypeVar("_ComplexT", bound=_Complex)
_ScalarT = TypeVar("_ScalarT", bound=Numeric)

_ToLinearOperator: TypeAlias = onp.CanArrayND[_ScalarT] | _spbase[_ScalarT] | LinearOperator[_ScalarT]

_Ignored: TypeAlias = object
_Callback: TypeAlias = Callable[[onp.Array1D[_ScalarT]], _Ignored]

###

@overload  # real
def bicg(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_FloatT | _ToInt] | None = None,
    callback: _Callback[_FloatT] | None = None,
) -> tuple[onp.Array1D[_FloatT], int | bool]: ...
@overload  # complex | float | int | bool
def bicg(
    A: _ToLinearOperator[_ComplexT],
    b: onp.ToComplex1D,
    x0: onp.ToComplex1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_ComplexT] | None = None,
    callback: _Callback[_ComplexT] | None = None,
) -> tuple[onp.Array1D[_ComplexT], int | bool]: ...

#
@overload  # real
def bicgstab(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_FloatT | _ToInt] | None = None,
    callback: _Callback[_FloatT] | None = None,
) -> tuple[onp.Array1D[_FloatT], int | bool]: ...
@overload  # complex | float | int | bool
def bicgstab(
    A: _ToLinearOperator[_ComplexT],
    b: onp.ToComplex1D,
    x0: onp.ToComplex1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_ComplexT] | None = None,
    callback: _Callback[_ComplexT] | None = None,
) -> tuple[onp.Array1D[_ComplexT], int | bool]: ...

#
@overload  # real
def cg(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_FloatT | _ToInt] | None = None,
    callback: _Callback[_FloatT] | None = None,
) -> tuple[onp.Array1D[_FloatT], int | bool]: ...
@overload  # complex | float | int | bool
def cg(
    A: _ToLinearOperator[_ComplexT],
    b: onp.ToComplex1D,
    x0: onp.ToComplex1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_ComplexT] | None = None,
    callback: _Callback[_ComplexT] | None = None,
) -> tuple[onp.Array1D[_ComplexT], int | bool]: ...

#
def cgs(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_FloatT | _ToInt] | None = None,
    callback: _Callback[_FloatT] | None = None,
) -> tuple[onp.Array1D[_FloatT], int | bool]: ...

#
@overload  # real, callback_type: {"pr_norm", "legacy"} | None = ...
def gmres(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    restart: int | bool | None = None,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_ToFloat] | None = None,
    callback: Callable[[float | int | bool], _Ignored] | Callable[[np.float64], _Ignored] | None = None,
    callback_type: Literal["pr_norm", "legacy"] | None = ...,
) -> tuple[onp.Array1D[_FloatT], int | bool]: ...
@overload  # real, callback_type: {"x"}
def gmres(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    restart: int | bool | None = None,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_ToFloat] | None = None,
    callback: _Callback[_FloatT] | None = None,
    callback_type: Literal["x"],
) -> tuple[onp.Array1D[_FloatT], int | bool]: ...
@overload  # complex | float | int | bool, callback_type: {"pr_norm", "legacy"} | None = ...
def gmres(
    A: _ToLinearOperator[_ComplexT],
    b: onp.ToComplex1D,
    x0: onp.ToComplex1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    restart: int | bool | None = None,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_ToFloat] | None = None,
    callback: Callable[[float | int | bool], _Ignored] | Callable[[np.float64], _Ignored] | None = None,
    callback_type: Literal["pr_norm", "legacy"] | None = ...,
) -> tuple[onp.Array1D[_ComplexT], int | bool]: ...
@overload  # complex | float | int | bool, callback_type: {"x"}
def gmres(
    A: _ToLinearOperator[_ComplexT],
    b: onp.ToComplex1D,
    x0: onp.ToComplex1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    restart: int | bool | None = None,
    maxiter: int | bool | None = None,
    M: _ToLinearOperator[_ToComplex] | None = None,
    callback: _Callback[_ComplexT] | None = None,
    callback_type: Literal["x"],
) -> tuple[onp.Array1D[_ComplexT], int | bool]: ...

#
@overload  # real
def qmr(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | bool | None = None,
    M1: _ToLinearOperator[_ToFloat] | None = None,
    M2: _ToLinearOperator[_ToFloat] | None = None,
    callback: _Callback[_FloatT] | None = None,
) -> tuple[onp.Array1D[_FloatT], int | bool]: ...
@overload  # complex | float | int | bool
def qmr(
    A: _ToLinearOperator[_ComplexT],
    b: onp.ToComplex1D,
    x0: onp.ToComplex1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | bool | None = None,
    M1: _ToLinearOperator[_ToComplex] | None = None,
    M2: _ToLinearOperator[_ToComplex] | None = None,
    callback: _Callback[_ComplexT] | None = None,
) -> tuple[onp.Array1D[_ComplexT], int | bool]: ...
