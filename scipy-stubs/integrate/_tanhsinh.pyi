from collections.abc import Callable
from typing import Any, Concatenate, Final, Generic, Literal, TypeAlias, TypedDict, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._lib._util import _RichResult

__all__ = ["nsum"]

_IntegralT_co = TypeVar("_IntegralT_co", covariant=True)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

_Args: TypeAlias = tuple[onp.ToScalar | onp.ToArrayND, ...]

@type_check_only
class _Tolerances(TypedDict, total=False):
    rtol: float
    atol: float

@type_check_only
class _TanhSinhResult(_RichResult[int | _IntegralT_co], Generic[_IntegralT_co]):
    success: Final[bool]
    status: Final[Literal[0, -1, -2, -3, -4, 1]]
    integral: _IntegralT_co
    error: _IntegralT_co
    maxlevel: Final[int]
    nfev: Final[int]

@type_check_only
class _NSumResult0(_RichResult[np.bool_ | np.int32 | np.float64]):
    success: Final[np.bool_]
    status: Final[np.int32]
    sum: Final[np.float64]
    error: Final[np.float64]
    nfev: Final[np.int32]

@type_check_only
class _NSumResultN(_RichResult[onp.ArrayND[np.bool_] | onp.ArrayND[np.int32] | onp.ArrayND[np.float64]]):
    success: Final[onp.ArrayND[np.bool_]]
    status: Final[onp.ArrayND[np.int32]]
    sum: Final[onp.ArrayND[np.float64]]
    error: Final[onp.ArrayND[np.float64]]
    nfev: Final[onp.ArrayND[np.int32]]

###

# The resulting integral type depends on the shape-types of the function signature, as well as the types of a, b, and args.
# This makes it infeasible to type precisely, as we cannot expect *every* user to annotate their functions with precise shape
# information. For an example of this, see https://github.com/scipy/scipy-stubs/issues/1005
@overload  # real f, scalar a, scalar b
def tanhsinh(
    f: Callable[Concatenate[onp.ArrayND[np.float64], ...], onp.ArrayND[npc.floating]],
    a: onp.ToFloat,
    b: onp.ToFloat,
    *,
    args: _Args = (),
    log: bool = False,
    maxlevel: int | None = None,
    minlevel: int | None = 2,
    atol: float | None = None,
    rtol: float | None = None,
    preserve_shape: bool = False,
    callback: Callable[[_TanhSinhResult[np.float64 | Any]], None] | None = None,
) -> _TanhSinhResult[np.float64 | Any]: ...
@overload  # real f, scalar/array a, array b
def tanhsinh(
    f: Callable[Concatenate[onp.ArrayND[np.float64], ...], onp.ArrayND[npc.floating]],
    a: onp.ToFloat | onp.ToFloatND,
    b: onp.ToFloatND,
    *,
    args: _Args = (),
    log: bool = False,
    maxlevel: int | None = None,
    minlevel: int | None = 2,
    atol: float | None = None,
    rtol: float | None = None,
    preserve_shape: bool = False,
    callback: Callable[[_TanhSinhResult[onp.ArrayND[np.float64]]], None] | None = None,
) -> _TanhSinhResult[onp.ArrayND[np.float64]]: ...
@overload  # real f, array a, scalar/array b
def tanhsinh(
    f: Callable[Concatenate[onp.ArrayND[np.float64], ...], onp.ArrayND[npc.floating]],
    a: onp.ToFloatND,
    b: onp.ToFloat | onp.ToFloatND,
    *,
    args: _Args = (),
    log: bool = False,
    maxlevel: int | None = None,
    minlevel: int | None = 2,
    atol: float | None = None,
    rtol: float | None = None,
    preserve_shape: bool = False,
    callback: Callable[[_TanhSinhResult[onp.ArrayND[np.float64]]], None] | None = None,
) -> _TanhSinhResult[onp.ArrayND[np.float64]]: ...
@overload  # complex f, scalar a, scalar b
def tanhsinh(
    f: Callable[Concatenate[onp.ArrayND[np.float64] | onp.ArrayND[np.complex128], ...], onp.ArrayND[npc.complexfloating]],
    a: onp.ToFloat,
    b: onp.ToFloat,
    *,
    args: _Args = (),
    log: bool = False,
    maxlevel: int | None = None,
    minlevel: int | None = 2,
    atol: float | None = None,
    rtol: float | None = None,
    preserve_shape: bool = False,
    callback: Callable[[_TanhSinhResult[np.complex128 | Any]], None] | None = None,
) -> _TanhSinhResult[np.complex128 | Any]: ...
@overload  # complex f, scalar/array a, array b
def tanhsinh(
    f: Callable[Concatenate[onp.ArrayND[np.float64] | onp.ArrayND[np.complex128], ...], onp.ArrayND[npc.complexfloating]],
    a: onp.ToFloat | onp.ToFloatND,
    b: onp.ToFloatND,
    *,
    args: _Args = (),
    log: bool = False,
    maxlevel: int | None = None,
    minlevel: int | None = 2,
    atol: float | None = None,
    rtol: float | None = None,
    preserve_shape: bool = False,
    callback: Callable[[_TanhSinhResult[onp.ArrayND[np.complex128]]], None] | None = None,
) -> _TanhSinhResult[onp.ArrayND[np.complex128]]: ...
@overload  # complex f, array a, scalar/array b
def tanhsinh(
    f: Callable[Concatenate[onp.ArrayND[np.float64] | onp.ArrayND[np.complex128], ...], onp.ArrayND[npc.complexfloating]],
    a: onp.ToFloatND,
    b: onp.ToFloat | onp.ToFloatND,
    *,
    args: _Args = (),
    log: bool = False,
    maxlevel: int | None = None,
    minlevel: int | None = 2,
    atol: float | None = None,
    rtol: float | None = None,
    preserve_shape: bool = False,
    callback: Callable[[_TanhSinhResult[onp.ArrayND[np.complex128]]], None] | None = None,
) -> _TanhSinhResult[onp.ArrayND[np.complex128]]: ...

#
@overload
def nsum(
    f: Callable[Concatenate[onp.ArrayND[np.float64, _ShapeT], ...], onp.ArrayND[npc.floating, _ShapeT]],
    a: onp.ToFloat,
    b: onp.ToFloat,
    *,
    step: onp.ToFloat = 1,
    args: _Args = (),
    log: bool = False,
    maxterms: int = 0x10_00_00,
    tolerances: _Tolerances | None = None,
) -> _NSumResult0: ...
@overload
def nsum(
    f: Callable[Concatenate[onp.ArrayND[np.float64, _ShapeT], ...], onp.ArrayND[npc.floating, _ShapeT]],
    a: onp.ToFloat | onp.ToFloatND,
    b: onp.ToFloatND,
    *,
    step: onp.ToFloat | onp.ToFloatND = 1,
    args: _Args = (),
    log: bool = False,
    maxterms: int = 0x10_00_00,
    tolerances: _Tolerances | None = None,
) -> _NSumResultN: ...
@overload
def nsum(
    f: Callable[Concatenate[onp.ArrayND[np.float64, _ShapeT], ...], onp.ArrayND[npc.floating, _ShapeT]],
    a: onp.ToFloatND,
    b: onp.ToFloat | onp.ToFloatND,
    *,
    step: onp.ToFloat | onp.ToFloatND = 1,
    args: _Args = (),
    log: bool = False,
    maxterms: int = 0x10_00_00,
    tolerances: _Tolerances | None = None,
) -> _NSumResultN: ...
@overload
def nsum(
    f: Callable[Concatenate[onp.ArrayND[np.float64, _ShapeT], ...], onp.ArrayND[npc.floating, _ShapeT]],
    a: onp.ToFloat | onp.ToFloatND,
    b: onp.ToFloat | onp.ToFloatND,
    *,
    step: onp.ToFloatND,
    args: _Args = (),
    log: bool = False,
    maxterms: int = 0x10_00_00,
    tolerances: _Tolerances | None = None,
) -> _NSumResultN: ...
