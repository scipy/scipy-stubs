from _thread import _local as _Cache  # seriously, typeshed?
from typing import TypeAlias, overload

import numpy as np
import optype.numpy as onp

__all__ = ["cc_diff", "cs_diff", "diff", "hilbert", "ihilbert", "itilbert", "sc_diff", "shift", "ss_diff", "tilbert"]

# the suffix correspond to the relevant dtype charcode(s)
_Vec_d: TypeAlias = onp.Array1D[np.float64]
_Vec_dD: TypeAlias = onp.Array1D[np.float64 | np.complex128]

###

#
@overload
def diff(x: onp.ToFloat1D, order: int | bool = 1, period: onp.ToFloat | None = None, _cache: _Cache = ...) -> _Vec_d: ...
@overload
def diff(x: onp.ToComplex1D, order: int | bool = 1, period: onp.ToFloat | None = None, _cache: _Cache = ...) -> _Vec_dD: ...

#
@overload
def tilbert(x: onp.ToFloat1D, h: onp.ToFloat, period: onp.ToFloat | None = None, _cache: _Cache = ...) -> _Vec_d: ...
@overload
def tilbert(x: onp.ToComplex1D, h: onp.ToFloat, period: onp.ToFloat | None = None, _cache: _Cache = ...) -> _Vec_dD: ...

#
@overload
def itilbert(x: onp.ToFloat1D, h: onp.ToFloat, period: onp.ToFloat | None = None, _cache: _Cache = ...) -> _Vec_d: ...
@overload
def itilbert(x: onp.ToComplex1D, h: onp.ToFloat, period: onp.ToFloat | None = None, _cache: _Cache = ...) -> _Vec_dD: ...

#
@overload
def hilbert(x: onp.ToFloat1D, _cache: _Cache = ...) -> _Vec_d: ...
@overload
def hilbert(x: onp.ToComplex1D, _cache: _Cache = ...) -> _Vec_dD: ...

#
@overload
def ihilbert(x: onp.ToFloat1D) -> _Vec_d: ...
@overload
def ihilbert(x: onp.ToComplex1D) -> _Vec_dD: ...

#
@overload
def cs_diff(
    x: onp.ToFloat1D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    period: onp.ToFloat | None = None,
    _cache: _Cache = ...,
) -> _Vec_d: ...
@overload
def cs_diff(
    x: onp.ToComplex1D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    period: onp.ToFloat | None = None,
    _cache: _Cache = ...,
) -> _Vec_dD: ...

#
@overload
def sc_diff(
    x: onp.ToFloat1D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    period: onp.ToFloat | None = None,
    _cache: _Cache = ...,
) -> _Vec_d: ...
@overload
def sc_diff(
    x: onp.ToComplex1D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    period: onp.ToFloat | None = None,
    _cache: _Cache = ...,
) -> _Vec_dD: ...

#
@overload
def ss_diff(
    x: onp.ToFloat1D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    period: onp.ToFloat | None = None,
    _cache: _Cache = ...,
) -> _Vec_d: ...
@overload
def ss_diff(
    x: onp.ToComplex1D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    period: onp.ToFloat | None = None,
    _cache: _Cache = ...,
) -> _Vec_dD: ...

#
@overload
def cc_diff(
    x: onp.ToFloat1D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    period: onp.ToFloat | None = None,
    _cache: _Cache = ...,
) -> _Vec_d: ...
@overload
def cc_diff(
    x: onp.ToComplex1D,
    a: onp.ToFloat,
    b: onp.ToFloat,
    period: onp.ToFloat | None = None,
    _cache: _Cache = ...,
) -> _Vec_dD: ...

#
@overload
def shift(x: onp.ToFloat1D, a: onp.ToFloat, period: onp.ToFloat | None = None, _cache: _Cache = ...) -> _Vec_d: ...
@overload
def shift(x: onp.ToComplex1D, a: onp.ToFloat, period: onp.ToFloat | None = None, _cache: _Cache = ...) -> _Vec_dD: ...
