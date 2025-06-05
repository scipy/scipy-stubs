from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
from numpy.linalg import LinAlgError

from scipy._typing import AnyBool, Falsy, Truthy

__all__ = ["LinAlgError", "LinAlgWarning", "norm"]

_Inf: TypeAlias = float
_Order: TypeAlias = Literal["fro", "nuc", 0, 1, -1, 2, -2] | _Inf
_Axis: TypeAlias = op.CanIndex | tuple[op.CanIndex, op.CanIndex]

_SubScalar: TypeAlias = np.complex128 | np.float64 | np.integer[Any] | np.bool_

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

###

class LinAlgWarning(RuntimeWarning): ...

# NOTE: the mypy errors are false positives (join vs union)

@overload  # scalar, axis: None = ...
def norm(  # type: ignore[overload-overlap]
    a: complex | _SubScalar,
    ord: _Order | None = None,
    axis: None = None,
    keepdims: op.CanBool = False,
    check_finite: AnyBool = True,
) -> np.float64: ...
@overload  # inexact32, axis: None = ...
def norm(
    a: np.float32 | np.complex64,
    ord: _Order | None = None,
    axis: None = None,
    keepdims: op.CanBool = False,
    check_finite: AnyBool = True,
) -> np.float32: ...
@overload  # longdouble, axis: None = ...
def norm(
    a: np.longdouble | np.clongdouble,
    ord: _Order | None = None,
    axis: None = None,
    keepdims: op.CanBool = False,
    check_finite: AnyBool = True,
) -> np.longdouble: ...
@overload  # scalar array, axis: None = ..., keepdims: False = ...
def norm(
    a: onp.CanArrayND[_SubScalar] | onp.SequenceND[onp.CanArrayND[_SubScalar]] | onp.SequenceND[_SubScalar],
    ord: _Order | None = None,
    axis: None = None,
    keepdims: Falsy = False,
    check_finite: AnyBool = True,
) -> np.float64: ...
@overload  # float64-coercible array, keepdims: True (positional)
def norm(  # type: ignore[overload-overlap]
    a: onp.ArrayND[_SubScalar, _ShapeT], ord: _Order | None, axis: _Axis | None, keepdims: Truthy, check_finite: AnyBool = True
) -> onp.ArrayND[np.float64, _ShapeT]: ...
@overload  # float64-coercible array, keepdims: True (keyword)
def norm(  # type: ignore[overload-overlap]
    a: onp.ArrayND[_SubScalar, _ShapeT],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float64, _ShapeT]: ...
@overload  # float64-coercible array-like, keepdims: True (positional)
def norm(  # type: ignore[overload-overlap]
    a: onp.SequenceND[onp.CanArrayND[_SubScalar]] | onp.SequenceND[complex | _SubScalar],
    ord: _Order | None,
    axis: _Axis | None,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # float64-coercible array-like, keepdims: True (keyword)
def norm(  # type: ignore[overload-overlap]
    a: onp.SequenceND[onp.CanArrayND[_SubScalar]] | onp.SequenceND[complex | _SubScalar],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # shaped inexact32 array, keepdims: True (positional)
def norm(
    a: onp.ArrayND[np.float32 | np.complex64, _ShapeT],
    ord: _Order | None,
    axis: _Axis | None,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float32, _ShapeT]: ...
@overload  # shaped longdouble array, keepdims: True (positional)
def norm(
    a: onp.ArrayND[np.longdouble | np.clongdouble, _ShapeT],
    ord: _Order | None,
    axis: _Axis | None,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.longdouble, _ShapeT]: ...
@overload  # shaped inexact32 array, keepdims: True (keyword)
def norm(
    a: onp.ArrayND[np.float32 | np.complex64, _ShapeT],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float32, _ShapeT]: ...
@overload  # shaped longdouble array, keepdims: True (keyword)
def norm(
    a: onp.ArrayND[np.longdouble | np.clongdouble, _ShapeT],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.longdouble, _ShapeT]: ...
@overload  # scalar array-like, keepdims: True (positional)
def norm(
    a: onp.SequenceND[onp.CanArrayND[np.float32 | np.complex64]] | onp.SequenceND[np.float32 | np.complex64],
    ord: _Order | None,
    axis: _Axis | None,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # scalar array-like, keepdims: True (positional)
def norm(
    a: onp.SequenceND[onp.CanArrayND[np.longdouble | np.clongdouble]] | onp.SequenceND[np.longdouble | np.clongdouble],
    ord: _Order | None,
    axis: _Axis | None,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.longdouble]: ...
@overload  # scalar array-like, keepdims: True (keyword)
def norm(
    a: onp.SequenceND[onp.CanArrayND[np.float32 | np.complex64]] | onp.SequenceND[np.float32 | np.complex64],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # scalar array-like, keepdims: True (keyword)
def norm(
    a: onp.SequenceND[onp.CanArrayND[np.longdouble | np.clongdouble]] | onp.SequenceND[np.longdouble | np.clongdouble],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.longdouble]: ...
@overload  # array-like, axis: None = ..., keepdims: False = ...
def norm(
    a: onp.ToComplexND, ord: _Order | None = None, axis: None = None, keepdims: Falsy = False, check_finite: AnyBool = True
) -> np.float64: ...
@overload  # array-like, keepdims: True (positional)
def norm(
    a: onp.ToComplexND, ord: _Order | None, axis: _Axis | None, keepdims: Truthy, check_finite: AnyBool = True
) -> onp.ArrayND[np.floating[Any]]: ...
@overload  # array-like, keepdims: True (keyword)
def norm(
    a: onp.ToComplexND, ord: _Order | None = None, axis: _Axis | None = None, *, keepdims: Truthy, check_finite: AnyBool = True
) -> onp.ArrayND[np.floating[Any]]: ...
@overload  # catch-all
def norm(
    a: onp.ToArrayND,
    ord: _Order | None = None,
    axis: _Axis | None = None,
    keepdims: AnyBool = False,
    check_finite: AnyBool = True,
) -> np.floating[Any] | onp.ArrayND[np.floating[Any]]: ...
