from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp

from scipy._typing import Falsy, Truthy

__all__ = ["diagsvd", "null_space", "orth", "subspace_angles", "svd", "svdvals"]

_Float: TypeAlias = np.float32 | np.float64
_FloatND: TypeAlias = onp.ArrayND[_Float]

_Complex: TypeAlias = np.complex64 | np.complex128
_ComplexND: TypeAlias = onp.ArrayND[_Complex]

_LapackDriver: TypeAlias = Literal["gesdd", "gesvd"]

_FloatSVD: TypeAlias = tuple[_FloatND, _FloatND, _FloatND]
_ComplexSVD: TypeAlias = tuple[_ComplexND, _FloatND, _ComplexND]

_RealT = TypeVar("_RealT", bound=np.bool_ | np.integer[Any] | np.floating[Any])
_InexactT = TypeVar("_InexactT", bound=_Float | _Complex)

###

@overload
def svd(
    a: onp.ToFloatND,
    full_matrices: onp.ToBool = True,
    compute_uv: Truthy = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _FloatSVD: ...
@overload
def svd(
    a: onp.ToComplexND,
    full_matrices: onp.ToBool = True,
    compute_uv: Truthy = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _FloatSVD | _ComplexSVD: ...
@overload  # complex, compute_uv: {False}
def svd(
    a: onp.ToComplexND,
    full_matrices: onp.ToBool,
    compute_uv: Falsy,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _FloatND: ...
@overload  # complex, *, compute_uv: {False}
def svd(
    a: onp.ToComplexND,
    full_matrices: onp.ToBool = True,
    *,
    compute_uv: Falsy,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _FloatND: ...

#
def svdvals(a: onp.ToComplexND, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _FloatND: ...

# beware the overlapping overloads for bool <: int (<: float)
@overload
def diagsvd(s: onp.SequenceND[_RealT] | onp.CanArrayND[_RealT], M: op.CanIndex, N: op.CanIndex) -> onp.ArrayND[_RealT]: ...
@overload
def diagsvd(s: onp.SequenceND[bool], M: op.CanIndex, N: op.CanIndex) -> onp.ArrayND[np.bool_]: ...
@overload
def diagsvd(s: onp.SequenceND[op.JustInt], M: op.CanIndex, N: op.CanIndex) -> onp.ArrayND[np.int_]: ...
@overload
def diagsvd(s: onp.SequenceND[op.JustFloat], M: op.CanIndex, N: op.CanIndex) -> onp.ArrayND[np.float64]: ...

#
@overload
def orth(A: onp.ToIntND | onp.ToJustFloat64_ND, rcond: onp.ToFloat | None = None) -> onp.ArrayND[np.float64]: ...
@overload
def orth(A: onp.ToJustComplex128_ND, rcond: onp.ToFloat | None = None) -> onp.ArrayND[np.complex128]: ...
@overload
def orth(
    A: onp.SequenceND[_InexactT] | onp.CanArrayND[_InexactT], rcond: onp.ToFloat | None = None
) -> onp.ArrayND[_InexactT]: ...

#
@overload
def null_space(
    A: onp.ToIntND | onp.ToJustFloat64_ND,
    rcond: onp.ToFloat | None = None,
    *,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> onp.ArrayND[np.float64]: ...
@overload
def null_space(
    A: onp.ToJustComplex128_ND,
    rcond: onp.ToFloat | None = None,
    *,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> onp.ArrayND[np.complex128]: ...
@overload
def null_space(
    A: onp.SequenceND[_InexactT] | onp.CanArrayND[_InexactT],
    rcond: onp.ToFloat | None = None,
    *,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> onp.ArrayND[_InexactT]: ...

#
def subspace_angles(A: onp.ToComplexND, B: onp.ToComplexND) -> _FloatND: ...
