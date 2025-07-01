from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._typing import Falsy, Truthy

__all__ = ["diagsvd", "null_space", "orth", "subspace_angles", "svd", "svdvals"]

_T = TypeVar("_T")
_Tuple3: TypeAlias = tuple[_T, _T, _T]

_Float: TypeAlias = np.float32 | np.float64
_FloatND: TypeAlias = onp.ArrayND[_Float]

_Complex: TypeAlias = np.complex64 | np.complex128

_LapackDriver: TypeAlias = Literal["gesdd", "gesvd"]

_RealT = TypeVar("_RealT", bound=np.bool_ | np.integer[Any] | np.floating[Any])
_InexactT = TypeVar("_InexactT", bound=_Float | _Complex)

_as_f32: TypeAlias = np.float32 | np.float16  # noqa: PYI042
_as_f64: TypeAlias = np.longdouble | np.float64 | npc.integer | np.bool_  # noqa: PYI042

###

@overload  # nd float64
def svd(
    a: onp.ToArrayND[float, _as_f64],
    full_matrices: onp.ToBool = True,
    compute_uv: Truthy = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _Tuple3[onp.ArrayND[np.float64]]: ...
@overload  # nd float32
def svd(
    a: onp.ToArrayND[_as_f32, _as_f32],
    full_matrices: onp.ToBool = True,
    compute_uv: Truthy = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> _Tuple3[onp.ArrayND[np.float32]]: ...
@overload  # nd complex128
def svd(
    a: onp.ToArrayND[op.JustComplex, np.complex128 | np.clongdouble],
    full_matrices: onp.ToBool = True,
    compute_uv: Truthy = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.float64], onp.ArrayND[np.complex128]]: ...
@overload  # nd complex64
def svd(
    a: onp.ToArrayND[np.complex64, np.complex64],
    full_matrices: onp.ToBool = True,
    compute_uv: Truthy = True,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.float32], onp.ArrayND[np.complex64]]: ...
@overload  # nd float64 | complex128, compute_uv=False (keyword)
def svd(
    a: onp.ToArrayND[complex, _as_f64 | np.complex128 | np.clongdouble],
    full_matrices: onp.ToBool = True,
    *,
    compute_uv: Falsy,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> onp.ArrayND[np.float64]: ...
@overload  # nd float32 | complex64, compute_uv=False (keyword)
def svd(
    a: onp.ToArrayND[_as_f32, _as_f32 | np.complex64],
    full_matrices: onp.ToBool = True,
    *,
    compute_uv: Falsy,
    overwrite_a: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver = "gesdd",
) -> onp.ArrayND[np.float32]: ...

#
def svdvals(a: onp.ToComplexND, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _FloatND: ...

#
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
