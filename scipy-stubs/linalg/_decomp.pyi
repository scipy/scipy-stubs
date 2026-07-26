from collections.abc import Iterable, Sequence
from typing import Any, Literal, Never, overload
from typing_extensions import deprecated

import numpy as np
import numpy_typing_compat as nptc
import optype.numpy as onp
import optype.numpy.compat as npc
import optype.typing as opt

__all__ = [
    "cdf2rdf",
    "eig",
    "eig_banded",
    "eigh",
    "eigh_tridiagonal",
    "eigvals",
    "eigvals_banded",
    "eigvalsh",
    "eigvalsh_tridiagonal",
    "hessenberg",
]

# input types

type _ToBoolF16ND = onp.ToArrayND[Never, np.bool | np.float16]
type _ToF64ND = onp.ToArrayND[float, npc.integer32 | npc.integer64 | npc.floating64]
type _ToC128ND = onp.ToArrayND[complex, npc.integer32 | npc.number64]
type _ToInexact80ND = onp.ToArrayND[Never, npc.inexact80]

# NOTE: only "a", "v" and "i" are documented for the `select` params, but internally 0, 1, and 2 are used, respectively.
type _SelectA = Literal["a", "all", 0]
type _SelectV = Literal["v", "value", 1]
type _SelectI = Literal["i", "index", 2]

# NOTE: `_check_select()` requires the `select_range` array-like to be of `int{16,32,64}` when `select: _SelectIndex`
# https://github.com/scipy/scipy-stubs/issues/154
# NOTE: This `select_range` parameter type must be of shape `(2,)` and in nondescending order
type _SelectRange = Sequence[float | npc.integer | npc.floating]
type _SelectRangeI = Sequence[int | np.int16 | np.int32 | np.int64]  # no bool, int8 or unsigned ints

type _EigHType = Literal[1, 2, 3]
type _EigHSubsetByIndex = Iterable[opt.AnyInt]
type _EigHSubsetByValue = Iterable[onp.ToFloat]

type _DriverGV = Literal["gv", "gvd", "gvx"]
type _DriverEV = Literal["ev", "evd", "evx", "evr"]
type _DriverSTE = Literal["stemr", "stebz", "sterf", "stev"]
type _DriverAuto = Literal["auto"]

# output types

type _FloatND = onp.ArrayND[np.float64 | np.float32]
type _ComplexND = onp.ArrayND[np.complex128 | np.complex64]
type _InexactND = onp.ArrayND[np.complex128 | np.complex64 | np.float64 | np.float32]

###

# NOTE: mypy incorrectly sees disjoint dtypes like `npc.integer32` and `npc.integer8` as overlapping
# mypy: disable-error-code=overload-overlap

# NOTE: The eigenvectors of real `a` can be either real or complex, depending on its values.
# TODO(@jorenham): f32/f64/c64/c128-specific overloads
@overload  # complex, left: False, right: False (positional)
def eig(
    a: onp.ToComplexND,
    b: onp.ToComplexND | None,
    left: Literal[False],
    right: Literal[False],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> _ComplexND: ...
@overload  # complex, left: False = ..., right: False (keyword)
def eig(
    a: onp.ToComplexND,
    b: onp.ToComplexND | None = None,
    left: Literal[False] = False,
    *,
    right: Literal[False],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> _ComplexND: ...
@overload  # float, left: False = ..., right: True = ...
def eig(
    a: onp.ToFloatND,
    b: onp.ToFloatND | None = None,
    left: Literal[False] = False,
    right: Literal[True] = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_ComplexND, _InexactND]: ...
@overload  # complex, left: False = ..., right: True = ...
def eig(
    a: onp.ToJustComplexND,
    b: onp.ToComplexND | None = None,
    left: Literal[False] = False,
    right: Literal[True] = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_ComplexND, _ComplexND]: ...
@overload  # float, left: True (positional), right: False
def eig(
    a: onp.ToFloatND,
    b: onp.ToFloatND | None,
    left: Literal[True],
    right: Literal[False],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_ComplexND, _InexactND]: ...
@overload  # complex, left: True (positional), right: False
def eig(
    a: onp.ToJustComplexND,
    b: onp.ToComplexND | None,
    left: Literal[True],
    right: Literal[False],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_ComplexND, _ComplexND]: ...
@overload  # float, left: True (keyword), right: False
def eig(
    a: onp.ToFloatND,
    b: onp.ToFloatND | None = None,
    *,
    left: Literal[True],
    right: Literal[False],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_ComplexND, _InexactND]: ...
@overload  # complex, left: True (keyword), right: False (keyword)
def eig(
    a: onp.ToJustComplexND,
    b: onp.ToComplexND | None = None,
    *,
    left: Literal[True],
    right: Literal[False],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_ComplexND, _ComplexND]: ...
@overload  # float, left: True (positional), right: True = ...
def eig(
    a: onp.ToFloatND,
    b: onp.ToFloatND | None,
    left: Literal[True],
    right: Literal[True] = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_ComplexND, _InexactND, _InexactND]: ...
@overload  # complex, left: True (positional), right: True = ...
def eig(
    a: onp.ToJustComplexND,
    b: onp.ToComplexND | None,
    left: Literal[True],
    right: Literal[True] = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_ComplexND, _ComplexND, _ComplexND]: ...
@overload  # float, left: True (keyword), right: True = ...
def eig(
    a: onp.ToFloatND,
    b: onp.ToFloatND | None = None,
    *,
    left: Literal[True],
    right: Literal[True] = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_ComplexND, _InexactND, _InexactND]: ...
@overload  # complex, left: True (keyword), right: True = ...
def eig(
    a: onp.ToJustComplexND,
    b: onp.ToComplexND | None = None,
    *,
    left: Literal[True],
    right: Literal[True] = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> tuple[_ComplexND, _ComplexND, _ComplexND]: ...
@overload  # catch-all
def eig(
    a: onp.ToComplexND,
    b: onp.ToComplexND | None = None,
    left: bool = False,
    right: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> (
    _ComplexND
    | tuple[_ComplexND, _InexactND]
    | tuple[_ComplexND, _InexactND, _InexactND]
):  # fmt: skip
    ...

#
@overload  # +float64, eigvals_only: False = ...
def eigh(  #
    a: onp.ToArrayND[float, np.float64 | npc.floating80 | npc.integer64 | npc.integer32],
    b: onp.ToFloat64_ND | None = None,
    *,
    lower: bool = True,
    eigvals_only: Literal[False] = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...
@overload  # +float, eigvals_only: False = ...
def eigh(
    a: onp.ToFloatND,
    b: onp.ToFloatND | None = None,
    *,
    lower: bool = True,
    eigvals_only: Literal[False] = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> tuple[_FloatND, _FloatND]: ...
@overload  # ~complex, eigvals_only: False = ...
def eigh(
    a: onp.ToJustComplexND,
    b: onp.ToComplexND | None = None,
    *,
    lower: bool = True,
    eigvals_only: Literal[False] = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> tuple[_FloatND, _ComplexND]: ...
@overload  # +complex, eigvals_only: False = ...
def eigh(
    a: onp.ToComplexND,
    b: onp.ToComplexND | None = None,
    *,
    lower: bool = True,
    eigvals_only: Literal[False] = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> tuple[_FloatND, _InexactND]: ...
@overload  # +complex128, eigvals_only: True
def eigh(
    a: onp.ToArrayND[float, npc.inexact80 | npc.number64 | npc.integer32],
    b: onp.ToComplex128_ND | None = None,
    *,
    lower: bool = True,
    eigvals_only: Literal[True],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _EigHSubsetByValue | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload  # +complex, eigvals_only: True
def eigh(
    a: onp.ToComplexND,
    b: onp.ToComplexND | None = None,
    *,
    lower: bool = True,
    eigvals_only: Literal[True],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _EigHSubsetByValue | None = None,
) -> _FloatND: ...

#
@overload  # float, eigvals_only: False = ..., select: _SelectA = ...
def eig_banded(
    a_band: onp.ToFloatND,
    lower: bool = False,
    eigvals_only: Literal[False] = False,
    overwrite_a_band: bool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> tuple[_FloatND, _FloatND]: ...
@overload  # float, eigvals_only: False = ..., select: _SelectV (keyword)
def eig_banded(
    a_band: onp.ToFloatND,
    lower: bool = False,
    eigvals_only: Literal[False] = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> tuple[_FloatND, _FloatND]: ...
@overload  # float, eigvals_only: False = ..., select: _SelectI (keyword)
def eig_banded(
    a_band: onp.ToFloatND,
    lower: bool = False,
    eigvals_only: Literal[False] = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> tuple[_FloatND, _FloatND]: ...
@overload  # complex, eigvals_only: False = ..., select: _SelectA = ...
def eig_banded(
    a_band: onp.ToComplexND,
    lower: bool = False,
    eigvals_only: Literal[False] = False,
    overwrite_a_band: bool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> tuple[_FloatND, _InexactND]: ...
@overload  # complex, eigvals_only: False = ..., select: _SelectV (keyword)
def eig_banded(
    a_band: onp.ToComplexND,
    lower: bool = False,
    eigvals_only: Literal[False] = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> tuple[_FloatND, _InexactND]: ...
@overload  # complex, eigvals_only: False = ..., select: _SelectI (keyword)
def eig_banded(
    a_band: onp.ToComplexND,
    lower: bool = False,
    eigvals_only: Literal[False] = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> tuple[_FloatND, _InexactND]: ...
@overload  # eigvals_only: True  (positional), select: _SelectA = ...
def eig_banded(
    a_band: onp.ToComplexND,
    lower: bool,
    eigvals_only: Literal[True],
    overwrite_a_band: bool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> _FloatND: ...
@overload  # eigvals_only: True  (keyword), select: _SelectA = ... (keyword)
def eig_banded(
    a_band: onp.ToComplexND,
    lower: bool = False,
    *,
    eigvals_only: Literal[True],
    overwrite_a_band: bool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> _FloatND: ...
@overload  # eigvals_only: True  (positional), select: _SelectV (keyword)
def eig_banded(
    a_band: onp.ToComplexND,
    lower: bool,
    eigvals_only: Literal[True],
    overwrite_a_band: bool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> _FloatND: ...
@overload  # eigvals_only: True  (keyword), select: _SelectV (keyword)
def eig_banded(
    a_band: onp.ToComplexND,
    lower: bool = False,
    *,
    eigvals_only: Literal[True],
    overwrite_a_band: bool = False,
    select: _SelectV,
    select_range: _SelectRange,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> _FloatND: ...
@overload  # eigvals_only: True (positional), select: _SelectI (keyword)
def eig_banded(
    a_band: onp.ToComplexND,
    lower: bool,
    eigvals_only: Literal[True],
    overwrite_a_band: bool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> _FloatND: ...
@overload  # eigvals_only: True (keyword), select: _SelectI (keyword)
def eig_banded(
    a_band: onp.ToComplexND,
    lower: bool = False,
    *,
    eigvals_only: Literal[True],
    overwrite_a_band: bool = False,
    select: _SelectI,
    select_range: _SelectRangeI,
    max_ev: onp.ToInt = 0,
    check_finite: bool = True,
) -> _FloatND: ...

# keep structurally in sync with `eigvalsh`
@overload  # ~bool | ~f16, +c64 | None
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvals(
    a: _ToBoolF16ND,
    b: onp.ToComplex64_ND | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> onp.ArrayND[np.complex64]: ...
@overload  # +c64, ~bool | ~f16
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvals(
    a: onp.ToComplex64_ND,
    b: _ToBoolF16ND,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> onp.ArrayND[np.complex64]: ...
@overload  # ~bool | ~f16, +c128
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvals(
    a: _ToBoolF16ND,
    b: _ToC128ND,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +c128, ~bool | ~f16
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvals(
    a: _ToC128ND,
    b: _ToBoolF16ND,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload  # ~f80 | ~c160, +complex | None
@deprecated("longdouble and clongdouble input will no longer be supported in SciPy 2.1")
def eigvals(
    a: _ToInexact80ND,
    b: onp.ToComplexND | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, ~f80 | ~c160
@deprecated("longdouble and clongdouble input will no longer be supported in SciPy 2.1")
def eigvals(
    a: onp.ToComplexND,
    b: _ToInexact80ND,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +c64, +c64 | None
def eigvals(
    a: onp.ToComplex64_ND,
    b: onp.ToComplex64_ND | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> onp.ArrayND[np.complex64]: ...
@overload  # +c128, +complex | None
def eigvals(
    a: _ToC128ND,
    b: onp.ToComplexND | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload  # +complex, +c128
def eigvals(
    a: onp.ToComplexND,
    b: _ToC128ND,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload  # catch-all
def eigvals(
    a: onp.ToComplexND,
    b: onp.ToComplexND | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    homogeneous_eigvals: bool = False,
) -> onp.ArrayND[np.complex128 | Any]: ...

# keep structurally in sync with `eigvals`
@overload  # ~bool | ~f16, +c64 | None
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh(
    a: _ToBoolF16ND,
    b: onp.ToComplex64_ND | None = None,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> onp.ArrayND[np.float32]: ...
@overload  # +c64, ~bool | ~f16
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh(
    a: onp.ToComplex64_ND,
    b: _ToBoolF16ND,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> onp.ArrayND[np.float32]: ...
@overload  # ~bool | ~f16, +c128
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh(
    a: _ToBoolF16ND,
    b: _ToC128ND,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload  # +c128, ~bool | ~f16
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh(
    a: _ToC128ND,
    b: _ToBoolF16ND,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload  # ~f80 | ~c160, +complex | None
@deprecated("longdouble and clongdouble input will no longer be supported in SciPy 2.1")
def eigvalsh(
    a: _ToInexact80ND,
    b: onp.ToComplexND | None = None,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload  # +complex, ~f80 | ~c160
@deprecated("longdouble and clongdouble input will no longer be supported in SciPy 2.1")
def eigvalsh(
    a: onp.ToComplexND,
    b: _ToInexact80ND,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload  # +c64, +c64 | None
def eigvalsh(
    a: onp.ToComplex64_ND,
    b: onp.ToComplex64_ND | None = None,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> onp.ArrayND[np.float32]: ...
@overload  # +c128, +complex | None
def eigvalsh(
    a: _ToC128ND,
    b: onp.ToComplexND | None = None,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload  # +complex, +c128
def eigvalsh(
    a: onp.ToComplexND,
    b: _ToC128ND,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload  # catch-all
def eigvalsh(
    a: onp.ToComplexND,
    b: onp.ToComplexND | None = None,
    *,
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    type: _EigHType = 1,
    check_finite: bool = True,
    subset_by_index: _EigHSubsetByIndex | None = None,
    subset_by_value: _EigHSubsetByValue | None = None,
    driver: _DriverEV | _DriverGV | None = None,
) -> onp.ArrayND[np.float64 | Any]: ...

#
@overload  # ~bool | ~f16
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvals_banded(
    a_band: _ToBoolF16ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # ~bool | ~f16, select: "v"
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvals_banded(
    a_band: _ToBoolF16ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # ~bool | ~f16, select: "i"
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvals_banded(
    a_band: _ToBoolF16ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # ~f80 | ~c160
@deprecated("longdouble and clongdouble input will no longer be supported in SciPy 2.1")
def eigvals_banded(
    a_band: _ToInexact80ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # ~f80 | ~c160, select: "v"
@deprecated("longdouble and clongdouble input will no longer be supported in SciPy 2.1")
def eigvals_banded(
    a_band: _ToInexact80ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # ~f80 | ~c160, select: "i"
@deprecated("longdouble and clongdouble input will no longer be supported in SciPy 2.1")
def eigvals_banded(
    a_band: _ToInexact80ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # +c64
def eigvals_banded(
    a_band: onp.ToComplex64_ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # +c64, select: "v"
def eigvals_banded(
    a_band: onp.ToComplex64_ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # +c64, select: "i"
def eigvals_banded(
    a_band: onp.ToComplex64_ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # +c128
def eigvals_banded(
    a_band: _ToC128ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # +c128, select: "v"
def eigvals_banded(
    a_band: _ToC128ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # +c128, select: "i"
def eigvals_banded(
    a_band: _ToC128ND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # catch-all
def eigvals_banded(
    a_band: onp.ToComplexND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64 | Any]: ...
@overload  # catch-all, select: "v"
def eigvals_banded(
    a_band: onp.ToComplexND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64 | Any]: ...
@overload  # catch-all, select: "i"
def eigvals_banded(
    a_band: onp.ToComplexND,
    lower: bool = False,
    overwrite_a_band: bool = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64 | Any]: ...

#
@overload  # ~bool | ~f16, +f32
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: _ToBoolF16ND,
    e: onp.ToFloat32_ND,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float32]: ...
@overload  # ~bool | ~f16, +f32, select: "v"
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: _ToBoolF16ND,
    e: onp.ToFloat32_ND,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float32]: ...
@overload  # ~bool | ~f16, +f32, select: "i"
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: _ToBoolF16ND,
    e: onp.ToFloat32_ND,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float32]: ...
@overload  # +f32, ~bool | ~f16
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: onp.ToFloat32_ND,
    e: _ToBoolF16ND,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float32]: ...
@overload  # +f32, ~bool | ~f16, select: "v"
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: onp.ToFloat32_ND,
    e: _ToBoolF16ND,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float32]: ...
@overload  # +f32, ~bool | ~f16, select: "i"
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: onp.ToFloat32_ND,
    e: _ToBoolF16ND,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float32]: ...
@overload  # ~bool | ~f16, +f64
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: _ToBoolF16ND,
    e: _ToF64ND,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # ~bool | ~f16, +f64, select: "v"
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: _ToBoolF16ND,
    e: _ToF64ND,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # ~bool | ~f16, +f64, select: "i"
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: _ToBoolF16ND,
    e: _ToF64ND,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +f64, ~bool | ~f16
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: _ToF64ND,
    e: _ToBoolF16ND,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +f64, ~bool | ~f16, select: "v"
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: _ToF64ND,
    e: _ToBoolF16ND,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +f64, ~bool | ~f16, select: "i"
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: _ToF64ND,
    e: _ToBoolF16ND,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # ~f80, +float
@deprecated("longdouble input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: onp.ToJustLongDoubleND,
    e: onp.ToFloatND,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # ~f80, +float, select: "v"
@deprecated("longdouble input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: onp.ToJustLongDoubleND,
    e: onp.ToFloatND,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # ~f80, +float, select: "i"
@deprecated("longdouble input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: onp.ToJustLongDoubleND,
    e: onp.ToFloatND,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +float, ~f80
@deprecated("longdouble input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToJustLongDoubleND,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +float, ~f80, select: "v"
@deprecated("longdouble input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToJustLongDoubleND,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +float, ~f80, select: "i"
@deprecated("longdouble input will no longer be supported in SciPy 2.1")
def eigvalsh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToJustLongDoubleND,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +f32, +f32
def eigvalsh_tridiagonal(
    d: onp.ToFloat32_ND,
    e: onp.ToFloat32_ND,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float32]: ...
@overload  # +f32, +f32, select: "v"
def eigvalsh_tridiagonal(
    d: onp.ToFloat32_ND,
    e: onp.ToFloat32_ND,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float32]: ...
@overload  # +f32, +f32, select: "i"
def eigvalsh_tridiagonal(
    d: onp.ToFloat32_ND,
    e: onp.ToFloat32_ND,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float32]: ...
@overload  # +f64, +float
def eigvalsh_tridiagonal(
    d: _ToF64ND,
    e: onp.ToFloatND,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +f64, +float, select: "v"
def eigvalsh_tridiagonal(
    d: _ToF64ND,
    e: onp.ToFloatND,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +f64, +float, select: "i"
def eigvalsh_tridiagonal(
    d: _ToF64ND,
    e: onp.ToFloatND,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +float, +f64
def eigvalsh_tridiagonal(
    d: onp.ToFloatND,
    e: _ToF64ND,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +float, +f64, select: "v"
def eigvalsh_tridiagonal(
    d: onp.ToFloatND,
    e: _ToF64ND,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # +float, +f64, select: "i"
def eigvalsh_tridiagonal(
    d: onp.ToFloatND,
    e: _ToF64ND,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64]: ...
@overload  # catch-all
def eigvalsh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64 | Any]: ...
@overload  # catch-all, select: "v"
def eigvalsh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64 | Any]: ...
@overload  # catch-all, select: "i"
def eigvalsh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> onp.ArrayND[np.float64 | Any]: ...

#
@overload  # eigvals_only: False = ..., select: _SelectA = ...
def eigh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    eigvals_only: Literal[False] = False,
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_FloatND, _FloatND]: ...
@overload  # eigvals_only: False, select: _SelectV (positional)
def eigh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    eigvals_only: Literal[False],
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_FloatND, _FloatND]: ...
@overload  # eigvals_only: False = ..., select: _SelectV (keyword)
def eigh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    eigvals_only: Literal[False] = False,
    *,
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_FloatND, _FloatND]: ...
@overload  # eigvals_only: False, select: _SelectI (positional)
def eigh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    eigvals_only: Literal[False],
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_FloatND, _FloatND]: ...
@overload  # eigvals_only: False = ..., select: _SelectI (keyword)
def eigh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    eigvals_only: Literal[False] = False,
    *,
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> tuple[_FloatND, _FloatND]: ...
@overload  # eigvals_only: True, select: _SelectA = ...
def eigh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    eigvals_only: Literal[True],
    select: _SelectA = "a",
    select_range: _SelectRange | None = None,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _FloatND: ...
@overload  # eigvals_only: True, select: _SelectV
def eigh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    eigvals_only: Literal[True],
    select: _SelectV,
    select_range: _SelectRange,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _FloatND: ...
@overload  # eigvals_only: True, select: _SelectI
def eigh_tridiagonal(
    d: onp.ToFloatND,
    e: onp.ToFloatND,
    eigvals_only: Literal[True],
    select: _SelectI,
    select_range: _SelectRangeI,
    check_finite: bool = True,
    tol: onp.ToFloat = 0.0,
    lapack_driver: _DriverSTE | _DriverAuto = "auto",
) -> _FloatND: ...

#
@overload  # ~bool | ~f16, calc_q: False = ...
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def hessenberg(
    a: _ToBoolF16ND, calc_q: Literal[False] = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float32]: ...
@overload  # ~bool | ~f16, calc_q: True
@deprecated("bool and float16 input will no longer be supported in SciPy 2.1")
def hessenberg(
    a: _ToBoolF16ND, calc_q: Literal[True], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]]: ...
@overload  # ~f80
@deprecated("longdouble input will no longer be supported in SciPy 2.1")
def hessenberg(
    a: onp.ToJustLongDoubleND, calc_q: Literal[False] = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float64]: ...
@overload  # ~f80, calc_q: True
@deprecated("longdouble input will no longer be supported in SciPy 2.1")
def hessenberg(
    a: onp.ToJustLongDoubleND, calc_q: Literal[True], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...
@overload  # ~c160
@deprecated("clongdouble input will no longer be supported in SciPy 2.1")
def hessenberg(
    a: onp.ToJustCLongDoubleND, calc_q: Literal[False] = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex128]: ...
@overload  # ~c160, calc_q: True
@deprecated("clongdouble input will no longer be supported in SciPy 2.1")
def hessenberg(
    a: onp.ToJustCLongDoubleND, calc_q: Literal[True], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]]: ...
@overload  # +f32
def hessenberg(
    a: onp.ToFloat32_ND, calc_q: Literal[False] = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float32]: ...
@overload  # +f32, calc_q: True
def hessenberg(
    a: onp.ToFloat32_ND, calc_q: Literal[True], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]]: ...
@overload  # +f64
def hessenberg(
    a: _ToF64ND, calc_q: Literal[False] = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float64]: ...
@overload  # +f64, calc_q: True
def hessenberg(
    a: _ToF64ND, calc_q: Literal[True], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...
@overload  # ~c64
def hessenberg(
    a: onp.ToJustComplex64_ND, calc_q: Literal[False] = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex64]: ...
@overload  # ~c64, calc_q: True
def hessenberg(
    a: onp.ToJustComplex64_ND, calc_q: Literal[True], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64]]: ...
@overload  # ~c128
def hessenberg(
    a: onp.ToJustComplex128_ND, calc_q: Literal[False] = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex128]: ...
@overload  # ~c128, calc_q: True
def hessenberg(
    a: onp.ToJustComplex128_ND, calc_q: Literal[True], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]]: ...
@overload  # catch-all
def hessenberg(
    a: onp.ToComplexND, calc_q: Literal[False] = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float64 | Any]: ...
@overload  # catch-all, calc_q: True
def hessenberg(
    a: onp.ToComplexND, calc_q: Literal[True], overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any]]: ...
@overload  # catch-all, calc_q: bool
def hessenberg(
    a: onp.ToComplexND, calc_q: bool, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float64 | Any] | tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any]]: ...

#
@overload
def cdf2rdf[FloatVT: npc.floating, FloatWT: npc.floating](
    w: nptc.CanArray[Any, np.dtype[FloatVT]], v: nptc.CanArray[Any, np.dtype[FloatWT]]
) -> tuple[onp.ArrayND[FloatVT], onp.ArrayND[FloatWT]]: ...
@overload
def cdf2rdf[FloatT: npc.floating](
    w: nptc.CanArray[Any, np.dtype[FloatT]], v: onp.ToComplexND
) -> tuple[onp.ArrayND[FloatT], _FloatND]: ...
@overload
def cdf2rdf[FloatT: npc.floating](
    w: onp.ToComplexND, v: nptc.CanArray[Any, np.dtype[FloatT]]
) -> tuple[_FloatND, onp.ArrayND[FloatT]]: ...
@overload
def cdf2rdf(w: onp.ToComplexND, v: onp.ToComplexND) -> tuple[_FloatND, _FloatND]: ...
