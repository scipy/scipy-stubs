from collections.abc import MutableSequence, Sequence
from typing import Any, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import numpy_typing_compat as nptc
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["cho_factor", "cho_solve", "cho_solve_banded", "cholesky", "cholesky_banded"]

###

_T = TypeVar("_T")
_Shape2T = TypeVar("_Shape2T", bound=tuple[int, int, *tuple[int, ...]])

_as_f32: TypeAlias = np.float32 | np.float16 | npc.integer16 | npc.integer8 | np.bool_  # noqa: PYI042
_as_f64: TypeAlias = npc.floating64 | npc.floating80 | npc.integer64 | npc.integer32  # noqa: PYI042
_as_c64: TypeAlias = np.complex64  # noqa: PYI042
_as_c128: TypeAlias = npc.complexfloating160 | npc.complexfloating128  # noqa: PYI042

_Sequence2D: TypeAlias = Sequence[Sequence[_T]]

###

# NOTE: The ignored `overload-overlap` mypy errors are false positives

# keep in sync with `cholesky_banded` and `cho_factor`
@overload  # Nd +f64
def cholesky(  # type: ignore[overload-overlap]
    a: nptc.CanArray[_Shape2T, np.dtype[_as_f64]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float64, _Shape2T]: ...
@overload  # Nd +f32
def cholesky(
    a: nptc.CanArray[_Shape2T, np.dtype[_as_f32]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float32, _Shape2T]: ...
@overload  # Nd +c128
def cholesky(
    a: nptc.CanArray[_Shape2T, np.dtype[_as_c128]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex128, _Shape2T]: ...
@overload  # Nd ~c64
def cholesky(
    a: nptc.CanArray[_Shape2T, np.dtype[_as_c64]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex64, _Shape2T]: ...
@overload  # Nd ~number
def cholesky(
    a: nptc.CanArray[_Shape2T, np.dtype[npc.number]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[Any, _Shape2T]: ...
@overload  # 2d +f64
def cholesky(
    a: _Sequence2D[float], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~c128
def cholesky(
    a: Sequence[MutableSequence[complex]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.Array2D[np.complex128]: ...
@overload  # ?d +f64
def cholesky(
    a: onp.SequenceND[_Sequence2D[float]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float64]: ...
@overload  # ?d ~c128
def cholesky(
    a: onp.SequenceND[Sequence[MutableSequence[complex]]],
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...

# keep in sync with `cholesky` (but swap `lower` and `overwrite_*`)
@overload  # Nd +f64
def cholesky_banded(  # type: ignore[overload-overlap]
    ab: nptc.CanArray[_Shape2T, np.dtype[_as_f64]], overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float64, _Shape2T]: ...
@overload  # Nd +f32
def cholesky_banded(
    ab: nptc.CanArray[_Shape2T, np.dtype[_as_f32]], overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float32, _Shape2T]: ...
@overload  # Nd +c128
def cholesky_banded(
    ab: nptc.CanArray[_Shape2T, np.dtype[_as_c128]], overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex128, _Shape2T]: ...
@overload  # Nd ~c64
def cholesky_banded(
    ab: nptc.CanArray[_Shape2T, np.dtype[_as_c64]], overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex64, _Shape2T]: ...
@overload  # Nd ~number
def cholesky_banded(
    ab: nptc.CanArray[_Shape2T, np.dtype[npc.number]], overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> onp.ArrayND[Any, _Shape2T]: ...
@overload  # 2d +f64
def cholesky_banded(
    ab: _Sequence2D[float], overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> onp.Array2D[np.float64]: ...
@overload  # 2d ~c128
def cholesky_banded(
    ab: Sequence[MutableSequence[complex]], overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> onp.Array2D[np.complex128]: ...
@overload  # ?d +f64
def cholesky_banded(
    ab: onp.SequenceND[_Sequence2D[float]], overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float64]: ...
@overload  # ?d ~c128
def cholesky_banded(
    ab: onp.SequenceND[Sequence[MutableSequence[complex]]],
    overwrite_ab: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...

# keep in sync with `cholesky`
@overload  # Nd +f64
def cho_factor(  # type: ignore[overload-overlap]
    a: nptc.CanArray[_Shape2T, np.dtype[_as_f64]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float64, _Shape2T], bool]: ...
@overload  # Nd +f32
def cho_factor(
    a: nptc.CanArray[_Shape2T, np.dtype[_as_f32]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float32, _Shape2T], bool]: ...
@overload  # Nd +c128
def cho_factor(
    a: nptc.CanArray[_Shape2T, np.dtype[_as_c128]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex128, _Shape2T], bool]: ...
@overload  # Nd ~c64
def cho_factor(
    a: nptc.CanArray[_Shape2T, np.dtype[_as_c64]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex64, _Shape2T], bool]: ...
@overload  # Nd ~number
def cho_factor(
    a: nptc.CanArray[_Shape2T, np.dtype[npc.number]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[Any, _Shape2T], bool]: ...
@overload  # 2d +f64
def cho_factor(
    a: _Sequence2D[float], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.Array2D[np.float64], bool]: ...
@overload  # 2d ~c128
def cho_factor(
    a: Sequence[MutableSequence[complex]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.Array2D[np.complex128], bool]: ...
@overload  # ?d +f64
def cho_factor(
    a: onp.SequenceND[_Sequence2D[float]], lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float64], bool]: ...
@overload  # ?d ~c128
def cho_factor(
    a: onp.SequenceND[Sequence[MutableSequence[complex]]],
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.complex128], bool]: ...

# keep in sync with `cho_solve_banded` and `lu_solve` in `_decomp_lu`
@overload  # ?d +f64\+f32, ?d +f64
def cho_solve(  # type: ignore[overload-overlap]
    c_and_lower: tuple[onp.ToArrayND[float, _as_f64], bool],
    b: onp.ToFloatND,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # ?d +f64, ?d +f64\+f32
def cho_solve(  # type: ignore[overload-overlap]
    c_and_lower: tuple[onp.ToFloatND, bool],
    b: onp.ToArrayND[float, _as_f64],
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # ?d +f32, ?d +f32
def cho_solve(
    c_and_lower: tuple[onp.ToFloat32_ND, bool], b: onp.ToFloat32_ND, overwrite_b: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float32]: ...
@overload  # ?d ~c128|c160, ?d +c128
def cho_solve(
    c_and_lower: tuple[onp.ToJustComplex128_ND | onp.ToJustCLongDoubleND, bool],
    b: onp.ToComplexND,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # ?d +c128, ?d ~c128|c160
def cho_solve(
    c_and_lower: tuple[onp.ToComplexND, bool],
    b: onp.ToJustComplex128_ND | onp.ToJustCLongDoubleND,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # ?d ~c64, ?d +c64
def cho_solve(
    c_and_lower: tuple[onp.ToJustComplex64_ND, bool], b: onp.ToComplex64_ND, overwrite_b: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex64]: ...
@overload  # ?d +c64, ?d ~c64
def cho_solve(
    c_and_lower: tuple[onp.ToComplex64_ND, bool], b: onp.ToJustComplex64_ND, overwrite_b: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex64]: ...
@overload  # ?d +cfloating, ?d ~cfloating (fallback)
def cho_solve(
    c_and_lower: tuple[onp.ToComplexND, bool], b: onp.ToComplexND, overwrite_b: bool = False, check_finite: bool = True
) -> onp.ArrayND[Any]: ...

# keep in sync with `cho_solve`
@overload  # ?d +f64\+f32, ?d +f64
def cho_solve_banded(  # type: ignore[overload-overlap]
    cb_and_lower: tuple[onp.ToArrayND[float, _as_f64], bool],
    b: onp.ToFloatND,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # ?d +f64, ?d +f64\+f32
def cho_solve_banded(  # type: ignore[overload-overlap]
    cb_and_lower: tuple[onp.ToFloatND, bool],
    b: onp.ToArrayND[float, _as_f64],
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # ?d +f32, ?d +f32
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToFloat32_ND, bool], b: onp.ToFloat32_ND, overwrite_b: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.float32]: ...
@overload  # ?d ~c128|c160, ?d +c128
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToJustComplex128_ND | onp.ToJustCLongDoubleND, bool],
    b: onp.ToComplexND,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # ?d +c128, ?d ~c128|c160
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToComplexND, bool],
    b: onp.ToJustComplex128_ND | onp.ToJustCLongDoubleND,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # ?d ~c64, ?d +c64
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToJustComplex64_ND, bool], b: onp.ToComplex64_ND, overwrite_b: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex64]: ...
@overload  # ?d +c64, ?d ~c64
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToComplex64_ND, bool], b: onp.ToJustComplex64_ND, overwrite_b: bool = False, check_finite: bool = True
) -> onp.ArrayND[np.complex64]: ...
@overload  # ?d +cfloating, ?d ~cfloating (fallback)
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToComplexND, bool], b: onp.ToComplexND, overwrite_b: bool = False, check_finite: bool = True
) -> onp.ArrayND[Any]: ...
