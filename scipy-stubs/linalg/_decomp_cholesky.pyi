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

_Float2D: TypeAlias = onp.Array2D[npc.floating]
_FloatND: TypeAlias = onp.ArrayND[npc.floating]
_Complex2D: TypeAlias = onp.Array2D[npc.inexact]
_ComplexND: TypeAlias = onp.ArrayND[npc.inexact]

_Sequence2D: TypeAlias = Sequence[Sequence[_T]]

###

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

#
@overload
def cho_factor(
    a: onp.ToFloatND, lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[_FloatND, bool]: ...
@overload
def cho_factor(
    a: onp.ToComplexND, lower: bool = False, overwrite_a: bool = False, check_finite: bool = True
) -> tuple[_ComplexND, bool]: ...

#
@overload
def cho_solve(
    c_and_lower: tuple[onp.ToFloatStrict2D, bool], b: onp.ToFloatStrict1D, overwrite_b: bool = False, check_finite: bool = True
) -> _Float2D: ...
@overload
def cho_solve(
    c_and_lower: tuple[onp.ToFloatND, bool], b: onp.ToFloatND, overwrite_b: bool = False, check_finite: bool = True
) -> _FloatND: ...
@overload
def cho_solve(
    c_and_lower: tuple[onp.ToComplexStrict2D, bool],
    b: onp.ToComplexStrict1D,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Complex2D: ...
@overload
def cho_solve(
    c_and_lower: tuple[onp.ToComplexND, bool], b: onp.ToComplexND, overwrite_b: bool = False, check_finite: bool = True
) -> _ComplexND: ...

#
@overload
def cholesky_banded(
    ab: onp.ToFloatStrict2D, overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> _Float2D: ...
@overload
def cholesky_banded(
    ab: onp.ToFloatND, overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> _FloatND: ...
@overload
def cholesky_banded(
    ab: onp.ToComplexStrict2D, overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> _Complex2D: ...
@overload
def cholesky_banded(
    ab: onp.ToComplexND, overwrite_ab: bool = False, lower: bool = False, check_finite: bool = True
) -> _ComplexND: ...

#
@overload
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToFloatStrict2D, bool], b: onp.ToComplexStrict1D, overwrite_b: bool = False, check_finite: bool = True
) -> _Complex2D: ...
@overload
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToFloatND, bool], b: onp.ToComplexND, overwrite_b: bool = False, check_finite: bool = True
) -> _ComplexND: ...
@overload
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToComplexStrict2D, bool],
    b: onp.ToComplexStrict1D,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> _Complex2D: ...
@overload
def cho_solve_banded(
    cb_and_lower: tuple[onp.ToComplexND, bool], b: onp.ToComplexND, overwrite_b: bool = False, check_finite: bool = True
) -> _ComplexND: ...
