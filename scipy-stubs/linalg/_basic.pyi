# mypy: disable-error-code=overload-overlap

from collections.abc import Sequence
from typing import Any, Final, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = [
    "det",
    "inv",
    "lstsq",
    "matmul_toeplitz",
    "matrix_balance",
    "pinv",
    "pinvh",
    "solve",
    "solve_banded",
    "solve_circulant",
    "solve_toeplitz",
    "solve_triangular",
    "solveh_banded",
]

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]
_COrCR: TypeAlias = _T | _Tuple2[_T]

_Float: TypeAlias = npc.floating
_Float0D: TypeAlias = onp.Array0D[_Float]
_Float1D: TypeAlias = onp.Array1D[_Float]
_Float2D: TypeAlias = onp.Array2D[_Float]
_FloatND: TypeAlias = onp.ArrayND[_Float]

_Complex: TypeAlias = npc.inexact  # float and complex input types are near impossible to distinguish
_Complex0D: TypeAlias = onp.Array0D[_Complex]
_Complex1D: TypeAlias = onp.Array1D[_Complex]
_Complex2D: TypeAlias = onp.Array2D[_Complex]
_ComplexND: TypeAlias = onp.ArrayND[_Complex]

_InputFloat: TypeAlias = onp.ToArrayND[float, np.float64 | np.longdouble | npc.integer | np.bool_]
_InputFloatStrict1D: TypeAlias = onp.ToArrayStrict1D[float, np.float64 | np.longdouble | npc.integer | np.bool_]
_InputFloatStrict2D: TypeAlias = onp.ToArrayStrict2D[float, np.float64 | np.longdouble | npc.integer | np.bool_]

_InputF64: TypeAlias = onp.ToArrayND[float, np.float64 | npc.integer | np.bool_]
_InputF64Strict1D: TypeAlias = onp.ToArrayStrict1D[float, np.float64 | npc.integer | np.bool_]
_InputF64Strict2D: TypeAlias = onp.ToArrayStrict2D[float, np.float64 | npc.integer | np.bool_]

_AsF32: TypeAlias = np.float16 | np.float32
_InputF32: TypeAlias = onp.CanArrayND[_AsF32] | onp.SequenceND[onp.CanArray[Any, np.dtype[_AsF32]]]
_InputF32Strict1D: TypeAlias = onp.CanArray1D[_AsF32] | Sequence[onp.CanArray0D[_AsF32]]
_InputF32Strict2D: TypeAlias = onp.CanArray2D[_AsF32] | Sequence[_InputF32Strict1D]

_InputC64: TypeAlias = onp.CanArrayND[np.complex64] | onp.SequenceND[onp.CanArray[Any, np.dtype[np.complex64]]]
_InputC64Strict1D: TypeAlias = onp.CanArray1D[np.complex64] | Sequence[onp.CanArray0D[np.complex64]]
_InputC64Strict2D: TypeAlias = onp.CanArray2D[np.complex64] | Sequence[_InputC64Strict1D]

_InputComplex: TypeAlias = onp.ToArrayND[op.JustComplex, np.complex128 | np.clongdouble]
_InputComplexStrict1D: TypeAlias = onp.ToArrayStrict1D[op.JustComplex, np.complex128 | np.clongdouble]
_InputComplexStrict2D: TypeAlias = onp.ToArrayStrict2D[op.JustComplex, np.complex128 | np.clongdouble]

_CoC64: TypeAlias = np.complex64 | _AsF32 | npc.integer16 | npc.integer8 | np.bool_
_CoInputC64: TypeAlias = onp.CanArrayND[_CoC64] | onp.SequenceND[onp.CanArray[Any, np.dtype[_CoC64]]]
_CoInputC64Strict1D: TypeAlias = onp.CanArray1D[_CoC64] | Sequence[onp.CanArray0D[_CoC64]]
_CoInputC64Strict2D: TypeAlias = onp.CanArray2D[_CoC64] | Sequence[_CoInputC64Strict1D]

_AssumeA: TypeAlias = Literal[
    "diagonal",
    "tridiagonal",
    "banded",
    "upper triangular",
    "lower triangular",
    "symmetric", "sym",
    "hermitian", "her",
    "positive definite", "pos",
    "general", "gen",
]  # fmt: skip
_TransSystem: TypeAlias = Literal[0, "N", 1, "T", 2, "C"]
_Singular: TypeAlias = Literal["lstsq", "raise"]
_LapackDriver: TypeAlias = Literal["gelsd", "gelsy", "gelss"]

###

lapack_cast_dict: Final[dict[str, str]] = ...

@overload  # 2d ~float32, ~float32
def solve(
    a: _InputF32Strict2D,
    b: _InputF32Strict1D | _InputF32Strict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.float32]: ...
@overload  # Nd ~float32, ~float32
def solve(
    a: _InputF32,
    b: _InputF32,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.float32]: ...
@overload  # 2D ~float64, +float64
def solve(
    a: _InputFloatStrict2D,
    b: onp.ToFloatStrict1D | onp.ToFloatStrict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.float64]: ...
@overload  # Nd ~float64, +float64
def solve(
    a: _InputFloat,
    b: onp.ToFloatND,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload  # 2d +float64, ~float64
def solve(
    a: onp.ToFloatStrict2D,
    b: _InputFloatStrict1D | _InputFloatStrict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.float64]: ...
@overload  # Nd +float64, ~float64
def solve(
    a: onp.ToFloatND,
    b: _InputFloat,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.float64]: ...
@overload  # 2d ~complex64, +complex64
def solve(
    a: _InputC64Strict2D,
    b: _CoInputC64Strict1D | _CoInputC64Strict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd ~complex64, +complex64
def solve(
    a: _InputC64,
    b: _CoInputC64,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.complex64]: ...
@overload  # 2d +complex64, ~complex64
def solve(
    a: _CoInputC64Strict2D,
    b: _InputC64Strict1D | _InputC64Strict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd +complex64, ~complex64
def solve(
    a: _CoInputC64,
    b: _InputC64,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.complex64]: ...
@overload  # 2d ~complex128, +complex128
def solve(
    a: _InputComplexStrict2D,
    b: onp.ToComplexStrict1D | onp.ToComplexStrict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd ~complex128, +complex128
def solve(
    a: _InputComplex,
    b: onp.ToComplexND,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 2d +complex128, ~complex128
def solve(
    a: onp.ToComplexStrict2D,
    b: _InputComplexStrict1D | _InputComplexStrict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd +complex128, ~complex128
def solve(
    a: onp.ToComplexND,
    b: _InputComplex,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 2d +floating, +floating
def solve(
    a: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict1D | onp.ToFloatStrict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.float32 | np.float64]: ...
@overload  # Nd +floating, +floating
def solve(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.float32 | np.float64]: ...
@overload  # 2d ~complexfloating, +complexfloating
def solve(
    a: onp.ToJustComplexStrict2D,
    b: onp.ToComplexStrict1D | onp.ToComplexStrict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.complex64 | np.complex128]: ...
@overload  # Nd ~complexfloating, +complexfloating
def solve(
    a: onp.ToJustComplexND,
    b: onp.ToComplexND,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.complex64 | np.complex128]: ...
@overload  # 2d +complexfloating, ~complexfloating
def solve(
    a: onp.ToComplexStrict2D,
    b: onp.ToJustComplexStrict1D | onp.ToJustComplexStrict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.complex64 | np.complex128]: ...
@overload  # Nd +complexfloating, ~complexfloating
def solve(
    a: onp.ToComplexND,
    b: onp.ToJustComplexND,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.complex64 | np.complex128]: ...
@overload  # 2d +complexfloating, +complexfloating
def solve(
    a: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict1D | onp.ToComplexStrict2D,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.Array2D[np.float32 | np.float64 | np.complex64 | np.complex128]: ...
@overload  # Nd +complexfloating, +complexfloating
def solve(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: _AssumeA | None = None,
    transposed: bool = False,
) -> onp.ArrayND[np.float32 | np.float64 | np.complex64 | np.complex128]: ...

#
@overload  # 1d ~float32, ~float32
def solve_triangular(
    a: _InputF32Strict2D,
    b: _InputF32Strict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float32]: ...
@overload  # 2d ~float32, ~float32
def solve_triangular(
    a: _InputF32Strict2D,
    b: _InputF32Strict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float32]: ...
@overload  # Nd ~float32, ~float32
def solve_triangular(
    a: _InputF32,
    b: _InputF32,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # 1D ~float64, +float64
def solve_triangular(
    a: _InputFloatStrict2D,
    b: onp.ToFloatStrict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float64]: ...
@overload  # 2D ~float64, +float64
def solve_triangular(
    a: _InputFloatStrict2D,
    b: onp.ToFloatStrict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float64]: ...
@overload  # Nd ~float64, +float64
def solve_triangular(
    a: _InputFloat,
    b: onp.ToFloatND,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # 1d +float64, ~float64
def solve_triangular(
    a: onp.ToFloatStrict2D,
    b: _InputFloatStrict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float64]: ...
@overload  # 2d +float64, ~float64
def solve_triangular(
    a: onp.ToFloatStrict2D,
    b: _InputFloatStrict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float64]: ...
@overload  # Nd +float64, ~float64
def solve_triangular(
    a: onp.ToFloatND,
    b: _InputFloat,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # 1d ~complex64, +complex64
def solve_triangular(
    a: _InputC64Strict2D,
    b: _CoInputC64Strict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64]: ...
@overload  # 2d ~complex64, +complex64
def solve_triangular(
    a: _InputC64Strict2D,
    b: _CoInputC64Strict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd ~complex64, +complex64
def solve_triangular(
    a: _InputC64,
    b: _CoInputC64,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64]: ...
@overload  # 1d +complex64, ~complex64
def solve_triangular(
    a: _CoInputC64Strict2D,
    b: _InputC64Strict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64]: ...
@overload  # 2d +complex64, ~complex64
def solve_triangular(
    a: _CoInputC64Strict2D,
    b: _InputC64Strict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd +complex64, ~complex64
def solve_triangular(
    a: _CoInputC64,
    b: _InputC64,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64]: ...
@overload  # 1d ~complex128, +complex128
def solve_triangular(
    a: _InputComplexStrict2D,
    b: onp.ToComplexStrict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d ~complex128, +complex128
def solve_triangular(
    a: _InputComplexStrict2D,
    b: onp.ToComplexStrict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd ~complex128, +complex128
def solve_triangular(
    a: _InputComplex,
    b: onp.ToComplexND,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 1d +complex128, ~complex128
def solve_triangular(
    a: onp.ToComplexStrict2D,
    b: _InputComplexStrict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d +complex128, ~complex128
def solve_triangular(
    a: onp.ToComplexStrict2D,
    b: _InputComplexStrict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd +complex128, ~complex128
def solve_triangular(
    a: onp.ToComplexND,
    b: _InputComplex,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 1d +floating, +floating
def solve_triangular(
    a: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float32 | np.float64]: ...
@overload  # 2d +floating, +floating
def solve_triangular(
    a: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float32 | np.float64]: ...
@overload  # Nd +floating, +floating
def solve_triangular(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32 | np.float64]: ...
@overload  # 1d ~complexfloating, +complexfloating
def solve_triangular(
    a: onp.ToJustComplexStrict2D,
    b: onp.ToComplexStrict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64 | np.complex128]: ...
@overload  # 2d ~complexfloating, +complexfloating
def solve_triangular(
    a: onp.ToJustComplexStrict2D,
    b: onp.ToComplexStrict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64 | np.complex128]: ...
@overload  # Nd ~complexfloating, +complexfloating
def solve_triangular(
    a: onp.ToJustComplexND,
    b: onp.ToComplexND,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64 | np.complex128]: ...
@overload  # 1d +complexfloating, ~complexfloating
def solve_triangular(
    a: onp.ToComplexStrict2D,
    b: onp.ToJustComplexStrict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64 | np.complex128]: ...
@overload  # 2d +complexfloating, ~complexfloating
def solve_triangular(
    a: onp.ToComplexStrict2D,
    b: onp.ToJustComplexStrict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64 | np.complex128]: ...
@overload  # Nd +complexfloating, ~complexfloating
def solve_triangular(
    a: onp.ToComplexND,
    b: onp.ToJustComplexND,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64 | np.complex128]: ...
@overload  # 1d +complexfloating, +complexfloating
def solve_triangular(
    a: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict1D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float32 | np.float64 | np.complex64 | np.complex128]: ...
@overload  # 2d +complexfloating, +complexfloating
def solve_triangular(
    a: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict2D,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float32 | np.float64 | np.complex64 | np.complex128]: ...
@overload  # Nd +complexfloating, +complexfloating
def solve_triangular(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    trans: _TransSystem = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32 | np.float64 | np.complex64 | np.complex128]: ...

#
@overload  # 1d ~float32, ~float32
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputF32Strict2D,
    b: _InputF32Strict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float32]: ...
@overload  # 2d ~float32, ~float32
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputF32Strict2D,
    b: _InputF32Strict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float32]: ...
@overload  # Nd ~float32, ~float32
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputF32,
    b: _InputF32,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # 1D ~float64, +float64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputFloatStrict2D,
    b: onp.ToFloatStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float64]: ...
@overload  # 2D ~float64, +float64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputFloatStrict2D,
    b: onp.ToFloatStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float64]: ...
@overload  # Nd ~float64, +float64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputFloat,
    b: onp.ToFloatND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # 1d +float64, ~float64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToFloatStrict2D,
    b: _InputFloatStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float64]: ...
@overload  # 2d +float64, ~float64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToFloatStrict2D,
    b: _InputFloatStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float64]: ...
@overload  # Nd +float64, ~float64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToFloatND,
    b: _InputFloat,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # 1d ~complex64, +complex64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputC64Strict2D,
    b: _InputC64Strict1D | _InputF32Strict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64]: ...
@overload  # 2d ~complex64, +complex64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputC64Strict2D,
    b: _InputC64Strict2D | _InputF32Strict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd ~complex64, +complex64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputC64,
    b: _InputC64 | _InputF32,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64]: ...
@overload  # 1d +complex64, ~complex64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputC64Strict2D | _InputF32Strict2D,
    b: _InputC64Strict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64]: ...
@overload  # 2d +complex64, ~complex64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputC64Strict2D | _InputF32Strict2D,
    b: _InputC64Strict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd +complex64, ~complex64
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputC64 | _InputF32,
    b: _InputC64,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64]: ...
@overload  # 1d ~complex128, +complex128
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputComplexStrict2D,
    b: onp.ToComplexStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d ~complex128, +complex128
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputComplexStrict2D,
    b: onp.ToComplexStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd ~complex128, +complex128
def solve_banded(
    l_and_u: tuple[int, int],
    ab: _InputComplex,
    b: onp.ToComplexND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 1d +complex128, ~complex128
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToComplexStrict2D,
    b: _InputComplexStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d +complex128, ~complex128
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToComplexStrict2D,
    b: _InputComplexStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd +complex128, ~complex128
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToComplexND,
    b: _InputComplex,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 1d +floating, +floating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float32 | np.float64]: ...
@overload  # 2d +floating, +floating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float32 | np.float64]: ...
@overload  # Nd +floating, +floating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToFloatND,
    b: onp.ToFloatND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32 | np.float64]: ...
@overload  # 1d ~complexfloating, +complexfloating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToJustComplexStrict2D,
    b: onp.ToComplexStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64 | np.complex128]: ...
@overload  # 2d ~complexfloating, +complexfloating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToJustComplexStrict2D,
    b: onp.ToComplexStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64 | np.complex128]: ...
@overload  # Nd ~complexfloating, +complexfloating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToJustComplexND,
    b: onp.ToComplexND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64 | np.complex128]: ...
@overload  # 1d +complexfloating, ~complexfloating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToComplexStrict2D,
    b: onp.ToJustComplexStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64 | np.complex128]: ...
@overload  # 2d +complexfloating, ~complexfloating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToComplexStrict2D,
    b: onp.ToJustComplexStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64 | np.complex128]: ...
@overload  # Nd +complexfloating, ~complexfloating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToComplexND,
    b: onp.ToJustComplexND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64 | np.complex128]: ...
@overload  # 1d +complexfloating, +complexfloating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float32 | np.float64 | np.complex64 | np.complex128]: ...
@overload  # 2d +complexfloating, +complexfloating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float32 | np.float64 | np.complex64 | np.complex128]: ...
@overload  # Nd +complexfloating, +complexfloating
def solve_banded(
    l_and_u: tuple[int, int],
    ab: onp.ToComplexND,
    b: onp.ToComplexND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32 | np.float64 | np.complex64 | np.complex128]: ...

#
@overload  # 1d ~float32, ~float32
def solveh_banded(
    ab: _InputF32Strict2D,
    b: _InputF32Strict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float32]: ...
@overload  # 2d ~float32, ~float32
def solveh_banded(
    ab: _InputF32Strict2D,
    b: _InputF32Strict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float32]: ...
@overload  # Nd ~float32, ~float32
def solveh_banded(
    ab: _InputF32,
    b: _InputF32,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32]: ...
@overload  # 1D ~float64, +float64
def solveh_banded(
    ab: _InputFloatStrict2D,
    b: onp.ToFloatStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float64]: ...
@overload  # 2D ~float64, +float64
def solveh_banded(
    ab: _InputFloatStrict2D,
    b: onp.ToFloatStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float64]: ...
@overload  # Nd ~float64, +float64
def solveh_banded(
    ab: _InputFloat,
    b: onp.ToFloatND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # 1d +float64, ~float64
def solveh_banded(
    ab: onp.ToFloatStrict2D,
    b: _InputFloatStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float64]: ...
@overload  # 2d +float64, ~float64
def solveh_banded(
    ab: onp.ToFloatStrict2D,
    b: _InputFloatStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float64]: ...
@overload  # Nd +float64, ~float64
def solveh_banded(
    ab: onp.ToFloatND,
    b: _InputFloat,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # 1d ~complex64, +complex64
def solveh_banded(
    ab: _InputC64Strict2D,
    b: _InputC64Strict1D | _InputF32Strict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64]: ...
@overload  # 2d ~complex64, +complex64
def solveh_banded(
    ab: _InputC64Strict2D,
    b: _InputC64Strict2D | _InputF32Strict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd ~complex64, +complex64
def solveh_banded(
    ab: _InputC64,
    b: _InputC64 | _InputF32,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64]: ...
@overload  # 1d +complex64, ~complex64
def solveh_banded(
    ab: _InputC64Strict2D | _InputF32Strict2D,
    b: _InputC64Strict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64]: ...
@overload  # 2d +complex64, ~complex64
def solveh_banded(
    ab: _InputC64Strict2D | _InputF32Strict2D,
    b: _InputC64Strict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd +complex64, ~complex64
def solveh_banded(
    ab: _InputC64 | _InputF32,
    b: _InputC64,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64]: ...
@overload  # 1d ~complex128, +complex128
def solveh_banded(
    ab: _InputComplexStrict2D,
    b: onp.ToComplexStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d ~complex128, +complex128
def solveh_banded(
    ab: _InputComplexStrict2D,
    b: onp.ToComplexStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd ~complex128, +complex128
def solveh_banded(
    ab: _InputComplex,
    b: onp.ToComplexND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 1d +complex128, ~complex128
def solveh_banded(
    ab: onp.ToComplexStrict2D,
    b: _InputComplexStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d +complex128, ~complex128
def solveh_banded(
    ab: onp.ToComplexStrict2D,
    b: _InputComplexStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd +complex128, ~complex128
def solveh_banded(
    ab: onp.ToComplexND,
    b: _InputComplex,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 1d +floating, +floating
def solveh_banded(
    ab: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float32 | np.float64]: ...
@overload  # 2d +floating, +floating
def solveh_banded(
    ab: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float32 | np.float64]: ...
@overload  # Nd +floating, +floating
def solveh_banded(
    ab: onp.ToFloatND,
    b: onp.ToFloatND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32 | np.float64]: ...
@overload  # 1d ~complexfloating, +complexfloating
def solveh_banded(
    ab: onp.ToJustComplexStrict2D,
    b: onp.ToComplexStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64 | np.complex128]: ...
@overload  # 2d ~complexfloating, +complexfloating
def solveh_banded(
    ab: onp.ToJustComplexStrict2D,
    b: onp.ToComplexStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64 | np.complex128]: ...
@overload  # Nd ~complexfloating, +complexfloating
def solveh_banded(
    ab: onp.ToJustComplexND,
    b: onp.ToComplexND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64 | np.complex128]: ...
@overload  # 1d +complexfloating, ~complexfloating
def solveh_banded(
    ab: onp.ToComplexStrict2D,
    b: onp.ToJustComplexStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.complex64 | np.complex128]: ...
@overload  # 2d +complexfloating, ~complexfloating
def solveh_banded(
    ab: onp.ToComplexStrict2D,
    b: onp.ToJustComplexStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.complex64 | np.complex128]: ...
@overload  # Nd +complexfloating, ~complexfloating
def solveh_banded(
    ab: onp.ToComplexND,
    b: onp.ToJustComplexND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex64 | np.complex128]: ...
@overload  # 1d +complexfloating, +complexfloating
def solveh_banded(
    ab: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict1D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array1D[np.float32 | np.float64 | np.complex64 | np.complex128]: ...
@overload  # 2d +complexfloating, +complexfloating
def solveh_banded(
    ab: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict2D,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.Array2D[np.float32 | np.float64 | np.complex64 | np.complex128]: ...
@overload  # Nd +complexfloating, +complexfloating
def solveh_banded(
    ab: onp.ToComplexND,
    b: onp.ToComplexND,
    overwrite_ab: bool = False,
    overwrite_b: bool = False,
    lower: bool = False,
    check_finite: bool = True,
) -> onp.ArrayND[np.float32 | np.float64 | np.complex64 | np.complex128]: ...

#
@overload  # 1d +float, +float
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToFloatStrict1D], b: onp.ToFloatStrict1D, check_finite: bool = True
) -> onp.Array1D[np.float64]: ...
@overload  # 2d +float, +float
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToFloatStrict1D], b: onp.ToFloatStrict2D, check_finite: bool = True
) -> onp.Array2D[np.float64]: ...
@overload  # Nd +float, +float
def solve_toeplitz(c_or_cr: _COrCR[onp.ToFloatND], b: onp.ToFloatND, check_finite: bool = True) -> onp.ArrayND[np.float64]: ...
@overload  # 1d ~complex, +complex
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToJustComplexStrict1D], b: onp.ToComplexStrict1D, check_finite: bool = True
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d ~complex, +complex
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToJustComplexStrict1D], b: onp.ToComplexStrict2D, check_finite: bool = True
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd ~complex, +complex
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToJustComplexND], b: onp.ToComplexND, check_finite: bool = True
) -> onp.ArrayND[np.complex128]: ...
@overload  # 1d +complex, ~complex
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToComplexStrict1D], b: onp.ToJustComplexStrict1D, check_finite: bool = True
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d +complex, ~complex
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToComplexStrict1D], b: onp.ToJustComplexStrict2D, check_finite: bool = True
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd +complex, ~complex
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToComplexND], b: onp.ToJustComplexND, check_finite: bool = True
) -> onp.ArrayND[np.complex128]: ...
@overload  # 1d +complex, +complex
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToComplexStrict1D], b: onp.ToComplexStrict1D, check_finite: bool = True
) -> onp.Array1D[np.float64 | np.complex128]: ...
@overload  # 2d +complex, +complex
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToComplexStrict1D], b: onp.ToComplexStrict2D, check_finite: bool = True
) -> onp.Array2D[np.float64 | np.complex128]: ...
@overload  # Nd +complex, +complex
def solve_toeplitz(
    c_or_cr: _COrCR[onp.ToComplexND], b: onp.ToComplexND, check_finite: bool = True
) -> onp.ArrayND[np.float64 | np.complex128]: ...

#
@overload  # 1d ~float32, ~float32
def solve_circulant(
    c: _InputF32Strict1D,
    b: _InputF32Strict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[np.float32]: ...
@overload  # 2d ~float32, ~float32
def solve_circulant(
    c: _InputF32Strict1D,
    b: _InputF32Strict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[np.float32]: ...
@overload  # Nd ~float32, ~float32
def solve_circulant(
    c: _InputF32,
    b: _InputF32,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[np.float32]: ...
@overload  # 1D ~float64, +float64
def solve_circulant(
    c: _InputF64Strict1D,
    b: onp.ToFloat64Strict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[np.float64]: ...
@overload  # 2D ~float64, +float64
def solve_circulant(
    c: _InputF64Strict1D,
    b: onp.ToFloat64Strict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[np.float64]: ...
@overload  # Nd ~float64, +float64
def solve_circulant(
    c: _InputF64,
    b: onp.ToFloat64_ND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[np.float64]: ...
@overload  # 1d +float64, ~float64
def solve_circulant(
    c: onp.ToFloat64Strict1D,
    b: _InputF64Strict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[np.float64]: ...
@overload  # 2d +float64, ~float64
def solve_circulant(
    c: onp.ToFloat64Strict1D,
    b: _InputF64Strict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[np.float64]: ...
@overload  # Nd +float64, ~float64
def solve_circulant(
    c: onp.ToFloatND,
    b: _InputF64,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[np.float64]: ...
@overload  # 1d ~complex64, +complex64
def solve_circulant(
    c: _InputC64Strict1D,
    b: _InputC64Strict1D | _InputF32Strict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[np.complex64]: ...
@overload  # 2d ~complex64, +complex64
def solve_circulant(
    c: _InputC64Strict1D,
    b: _CoInputC64Strict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd ~complex64, +complex64
def solve_circulant(
    c: _InputC64,
    b: _CoInputC64,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[np.complex64]: ...
@overload  # 1d +complex64, ~complex64
def solve_circulant(
    c: _CoInputC64Strict1D,
    b: _InputC64Strict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[np.complex64]: ...
@overload  # 2d +complex64, ~complex64
def solve_circulant(
    c: _CoInputC64Strict1D,
    b: _InputC64Strict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[np.complex64]: ...
@overload  # Nd +complex64, ~complex64
def solve_circulant(
    c: _CoInputC64,
    b: _InputC64,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[np.complex64]: ...
@overload  # 1d ~complex128, +complex128
def solve_circulant(
    c: onp.ToJustComplex128Strict1D,
    b: onp.ToComplex128Strict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d ~complex128, +complex128
def solve_circulant(
    c: onp.ToJustComplex128Strict1D,
    b: onp.ToComplex128Strict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd ~complex128, +complex128
def solve_circulant(
    c: onp.ToJustComplex128_ND,
    b: onp.ToComplex128_ND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 1d +complex128, ~complex128
def solve_circulant(
    c: onp.ToComplex128Strict1D,
    b: onp.ToJustComplex128Strict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[np.complex128]: ...
@overload  # 2d +complex128, ~complex128
def solve_circulant(
    c: onp.ToComplex128Strict1D,
    b: onp.ToJustComplex128Strict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[np.complex128]: ...
@overload  # Nd +complex128, ~complex128
def solve_circulant(
    c: onp.ToComplex128_ND,
    b: onp.ToJustComplex128_ND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[np.complex128]: ...
@overload  # 1d +floating, +floating
def solve_circulant(
    c: onp.ToFloatStrict1D,
    b: onp.ToFloatStrict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[npc.floating]: ...
@overload  # 2d +floating, +floating
def solve_circulant(
    c: onp.ToFloatStrict1D,
    b: onp.ToFloatStrict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[npc.floating]: ...
@overload  # Nd +floating, +floating
def solve_circulant(
    c: onp.ToFloatND,
    b: onp.ToFloatND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[npc.floating]: ...
@overload  # 1d ~complexfloating, +complexfloating
def solve_circulant(
    c: onp.ToJustComplexStrict1D,
    b: onp.ToComplexStrict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[npc.complexfloating]: ...
@overload  # 2d ~complexfloating, +complexfloating
def solve_circulant(
    c: onp.ToJustComplexStrict1D,
    b: onp.ToComplexStrict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[npc.complexfloating]: ...
@overload  # Nd ~complexfloating, +complexfloating
def solve_circulant(
    c: onp.ToJustComplexND,
    b: onp.ToComplexND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[npc.complexfloating]: ...
@overload  # 1d +complexfloating, ~complexfloating
def solve_circulant(
    c: onp.ToComplexStrict1D,
    b: onp.ToJustComplexStrict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[npc.complexfloating]: ...
@overload  # 2d +complexfloating, ~complexfloating
def solve_circulant(
    c: onp.ToComplexStrict1D,
    b: onp.ToJustComplexStrict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[npc.complexfloating]: ...
@overload  # Nd +complexfloating, ~complexfloating
def solve_circulant(
    c: onp.ToComplexND,
    b: onp.ToJustComplexND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[npc.complexfloating]: ...
@overload  # 1d +complexfloating, +complexfloating
def solve_circulant(
    c: onp.ToComplexStrict1D,
    b: onp.ToComplexStrict1D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array1D[npc.inexact]: ...
@overload  # 2d +complexfloating, +complexfloating
def solve_circulant(
    c: onp.ToComplexStrict1D,
    b: onp.ToComplexStrict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.Array2D[npc.inexact]: ...
@overload  # Nd +complexfloating, +complexfloating
def solve_circulant(
    c: onp.ToComplexND,
    b: onp.ToComplexND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> onp.ArrayND[npc.inexact]: ...

# TODO(jorenham): improve this
@overload  # floating 2d
def inv(a: onp.ToFloatStrict2D, overwrite_a: bool = False, check_finite: bool = True) -> _Float2D: ...
@overload  # floating
def inv(a: onp.ToFloatND, overwrite_a: bool = False, check_finite: bool = True) -> _FloatND: ...
@overload  # complexfloating 2d
def inv(a: onp.ToComplexStrict2D, overwrite_a: bool = False, check_finite: bool = True) -> _Complex2D: ...
@overload  # complexfloating
def inv(a: onp.ToComplexND, overwrite_a: bool = False, check_finite: bool = True) -> _ComplexND: ...

# TODO(jorenham): improve this
@overload  # floating 2d
def det(a: onp.ToFloatStrict2D, overwrite_a: bool = False, check_finite: bool = True) -> _Float: ...
@overload  # floating 3d
def det(a: onp.ToFloatStrict3D, overwrite_a: bool = False, check_finite: bool = True) -> _Float1D: ...
@overload  # floating
def det(a: onp.ToFloatND, overwrite_a: bool = False, check_finite: bool = True) -> _Float | _FloatND: ...
@overload  # complexfloating 2d
def det(a: onp.ToJustComplexStrict2D, overwrite_a: bool = False, check_finite: bool = True) -> _Complex1D: ...
@overload  # complexfloating 3d
def det(a: onp.ToJustComplexStrict3D, overwrite_a: bool = False, check_finite: bool = True) -> _ComplexND: ...
@overload  # complexfloating
def det(a: onp.ToComplexND, overwrite_a: bool = False, check_finite: bool = True) -> _Complex | _ComplexND: ...

# TODO(jorenham): improve this
@overload  # (float[:, :], float[:]) -> (float[:], float[], ...)
def lstsq(
    a: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict1D,
    cond: onp.ToFloat | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_Float1D, _Float0D, int, _Float1D | None]: ...
@overload  # (float[:, :], float[:, :]) -> (float[:, :], float[:], ...)
def lstsq(
    a: onp.ToFloatND,
    b: onp.ToFloatStrict2D,
    cond: onp.ToFloat | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_FloatND, _FloatND, int, _FloatND | None]: ...
@overload  # (float[:, :], float[:, :?]) -> (float[:, :?], float[:?], ...)
def lstsq(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    cond: onp.ToFloat | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_FloatND, _Float0D | _FloatND, int, _FloatND | None]: ...
@overload  # (complex[:, :], complex[:, :?]) -> (complex[:, :?], complex[:?], ...)
def lstsq(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    cond: onp.ToFloat | None = None,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_ComplexND, _Complex0D | _ComplexND, int, _ComplexND | None]: ...

# TODO(jorenham): improve this
@overload
def pinv(  # (float[:, :], return_rank=False) -> float[:, :]
    a: onp.ToFloatND,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: onp.ToFalse = False,
    check_finite: bool = True,
) -> _FloatND: ...
@overload  # (float[:, :], return_rank=True) -> (float[:, :], int)
def pinv(
    a: onp.ToFloatND,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_FloatND, int]: ...
@overload  # (complex[:, :], return_rank=False) -> complex[:, :]
def pinv(
    a: onp.ToComplexND,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: onp.ToFalse = False,
    check_finite: bool = True,
) -> _ComplexND: ...
@overload  # (complex[:, :], return_rank=True) -> (complex[:, :], int)
def pinv(
    a: onp.ToComplexND,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_ComplexND, int]: ...

# TODO(jorenham): improve this
@overload  # (float[:, :], return_rank=False) -> float[:, :]
def pinvh(
    a: onp.ToFloatND,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: bool = True,
    return_rank: onp.ToFalse = False,
    check_finite: bool = True,
) -> _FloatND: ...
@overload  # (float[:, :], return_rank=True, /) -> (float[:, :], int)
def pinvh(
    a: onp.ToFloatND,
    atol: onp.ToFloat | None,
    rtol: onp.ToFloat | None,
    lower: bool,
    return_rank: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_FloatND, int]: ...
@overload  # (float[:, :], *, return_rank=True) -> (float[:, :], int)
def pinvh(
    a: onp.ToFloatND,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: bool = True,
    *,
    return_rank: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_FloatND, int]: ...
@overload  # (complex[:, :], return_rank=False) -> complex[:, :]
def pinvh(
    a: onp.ToComplexND,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: bool = True,
    return_rank: onp.ToFalse = False,
    check_finite: bool = True,
) -> _ComplexND: ...
@overload  # (complex[:, :], return_rank=True, /) -> (complex[:, :], int)
def pinvh(
    a: onp.ToComplexND,
    atol: onp.ToFloat | None,
    rtol: onp.ToFloat | None,
    lower: bool,
    return_rank: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_ComplexND, int]: ...
@overload  # (complex[:, :], *, return_rank=True) -> (complex[:, :], int)
def pinvh(
    a: onp.ToComplexND,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: bool = True,
    *,
    return_rank: onp.ToTrue,
    check_finite: bool = True,
) -> tuple[_ComplexND, int]: ...

# TODO(jorenham): improve this
@overload  # (float[:, :], separate=True) -> (float[:, :], float[:, :])
def matrix_balance(
    A: onp.ToFloatND,
    permute: onp.ToBool = True,
    scale: onp.ToBool = True,
    separate: onp.ToFalse = False,
    overwrite_a: bool = False,
) -> _Tuple2[_FloatND]: ...
@overload  # (float[:, :], separate=False, /) -> (float[:, :], (float[:], float[:]))
def matrix_balance(
    A: onp.ToFloatND, permute: onp.ToBool, scale: onp.ToBool, separate: onp.ToTrue, overwrite_a: bool = False
) -> tuple[_FloatND, _Tuple2[_FloatND]]: ...
@overload  # (float[:, :], *, separate=False) -> (float[:, :], (float[:], float[:]))
def matrix_balance(
    A: onp.ToFloatND, permute: onp.ToBool = True, scale: onp.ToBool = True, *, separate: onp.ToTrue, overwrite_a: bool = False
) -> tuple[_FloatND, _Tuple2[_FloatND]]: ...
@overload  # (complex[:, :], separate=True) -> (complex[:, :], complex[:, :])
def matrix_balance(
    A: onp.ToComplexND,
    permute: onp.ToBool = True,
    scale: onp.ToBool = True,
    separate: onp.ToFalse = False,
    overwrite_a: bool = False,
) -> _Tuple2[_ComplexND]: ...
@overload  # (complex[:, :], separate=False, /) -> (complex[:, :], (complex[:], complex[:]))
def matrix_balance(
    A: onp.ToComplexND, permute: onp.ToBool, scale: onp.ToBool, separate: onp.ToTrue, overwrite_a: bool = False
) -> tuple[_ComplexND, _Tuple2[_ComplexND]]: ...
@overload  # (complex[:, :], *, separate=False) -> (complex[:, :], (complex[:], complex[:]))
def matrix_balance(
    A: onp.ToComplexND, permute: onp.ToBool = True, scale: onp.ToBool = True, *, separate: onp.ToTrue, overwrite_a: bool = False
) -> tuple[_ComplexND, _Tuple2[_ComplexND]]: ...

# TODO(jorenham): improve this
@overload  # floating 1d, 1d
def matmul_toeplitz(
    c_or_cr: onp.ToFloatStrict1D | _Tuple2[onp.ToFloatStrict1D],
    x: onp.ToFloatStrict1D,
    check_finite: bool = False,
    workers: onp.ToJustInt | None = None,
) -> _Float1D: ...
@overload  # floating
def matmul_toeplitz(
    c_or_cr: onp.ToFloatND | _Tuple2[onp.ToFloatND],
    x: onp.ToFloatND,
    check_finite: bool = False,
    workers: onp.ToJustInt | None = None,
) -> _FloatND: ...
@overload  # complexfloating 1d, 1d
def matmul_toeplitz(
    c_or_cr: onp.ToComplexStrict1D | _Tuple2[onp.ToComplexStrict1D],
    x: onp.ToComplexStrict1D,
    check_finite: bool = False,
    workers: onp.ToJustInt | None = None,
) -> _Complex1D: ...
@overload  # complexfloating
def matmul_toeplitz(
    c_or_cr: onp.ToComplexND | _Tuple2[onp.ToComplexND],
    x: onp.ToComplexND,
    check_finite: bool = False,
    workers: onp.ToJustInt | None = None,
) -> _ComplexND: ...
