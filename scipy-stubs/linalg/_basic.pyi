from typing import Any, Final, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp

from scipy._typing import Falsy, Truthy

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

_Float: TypeAlias = np.floating[Any]
_Float0D: TypeAlias = onp.Array0D[_Float]
_Float1D: TypeAlias = onp.Array1D[_Float]
_Float2D: TypeAlias = onp.Array2D[_Float]
_FloatND: TypeAlias = onp.ArrayND[_Float]

_Complex: TypeAlias = np.inexact[Any]  # float and complex input types are near impossible to distinguish
_Complex0D: TypeAlias = onp.Array0D[_Complex]
_Complex1D: TypeAlias = onp.Array1D[_Complex]
_Complex2D: TypeAlias = onp.Array2D[_Complex]
_ComplexND: TypeAlias = onp.ArrayND[_Complex]

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

@overload  # floating, 2d, 2d
def solve(
    a: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict2D,
    lower: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    assume_a: _AssumeA | None = None,
    transposed: onp.ToBool = False,
) -> _Float2D: ...
@overload  # floating
def solve(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    lower: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    assume_a: _AssumeA | None = None,
    transposed: onp.ToBool = False,
) -> _FloatND: ...
@overload  # complexfloating 2d, 2d
def solve(
    a: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict2D,
    lower: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    assume_a: _AssumeA | None = None,
    transposed: onp.ToBool = False,
) -> _Complex2D: ...
@overload  # complexfloating
def solve(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    lower: onp.ToBool = False,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    assume_a: _AssumeA | None = None,
    transposed: onp.ToBool = False,
) -> _ComplexND: ...

#
@overload  # floating 2d, 1d
def solve_triangular(
    a: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict1D,
    trans: _TransSystem = 0,
    lower: onp.ToBool = False,
    unit_diagonal: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float1D: ...
@overload  # floating
def solve_triangular(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    trans: _TransSystem = 0,
    lower: onp.ToBool = False,
    unit_diagonal: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _FloatND: ...
@overload  # complexfloating 2d, 1d
def solve_triangular(
    a: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict1D,
    trans: _TransSystem = 0,
    lower: onp.ToBool = False,
    unit_diagonal: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex1D: ...
@overload  # complexfloating
def solve_triangular(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    trans: _TransSystem = 0,
    lower: onp.ToBool = False,
    unit_diagonal: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _ComplexND: ...

#
@overload  # floating 2d, 1d
def solve_banded(
    l_and_u: _Tuple2[onp.ToJustInt],
    ab: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict1D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float1D: ...
@overload  # floating
def solve_banded(
    l_and_u: _Tuple2[onp.ToJustInt],
    ab: onp.ToFloatND,
    b: onp.ToFloatND,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _FloatND: ...
@overload  # complexfloating 2d, 1d
def solve_banded(
    l_and_u: _Tuple2[onp.ToJustInt],
    ab: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict1D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex1D: ...
@overload  # complexfloating
def solve_banded(
    l_and_u: _Tuple2[onp.ToJustInt],
    ab: onp.ToComplexND,
    b: onp.ToComplexND,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _ComplexND: ...

#
@overload  # floating 2d, 1d
def solveh_banded(
    ab: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict1D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    lower: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Float1D: ...
@overload  # floating
def solveh_banded(
    ab: onp.ToFloatND,
    b: onp.ToFloatND,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    lower: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _FloatND: ...
@overload  # complexfloating 2d, 1d
def solveh_banded(
    ab: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict1D,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    lower: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Complex1D: ...
@overload  # complexfloating
def solveh_banded(
    ab: onp.ToComplexND,
    b: onp.ToComplexND,
    overwrite_ab: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    lower: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _ComplexND: ...

#
@overload  # floating 1d
def solve_toeplitz(
    c_or_cr: onp.ToFloatStrict1D | _Tuple2[onp.ToFloatStrict1D], b: onp.ToFloat1D, check_finite: onp.ToBool = True
) -> _Float1D: ...
@overload  # floating
def solve_toeplitz(
    c_or_cr: onp.ToFloatND | _Tuple2[onp.ToFloatND], b: onp.ToFloatND, check_finite: onp.ToBool = True
) -> _FloatND: ...
@overload  # complexfloating 1d
def solve_toeplitz(
    c_or_cr: onp.ToComplexStrict1D | _Tuple2[onp.ToComplexStrict1D], b: onp.ToComplex1D, check_finite: onp.ToBool = True
) -> _Complex1D: ...
@overload  # complexfloating
def solve_toeplitz(
    c_or_cr: onp.ToComplexND | _Tuple2[onp.ToComplexND], b: onp.ToComplexND, check_finite: onp.ToBool = True
) -> _ComplexND: ...

#
@overload  # floating 2d, 2d
def solve_circulant(
    c: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> _Float1D: ...
@overload  # floating
def solve_circulant(
    c: onp.ToFloatND,
    b: onp.ToFloatND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> _FloatND: ...
@overload  # complexfloating 2d, 2d
def solve_circulant(
    c: onp.ToComplexStrict2D,
    b: onp.ToComplexStrict2D,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> _Complex1D: ...
@overload  # complexfloating
def solve_circulant(
    c: onp.ToComplexND,
    b: onp.ToComplexND,
    singular: _Singular = "raise",
    tol: onp.ToFloat | None = None,
    caxis: op.CanIndex = -1,
    baxis: op.CanIndex = 0,
    outaxis: op.CanIndex = 0,
) -> _ComplexND: ...

#
@overload  # floating 2d
def inv(a: onp.ToFloatStrict2D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Float2D: ...
@overload  # floating
def inv(a: onp.ToFloatND, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _FloatND: ...
@overload  # complexfloating 2d
def inv(a: onp.ToComplexStrict2D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Complex2D: ...
@overload  # complexfloating
def inv(a: onp.ToComplexND, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _ComplexND: ...

#
@overload  # floating 2d
def det(a: onp.ToFloatStrict2D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Float: ...
@overload  # floating 3d
def det(a: onp.ToFloatStrict3D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Float1D: ...
@overload  # floating
def det(a: onp.ToFloatND, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Float | _FloatND: ...
@overload  # complexfloating 2d
def det(a: onp.ToJustComplexStrict2D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Complex1D: ...
@overload  # complexfloating 3d
def det(a: onp.ToJustComplexStrict3D, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _ComplexND: ...
@overload  # complexfloating
def det(a: onp.ToComplexND, overwrite_a: onp.ToBool = False, check_finite: onp.ToBool = True) -> _Complex | _ComplexND: ...

#
@overload  # (float[:, :], float[:]) -> (float[:], float[], ...)
def lstsq(
    a: onp.ToFloatStrict2D,
    b: onp.ToFloatStrict1D,
    cond: onp.ToFloat | None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_Float1D, _Float0D, int, _Float1D | None]: ...
@overload  # (float[:, :], float[:, :]) -> (float[:, :], float[:], ...)
def lstsq(
    a: onp.ToFloatND,
    b: onp.ToFloatStrict2D,
    cond: onp.ToFloat | None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_FloatND, _FloatND, int, _FloatND | None]: ...
@overload  # (float[:, :], float[:, :?]) -> (float[:, :?], float[:?], ...)
def lstsq(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    cond: onp.ToFloat | None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_FloatND, _Float0D | _FloatND, int, _FloatND | None]: ...
@overload  # (complex[:, :], complex[:, :?]) -> (complex[:, :?], complex[:?], ...)
def lstsq(
    a: onp.ToComplexND,
    b: onp.ToComplexND,
    cond: onp.ToFloat | None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
    lapack_driver: _LapackDriver | None = None,
) -> tuple[_ComplexND, _Complex0D | _ComplexND, int, _ComplexND | None]: ...

#
@overload
def pinv(  # (float[:, :], return_rank=False) -> float[:, :]
    a: onp.ToFloatND,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: Falsy = False,
    check_finite: onp.ToBool = True,
) -> _FloatND: ...
@overload  # (float[:, :], return_rank=True) -> (float[:, :], int)
def pinv(
    a: onp.ToFloatND,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_FloatND, int]: ...
@overload  # (complex[:, :], return_rank=False) -> complex[:, :]
def pinv(
    a: onp.ToComplexND,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: Falsy = False,
    check_finite: onp.ToBool = True,
) -> _ComplexND: ...
@overload  # (complex[:, :], return_rank=True) -> (complex[:, :], int)
def pinv(
    a: onp.ToComplexND,
    *,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_ComplexND, int]: ...

#
@overload  # (float[:, :], return_rank=False) -> float[:, :]
def pinvh(
    a: onp.ToFloatND,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: onp.ToBool = True,
    return_rank: Falsy = False,
    check_finite: onp.ToBool = True,
) -> _FloatND: ...
@overload  # (float[:, :], return_rank=True, /) -> (float[:, :], int)
def pinvh(
    a: onp.ToFloatND,
    atol: onp.ToFloat | None,
    rtol: onp.ToFloat | None,
    lower: onp.ToBool,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_FloatND, int]: ...
@overload  # (float[:, :], *, return_rank=True) -> (float[:, :], int)
def pinvh(
    a: onp.ToFloatND,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: onp.ToBool = True,
    *,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_FloatND, int]: ...
@overload  # (complex[:, :], return_rank=False) -> complex[:, :]
def pinvh(
    a: onp.ToComplexND,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: onp.ToBool = True,
    return_rank: Falsy = False,
    check_finite: onp.ToBool = True,
) -> _ComplexND: ...
@overload  # (complex[:, :], return_rank=True, /) -> (complex[:, :], int)
def pinvh(
    a: onp.ToComplexND,
    atol: onp.ToFloat | None,
    rtol: onp.ToFloat | None,
    lower: onp.ToBool,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_ComplexND, int]: ...
@overload  # (complex[:, :], *, return_rank=True) -> (complex[:, :], int)
def pinvh(
    a: onp.ToComplexND,
    atol: onp.ToFloat | None = None,
    rtol: onp.ToFloat | None = None,
    lower: onp.ToBool = True,
    *,
    return_rank: Truthy,
    check_finite: onp.ToBool = True,
) -> tuple[_ComplexND, int]: ...

#
@overload  # (float[:, :], separate=True) -> (float[:, :], float[:, :])
def matrix_balance(
    A: onp.ToFloatND,
    permute: onp.ToBool = True,
    scale: onp.ToBool = True,
    separate: Falsy = False,
    overwrite_a: onp.ToBool = False,
) -> _Tuple2[_FloatND]: ...
@overload  # (float[:, :], separate=False, /) -> (float[:, :], (float[:], float[:]))
def matrix_balance(
    A: onp.ToFloatND, permute: onp.ToBool, scale: onp.ToBool, separate: Truthy, overwrite_a: onp.ToBool = False
) -> tuple[_FloatND, _Tuple2[_FloatND]]: ...
@overload  # (float[:, :], *, separate=False) -> (float[:, :], (float[:], float[:]))
def matrix_balance(
    A: onp.ToFloatND, permute: onp.ToBool = True, scale: onp.ToBool = True, *, separate: Truthy, overwrite_a: onp.ToBool = False
) -> tuple[_FloatND, _Tuple2[_FloatND]]: ...
@overload  # (complex[:, :], separate=True) -> (complex[:, :], complex[:, :])
def matrix_balance(
    A: onp.ToComplexND,
    permute: onp.ToBool = True,
    scale: onp.ToBool = True,
    separate: Falsy = False,
    overwrite_a: onp.ToBool = False,
) -> _Tuple2[_ComplexND]: ...
@overload  # (complex[:, :], separate=False, /) -> (complex[:, :], (complex[:], complex[:]))
def matrix_balance(
    A: onp.ToComplexND, permute: onp.ToBool, scale: onp.ToBool, separate: Truthy, overwrite_a: onp.ToBool = False
) -> tuple[_ComplexND, _Tuple2[_ComplexND]]: ...
@overload  # (complex[:, :], *, separate=False) -> (complex[:, :], (complex[:], complex[:]))
def matrix_balance(
    A: onp.ToComplexND, permute: onp.ToBool = True, scale: onp.ToBool = True, *, separate: Truthy, overwrite_a: onp.ToBool = False
) -> tuple[_ComplexND, _Tuple2[_ComplexND]]: ...

#
@overload  # floating 1d, 1d
def matmul_toeplitz(
    c_or_cr: onp.ToFloatStrict1D | _Tuple2[onp.ToFloatStrict1D],
    x: onp.ToFloatStrict1D,
    check_finite: onp.ToBool = False,
    workers: onp.ToJustInt | None = None,
) -> _Float1D: ...
@overload  # floating
def matmul_toeplitz(
    c_or_cr: onp.ToFloatND | _Tuple2[onp.ToFloatND],
    x: onp.ToFloatND,
    check_finite: onp.ToBool = False,
    workers: onp.ToJustInt | None = None,
) -> _FloatND: ...
@overload  # complexfloating 1d, 1d
def matmul_toeplitz(
    c_or_cr: onp.ToComplexStrict1D | _Tuple2[onp.ToComplexStrict1D],
    x: onp.ToComplexStrict1D,
    check_finite: onp.ToBool = False,
    workers: onp.ToJustInt | None = None,
) -> _Complex1D: ...
@overload  # complexfloating
def matmul_toeplitz(
    c_or_cr: onp.ToComplexND | _Tuple2[onp.ToComplexND],
    x: onp.ToComplexND,
    check_finite: onp.ToBool = False,
    workers: onp.ToJustInt | None = None,
) -> _ComplexND: ...
