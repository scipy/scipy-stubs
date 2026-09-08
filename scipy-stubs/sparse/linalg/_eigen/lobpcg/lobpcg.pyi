from collections.abc import Callable
from typing import Any, TypeAliasType, overload
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

__all__ = ["lobpcg"]

###

type _ToInt = npc.integer | np.bool
type _ToFloat = npc.floating | _ToInt
type _ToComplex = npc.number | np.bool
type _ToF64 = npc.floating64 | npc.floating32 | npc.floating16 | _ToInt
type _AsF64 = npc.floating64 | npc.integer64 | npc.integer32

type _MatOp[ScalarT: _ToComplex] = _spbase[ScalarT] | LinearOperator[ScalarT]

type _ToMatF64 = onp.ToFloat64_2D | _MatOp[_ToF64]
type _ToMatC128 = onp.ToComplex128_2D | _MatOp[npc.inexact64 | npc.inexact32 | np.float16 | _ToInt]
type _AsMatF32 = onp.ToJustFloat32_2D | _spbase[npc.floating32]
type _AsMatF64 = onp.ToArray2D[op.JustFloat | op.JustInt, _AsF64] | _MatOp[_AsF64]
type _AsMatC128 = onp.ToJustComplex128_2D | _MatOp[npc.complexfloating128]

_InexactT = TypeVar("_InexactT", bound=npc.inexact, default=Any)
_ToMat = TypeAliasType(
    "_ToMat",
    onp.ToComplex2D | _MatOp[_ToComplex] | Callable[[onp.Array2D[_InexactT]], onp.ArrayND[_ToComplex]],
    type_params=(_InexactT,),
)

type _Result[ValT: npc.inexact, VecT: npc.inexact] = tuple[
    onp.Array1D[ValT],  # lambda
    onp.Array2D[VecT],  # v
]
type _Result1[ValT: npc.inexact, VecT: npc.inexact, HistT: npc.inexact] = tuple[
    onp.Array1D[ValT],  # lambda
    onp.Array2D[VecT],  # v
    list[onp.Array1D[HistT]],  # lambdaHistory / residualNormsHistory
]
type _Result2[ValT: npc.inexact, VecT: npc.inexact, HistT: npc.inexact] = tuple[
    onp.Array1D[ValT],  # lambda
    onp.Array2D[VecT],  # v
    list[onp.Array1D[HistT]],  # lambdaHistory
    list[onp.Array1D[HistT]],  # residualNormsHistory
]

###

# mypy reports false positive `overload-overlap` errors on `numpy>=2.2`
# mypy: disable-error-code=overload-overlap

@overload  # A: +f64, X: ~f64
def lobpcg(
    A: _AsMatF64,
    X: onp.ArrayND[np.float64],
    B: _ToMatF64 | None = None,
    M: _ToMatF64 | None = None,
    Y: onp.ArrayND[_ToFloat] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    retLambdaHistory: onp.ToFalse = False,
    retResidualNormsHistory: onp.ToFalse = False,
    restartControl: int = 20,
) -> _Result[np.float64, np.float64]: ...
@overload  # A: +f64, X: ~f64, retLambdaHistory
def lobpcg(
    A: _AsMatF64,
    X: onp.ArrayND[np.float64],
    B: _ToMatF64 | None = None,
    M: _ToMatF64 | None = None,
    Y: onp.ArrayND[_ToFloat] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    *,
    retLambdaHistory: onp.ToTrue,
    retResidualNormsHistory: onp.ToFalse = False,
    restartControl: int = 20,
) -> _Result1[np.float64, np.float64, np.float64]: ...
@overload  # A: +f64, X: ~f64, retResidualNormsHistory
def lobpcg(
    A: _AsMatF64,
    X: onp.ArrayND[np.float64],
    B: _ToMatF64 | None = None,
    M: _ToMatF64 | None = None,
    Y: onp.ArrayND[_ToFloat] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    retLambdaHistory: onp.ToFalse = False,
    *,
    retResidualNormsHistory: onp.ToTrue,
    restartControl: int = 20,
) -> _Result1[np.float64, np.float64, np.float64]: ...
@overload  # A: +f64, X: ~f64, retLambdaHistory, retResidualNormsHistory
def lobpcg(
    A: _AsMatF64,
    X: onp.ArrayND[np.float64],
    B: _ToMatF64 | None = None,
    M: _ToMatF64 | None = None,
    Y: onp.ArrayND[_ToFloat] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    *,
    retLambdaHistory: onp.ToTrue,
    retResidualNormsHistory: onp.ToTrue,
    restartControl: int = 20,
) -> _Result2[np.float64, np.float64, np.float64]: ...
@overload  # A: ~c128, X: ~f64 | ~c128
def lobpcg(
    A: _AsMatC128,
    X: onp.ArrayND[np.float64 | np.complex128],
    B: _ToMatC128 | None = None,
    M: _ToMatC128 | None = None,
    Y: onp.ArrayND[_ToComplex] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    retLambdaHistory: onp.ToFalse = False,
    retResidualNormsHistory: onp.ToFalse = False,
    restartControl: int = 20,
) -> _Result[np.float64, np.complex128]: ...
@overload  # A: ~c128, X: T:inexact64, retLambdaHistory
def lobpcg[InexactT: npc.inexact64](
    A: _AsMatC128,
    X: onp.ArrayND[InexactT],
    B: _ToMatC128 | None = None,
    M: _ToMatC128 | None = None,
    Y: onp.ArrayND[_ToComplex] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    *,
    retLambdaHistory: onp.ToTrue,
    retResidualNormsHistory: onp.ToFalse = False,
    restartControl: int = 20,
) -> _Result1[np.float64, np.complex128, InexactT]: ...
@overload  # A: ~c128, X: T:inexact64, retResidualNormsHistory
def lobpcg[ScalarT: npc.inexact64](
    A: _AsMatC128,
    X: onp.ArrayND[ScalarT],
    B: _ToMatC128 | None = None,
    M: _ToMatC128 | None = None,
    Y: onp.ArrayND[_ToComplex] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    retLambdaHistory: onp.ToFalse = False,
    *,
    retResidualNormsHistory: onp.ToTrue,
    restartControl: int = 20,
) -> _Result1[np.float64, np.complex128, ScalarT]: ...
@overload  # A: ~c128, X: T:inexact64, retLambdaHistory, retResidualNormsHistory
def lobpcg[InexactT: npc.inexact64](
    A: _AsMatC128,
    X: onp.ArrayND[InexactT],
    B: _ToMatC128 | None = None,
    M: _ToMatC128 | None = None,
    Y: onp.ArrayND[_ToComplex] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    *,
    retLambdaHistory: onp.ToTrue,
    retResidualNormsHistory: onp.ToTrue,
    restartControl: int = 20,
) -> _Result2[np.float64, np.complex128, InexactT]: ...
@overload  # A: ~f32, X: T:inexact32
def lobpcg[InexactT: npc.inexact32](
    A: _AsMatF32,
    X: onp.ArrayND[InexactT],
    B: _AsMatF32 | None = None,
    M: _AsMatF32 | None = None,
    Y: onp.ArrayND[_ToFloat] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    retLambdaHistory: onp.ToFalse = False,
    retResidualNormsHistory: onp.ToFalse = False,
    restartControl: int = 20,
) -> _Result[np.float32, InexactT]: ...
@overload  # A: ~f32, X: T:inexact32, retLambdaHistory
def lobpcg[InexactT: npc.inexact32](
    A: _AsMatF32,
    X: onp.ArrayND[InexactT],
    B: _AsMatF32 | None = None,
    M: _AsMatF32 | None = None,
    Y: onp.ArrayND[_ToFloat] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    *,
    retLambdaHistory: onp.ToTrue,
    retResidualNormsHistory: onp.ToFalse = False,
    restartControl: int = 20,
) -> _Result1[np.float32, InexactT, InexactT]: ...
@overload  # A: ~f32, X: T:inexact32, retResidualNormsHistory
def lobpcg[InexactT: npc.inexact32](
    A: _AsMatF32,
    X: onp.ArrayND[InexactT],
    B: _AsMatF32 | None = None,
    M: _AsMatF32 | None = None,
    Y: onp.ArrayND[_ToFloat] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    retLambdaHistory: onp.ToFalse = False,
    *,
    retResidualNormsHistory: onp.ToTrue,
    restartControl: int = 20,
) -> _Result1[np.float32, InexactT, InexactT]: ...
@overload  # A: ~f32, X: T:inexact32, retLambdaHistory, retResidualNormsHistory
def lobpcg[InexactT: npc.inexact32](
    A: _AsMatF32,
    X: onp.ArrayND[InexactT],
    B: _AsMatF32 | None = None,
    M: _AsMatF32 | None = None,
    Y: onp.ArrayND[_ToFloat] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    *,
    retLambdaHistory: onp.ToTrue,
    retResidualNormsHistory: onp.ToTrue,
    restartControl: int = 20,
) -> _Result2[np.float32, InexactT, InexactT]: ...
@overload  # fallback
def lobpcg(
    A: _ToMat,
    X: onp.ArrayND[npc.inexact],
    B: _ToMat | None = None,
    M: _ToMat | None = None,
    Y: onp.ArrayND[_ToComplex] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    retLambdaHistory: onp.ToFalse = False,
    retResidualNormsHistory: onp.ToFalse = False,
    restartControl: int = 20,
) -> _Result[np.float64 | Any, np.float64 | Any]: ...
@overload  # fallback, retLambdaHistory
def lobpcg[InexactT: npc.inexact](
    A: _ToMat[InexactT],
    X: onp.ArrayND[InexactT],
    B: _ToMat[InexactT] | None = None,
    M: _ToMat[InexactT] | None = None,
    Y: onp.ArrayND[_ToComplex] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    *,
    retLambdaHistory: onp.ToTrue,
    retResidualNormsHistory: onp.ToFalse = False,
    restartControl: int = 20,
) -> _Result1[np.float64 | Any, np.float64 | Any, InexactT]: ...
@overload  # fallback, retResidualNormsHistory
def lobpcg[InexactT: npc.inexact](
    A: _ToMat[InexactT],
    X: onp.ArrayND[InexactT],
    B: _ToMat[InexactT] | None = None,
    M: _ToMat[InexactT] | None = None,
    Y: onp.ArrayND[_ToComplex] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    retLambdaHistory: onp.ToFalse = False,
    *,
    retResidualNormsHistory: onp.ToTrue,
    restartControl: int = 20,
) -> _Result1[np.float64 | Any, np.float64 | Any, InexactT]: ...
@overload  # fallback, retLambdaHistory, retResidualNormsHistory
def lobpcg[InexactT: npc.inexact](
    A: _ToMat[InexactT],
    X: onp.ArrayND[InexactT],
    B: _ToMat[InexactT] | None = None,
    M: _ToMat[InexactT] | None = None,
    Y: onp.ArrayND[_ToComplex] | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    largest: bool = True,
    verbosityLevel: int = 0,
    *,
    retLambdaHistory: onp.ToTrue,
    retResidualNormsHistory: onp.ToTrue,
    restartControl: int = 20,
) -> _Result2[np.float64 | Any, np.float64 | Any, InexactT]: ...
