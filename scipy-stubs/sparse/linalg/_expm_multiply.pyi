from typing import Any, Never, SupportsIndex, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from ._interface import LinearOperator
from scipy.sparse._base import _spbase, sparray

__all__ = ["expm_multiply"]

###

type _ToLinearOperator[ScalarT: npc.number | np.bool] = (
    LinearOperator[ScalarT] | _spbase[ScalarT, tuple[int, int]] | onp.ArrayND[ScalarT]
)
type _SparseOrDense[ScalarT: npc.number | np.bool, ShapeT: tuple[Any, ...]] = (
    sparray[ScalarT, ShapeT] | onp.ArrayND[ScalarT, ShapeT]
)

type _AsFloat64 = np.float64 | npc.integer | np.bool
type _ToFloat64 = _AsFloat64 | np.float32 | np.float16

# workaround for mypy's and pyright's typing spec non-compliance regarding overloads
type _JustAnyShape = tuple[Never, Never, Never]

###

@overload  # start: <given>, +f64
def expm_multiply(
    A: _ToLinearOperator[_AsFloat64],
    B: _SparseOrDense[_ToFloat64, tuple[Any, ...]],
    start: onp.ToFloat,
    stop: onp.ToFloat,
    num: SupportsIndex | None = None,
    endpoint: bool | None = None,
    traceA: onp.ToFloat | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload  # start: <given>, ~integer | ~bool
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: _SparseOrDense[npc.integer | np.bool, tuple[Any, ...]],
    start: onp.ToFloat,
    stop: onp.ToFloat,
    num: SupportsIndex | None = None,
    endpoint: bool | None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.ArrayND[InexactT]: ...
@overload  # start: <given>, ~inexact
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: _SparseOrDense[InexactT, tuple[Any, ...]],
    start: onp.ToFloat,
    stop: onp.ToFloat,
    num: SupportsIndex | None = None,
    endpoint: bool | None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.ArrayND[InexactT]: ...
@overload  # ~sparse, +f64
def expm_multiply(
    A: _ToLinearOperator[_AsFloat64],
    B: _spbase[_ToFloat64, tuple[int, int]],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToFloat | None = None,
) -> _spbase[np.float64, tuple[int, int]]: ...
@overload  # ~sparse, ~integer | ~bool
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: _spbase[npc.integer | np.bool, tuple[int, int]],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> _spbase[InexactT, tuple[int, int]]: ...
@overload  # ~sparse, ~inexact
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: _spbase[InexactT, tuple[int, int]],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> _spbase[InexactT, tuple[int, int]]: ...
@overload  # any shape, +f64
def expm_multiply(
    A: _ToLinearOperator[_AsFloat64],
    B: _SparseOrDense[_ToFloat64, _JustAnyShape],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToFloat | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload  # any shape, ~integer | ~bool
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: _SparseOrDense[npc.integer | np.bool, _JustAnyShape],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.ArrayND[InexactT]: ...
@overload  # any shape, ~inexact
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: _SparseOrDense[InexactT, _JustAnyShape],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.ArrayND[InexactT]: ...
@overload  # 1-d, +f64
def expm_multiply(
    A: _ToLinearOperator[_AsFloat64],
    B: onp.Array1D[_ToFloat64],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToFloat | None = None,
) -> onp.Array1D[np.float64]: ...
@overload  # 1-d, ~integer | ~bool
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: onp.Array1D[npc.integer | np.bool],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.Array1D[InexactT]: ...
@overload  # 1-d, ~inexact
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: onp.Array1D[InexactT],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.Array1D[InexactT]: ...
@overload  # 2-d, +f64
def expm_multiply(
    A: _ToLinearOperator[_AsFloat64],
    B: onp.Array2D[_ToFloat64],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToFloat | None = None,
) -> onp.Array2D[np.float64]: ...
@overload  # 2-d, ~integer | ~bool
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: onp.Array2D[npc.integer | np.bool],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.Array2D[InexactT]: ...
@overload  # 2-d, ~inexact
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: onp.Array2D[InexactT],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.Array2D[InexactT]: ...
@overload  # 1-d or 2-d, +f64
def expm_multiply(
    A: _ToLinearOperator[_AsFloat64],
    B: onp.ArrayND[_ToFloat64],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToFloat | None = None,
) -> onp.ArrayND[np.float64]: ...
@overload  # 1-d or 2-d, ~integer | ~bool
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: onp.ArrayND[npc.integer | np.bool],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.ArrayND[InexactT]: ...
@overload  # 1-d or 2-d, ~inexact
def expm_multiply[InexactT: npc.inexact](
    A: _ToLinearOperator[InexactT],
    B: onp.ArrayND[InexactT],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.ArrayND[InexactT]: ...
@overload  # fallback
def expm_multiply(
    A: _ToLinearOperator[npc.number],
    B: onp.ArrayND[npc.number],
    start: None = None,
    stop: None = None,
    num: None = None,
    endpoint: None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.ArrayND[Any]: ...
