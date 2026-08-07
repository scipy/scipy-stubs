from collections.abc import Callable
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.sparse import (
    bsr_array,
    bsr_matrix,
    coo_array,
    coo_matrix,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
    dia_array,
    dia_matrix,
    dok_array,
    dok_matrix,
    lil_array,
    lil_matrix,
    sparray,
    spmatrix,
)
from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

###

type _Numeric = npc.number | np.bool

type _ToGraph[ScalarT: _Numeric] = onp.CanArrayND[ScalarT] | _spbase[ScalarT, tuple[int, int]]
type _ToAnyGraph = onp.ToComplex2D | _spbase[_Numeric, tuple[int, int]] | sparray[_Numeric] | spmatrix[_Numeric]

type _SpArray[ScalarT: _Numeric] = (
    bsr_array[ScalarT] | coo_array[ScalarT] | csc_array[ScalarT] | csr_array[ScalarT] | dok_array[ScalarT] | lil_array[ScalarT]
)
type _SpMatrix[ScalarT: _Numeric] = (
    bsr_matrix[ScalarT]
    | coo_matrix[ScalarT]
    | csc_matrix[ScalarT]
    | csr_matrix[ScalarT]
    | dok_matrix[ScalarT]
    | lil_matrix[ScalarT]
)

type _LaplacianFunction[ScalarT: _Numeric] = Callable[[onp.ToComplex2D], onp.Array2D[ScalarT]]
type _LaplacianMatrix = onp.Array2D[Any] | _spbase[Any, tuple[int, int]]
type _LaplacianAny = _LaplacianFunction[Any] | LinearOperator[Any, tuple[int, int]] | _LaplacianMatrix

type _Form = Literal["array", "function", "lo"]

###

@overload  # ~integer, normed: True, form: "function"
def laplacian(
    csgraph: _ToGraph[npc.integer],
    normed: onp.ToTrue,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["function"],
    dtype: None = None,
    symmetrized: bool = False,
) -> _LaplacianFunction[np.float64 | Any]: ...
@overload  # +numeric, form: "function"
def laplacian[ScalarT: _Numeric](
    csgraph: _ToGraph[ScalarT],
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["function"],
    dtype: None = None,
    symmetrized: bool = False,
) -> _LaplacianFunction[ScalarT | Any]: ...
@overload  # catch-all, form: "function"
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["function"],
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> _LaplacianFunction[Any]: ...
@overload  # +numeric, return_diag: True (keyword), form: "function", dtype: <known>
def laplacian[ScalarT1: _Numeric, ScalarT2: _Numeric](
    csgraph: _ToGraph[ScalarT1],
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["function"],
    dtype: onp.ToDType[ScalarT2],
    symmetrized: bool = False,
) -> tuple[_LaplacianFunction[ScalarT1 | Any], onp.Array1D[ScalarT2]]: ...
@overload  # ~integer, normed: True, return_diag: True (keyword), form: "function"
def laplacian(
    csgraph: _ToGraph[npc.integer],
    normed: onp.ToTrue,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["function"],
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianFunction[np.float64 | Any], onp.Array1D[np.float64]]: ...
@overload  # +numeric, return_diag: True (keyword), form: "function"
def laplacian[ScalarT: _Numeric](
    csgraph: _ToGraph[ScalarT],
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["function"],
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianFunction[ScalarT | Any], onp.Array1D[ScalarT]]: ...
@overload  # catch-all, return_diag: True (keyword), form: "function"
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["function"],
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianFunction[Any], onp.Array1D[Any]]: ...
@overload  # catch-all, return_diag: True (positional), form: "function"
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["function"],
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianFunction[Any], onp.Array1D[Any]]: ...
@overload  # +numeric, form: "lo", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: _ToGraph[Any],
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["lo"],
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> LinearOperator[ScalarT, tuple[int, int]]: ...
@overload  # ~integer, normed: True, form: "lo"
def laplacian(
    csgraph: _ToGraph[npc.integer],
    normed: onp.ToTrue,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["lo"],
    dtype: None = None,
    symmetrized: bool = False,
) -> LinearOperator[np.float64, tuple[int, int]]: ...
@overload  # +numeric, form: "lo"
def laplacian[ScalarT: _Numeric](
    csgraph: _ToGraph[ScalarT],
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["lo"],
    dtype: None = None,
    symmetrized: bool = False,
) -> LinearOperator[ScalarT, tuple[int, int]]: ...
@overload  # catch-all, form: "lo"
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["lo"],
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> LinearOperator[Any, tuple[int, int]]: ...
@overload  # +numeric, return_diag: True (keyword), form: "lo", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: _ToGraph[Any],
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["lo"],
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> tuple[LinearOperator[ScalarT, tuple[int, int]], onp.Array1D[ScalarT]]: ...
@overload  # ~integer, normed: True, return_diag: True (keyword), form: "lo"
def laplacian(
    csgraph: _ToGraph[npc.integer],
    normed: onp.ToTrue,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["lo"],
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[LinearOperator[np.float64, tuple[int, int]], onp.Array1D[np.float64]]: ...
@overload  # +numeric, return_diag: True (keyword), form: "lo"
def laplacian[ScalarT: _Numeric](
    csgraph: _ToGraph[ScalarT],
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["lo"],
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[LinearOperator[ScalarT, tuple[int, int]], onp.Array1D[ScalarT]]: ...
@overload  # catch-all, return_diag: True (keyword), form: "lo"
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["lo"],
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[LinearOperator[Any, tuple[int, int]], onp.Array1D[Any]]: ...
@overload  # catch-all, return_diag: True (positional), form: "lo"
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["lo"],
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[LinearOperator[Any, tuple[int, int]], onp.Array1D[Any]]: ...
@overload  # dia_array, normed: False, form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: dia_array[ScalarT],
    normed: onp.ToFalse = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> dia_array[ScalarT]: ...
@overload  # dia_matrix, normed: False, form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: dia_matrix[ScalarT],
    normed: onp.ToFalse = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> dia_matrix[ScalarT]: ...
@overload  # dia_array, normed: False, form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: dia_array[Any],
    normed: onp.ToFalse = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> dia_array[ScalarT]: ...
@overload  # dia_matrix, normed: False, form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: dia_matrix[Any],
    normed: onp.ToFalse = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> dia_matrix[ScalarT]: ...
@overload  # dia_array, normed: True, form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: dia_array[Any],
    normed: onp.ToTrue,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> coo_array[ScalarT, tuple[int, int]]: ...
@overload  # dia_matrix, normed: True, form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: dia_matrix[Any],
    normed: onp.ToTrue,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> coo_matrix[ScalarT]: ...
@overload  # sparray, form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: _SpArray[Any],
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> coo_array[ScalarT, tuple[int, int]]: ...
@overload  # spmatrix, form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: _SpMatrix[Any],
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> coo_matrix[ScalarT]: ...
@overload  # dense, form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: onp.CanArrayND[Any],
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> onp.Array2D[ScalarT]: ...
@overload  # sparray[~integer], normed: True, form: "array"
def laplacian(
    csgraph: _SpArray[npc.integer] | dia_array[npc.integer],
    normed: onp.ToTrue,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> coo_array[np.float64, tuple[int, int]]: ...
@overload  # spmatrix[~integer], normed: True, form: "array"
def laplacian(
    csgraph: _SpMatrix[npc.integer] | dia_matrix[npc.integer],
    normed: onp.ToTrue,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> coo_matrix[np.float64]: ...
@overload  # dense[~integer], normed: True, form: "array"
def laplacian(
    csgraph: onp.CanArrayND[npc.integer],
    normed: onp.ToTrue,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> onp.Array2D[np.float64]: ...
@overload  # dia_array, normed: True, form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: dia_array[ScalarT],
    normed: onp.ToTrue,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> coo_array[ScalarT, tuple[int, int]]: ...
@overload  # dia_matrix, normed: True, form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: dia_matrix[ScalarT],
    normed: onp.ToTrue,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> coo_matrix[ScalarT]: ...
@overload  # sparray, form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: _SpArray[ScalarT],
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> coo_array[ScalarT, tuple[int, int]]: ...
@overload  # spmatrix, form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: _SpMatrix[ScalarT],
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> coo_matrix[ScalarT]: ...
@overload  # dense, form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: onp.CanArrayND[ScalarT],
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> onp.Array2D[ScalarT]: ...
@overload  # catch-all, form: "array"
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> _LaplacianMatrix: ...
@overload  # dia_array, normed: False, return_diag: True (keyword), form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: dia_array[ScalarT],
    normed: onp.ToFalse = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[dia_array[ScalarT], onp.Array1D[ScalarT]]: ...
@overload  # dia_matrix, normed: False, return_diag: True (keyword), form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: dia_matrix[ScalarT],
    normed: onp.ToFalse = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[dia_matrix[ScalarT], onp.Array1D[ScalarT]]: ...
@overload  # dia_array, normed: False, return_diag: True (keyword), form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: dia_array[Any],
    normed: onp.ToFalse = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> tuple[dia_array[ScalarT], onp.Array1D[ScalarT]]: ...
@overload  # dia_matrix, normed: False, return_diag: True (keyword), form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: dia_matrix[Any],
    normed: onp.ToFalse = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> tuple[dia_matrix[ScalarT], onp.Array1D[ScalarT]]: ...
@overload  # dia_array, normed: True, return_diag: True (keyword), form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: dia_array[Any],
    normed: onp.ToTrue,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> tuple[coo_array[ScalarT, tuple[int, int]], onp.Array1D[ScalarT]]: ...
@overload  # dia_matrix, normed: True, return_diag: True (keyword), form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: dia_matrix[Any],
    normed: onp.ToTrue,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> tuple[coo_matrix[ScalarT], onp.Array1D[ScalarT]]: ...
@overload  # sparray, return_diag: True (keyword), form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: _SpArray[Any],
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> tuple[coo_array[ScalarT, tuple[int, int]], onp.Array1D[ScalarT]]: ...
@overload  # spmatrix, return_diag: True (keyword), form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: _SpMatrix[Any],
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> tuple[coo_matrix[ScalarT], onp.Array1D[ScalarT]]: ...
@overload  # dense, return_diag: True (keyword), form: "array", dtype: <known>
def laplacian[ScalarT: _Numeric](
    csgraph: onp.CanArrayND[Any],
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: onp.ToDType[ScalarT],
    symmetrized: bool = False,
) -> tuple[onp.Array2D[ScalarT], onp.Array1D[ScalarT]]: ...
@overload  # sparray[~integer], normed: True, return_diag: True (keyword), form: "array"
def laplacian(
    csgraph: _SpArray[npc.integer] | dia_array[npc.integer],
    normed: onp.ToTrue,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[coo_array[np.float64, tuple[int, int]], onp.Array1D[np.float64]]: ...
@overload  # spmatrix[~integer], normed: True, return_diag: True (keyword), form: "array"
def laplacian(
    csgraph: _SpMatrix[npc.integer] | dia_matrix[npc.integer],
    normed: onp.ToTrue,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[coo_matrix[np.float64], onp.Array1D[np.float64]]: ...
@overload  # dense[~integer], normed: True, return_diag: True (keyword), form: "array"
def laplacian(
    csgraph: onp.CanArrayND[npc.integer],
    normed: onp.ToTrue,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[onp.Array2D[np.float64], onp.Array1D[np.float64]]: ...
@overload  # dia_array, normed: True, return_diag: True (keyword), form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: dia_array[ScalarT],
    normed: onp.ToTrue,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[coo_array[ScalarT, tuple[int, int]], onp.Array1D[ScalarT]]: ...
@overload  # dia_matrix, normed: True, return_diag: True (keyword), form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: dia_matrix[ScalarT],
    normed: onp.ToTrue,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[coo_matrix[ScalarT], onp.Array1D[ScalarT]]: ...
@overload  # sparray, return_diag: True (keyword), form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: _SpArray[ScalarT],
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[coo_array[ScalarT, tuple[int, int]], onp.Array1D[ScalarT]]: ...
@overload  # spmatrix, return_diag: True (keyword), form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: _SpMatrix[ScalarT],
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[coo_matrix[ScalarT], onp.Array1D[ScalarT]]: ...
@overload  # dense, return_diag: True (keyword), form: "array"
def laplacian[ScalarT: _Numeric](
    csgraph: onp.CanArrayND[ScalarT],
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: None = None,
    symmetrized: bool = False,
) -> tuple[onp.Array2D[ScalarT], onp.Array1D[ScalarT]]: ...
@overload  # catch-all, return_diag: True (keyword), form: "array"
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianMatrix, onp.Array1D[Any]]: ...
@overload  # catch-all, return_diag: True (positional), form: "array"
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianMatrix, onp.Array1D[Any]]: ...
@overload  # catch-all, form: <any>
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: _Form = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> _LaplacianAny: ...
@overload  # catch-all, return_diag: True (keyword), form: <any>
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: _Form = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianAny, onp.Array1D[Any]]: ...
@overload  # catch-all, return_diag: True (positional), form: <any>
def laplacian(
    csgraph: _ToAnyGraph,
    normed: bool,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: _Form = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianAny, onp.Array1D[Any]]: ...
