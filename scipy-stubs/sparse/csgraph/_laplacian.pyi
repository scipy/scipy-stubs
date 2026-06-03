from collections.abc import Callable
from typing import Literal, overload

import numpy.typing as npt
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.sparse._base import _spbase

###

type _LaplacianFunction = Callable[[onp.ToComplex2D], onp.Array2D[npc.number]]
type _LaplacianDiag = onp.Array1D[npc.number]
type _ToCSGraph = onp.ToComplex2D | _spbase
type _FunctionForm = Literal["function", "lo"]

###

# form in {"function", "lo"}: always returns callable regardless of input type
@overload
def laplacian(
    csgraph: _ToCSGraph,
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: _FunctionForm,
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> _LaplacianFunction: ...
@overload
def laplacian(
    csgraph: _ToCSGraph,
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: _FunctionForm,
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianFunction, _LaplacianDiag]: ...

# form="array" with dense input → dense output
@overload
def laplacian(
    csgraph: onp.ToComplex2D,
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> onp.Array2D[npc.number]: ...
@overload
def laplacian(
    csgraph: onp.ToComplex2D,
    normed: bool,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[onp.Array2D[npc.number], _LaplacianDiag]: ...
@overload
def laplacian(
    csgraph: onp.ToComplex2D,
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[onp.Array2D[npc.number], _LaplacianDiag]: ...

# form="array" with sparse input → sparse output
@overload
def laplacian(
    csgraph: _spbase,
    normed: bool = False,
    return_diag: onp.ToFalse = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> _spbase: ...
@overload
def laplacian(
    csgraph: _spbase,
    normed: bool,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_spbase, _LaplacianDiag]: ...
@overload
def laplacian(
    csgraph: _spbase,
    normed: bool = False,
    *,
    return_diag: onp.ToTrue,
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_spbase, _LaplacianDiag]: ...
