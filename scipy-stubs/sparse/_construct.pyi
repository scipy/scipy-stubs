from _typeshed import Incomplete
from collections.abc import Callable, Iterable, Sequence as Seq
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, overload, type_check_only

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.numpy.compat as npc
import optype.typing as opt

from ._base import _spbase, sparray
from ._bsr import bsr_array, bsr_matrix
from ._coo import coo_array, coo_matrix
from ._csr import csr_array, csr_matrix
from ._dia import dia_array, dia_matrix
from ._matrix import spmatrix
from ._typing import Numeric, SPFormat, ToShape2D, _CanStack, _CanStackAs

__all__ = [
    "block_array",
    "block_diag",
    "bmat",
    "diags",
    "diags_array",
    "eye",
    "eye_array",
    "hstack",
    "identity",
    "kron",
    "kronsum",
    "rand",
    "random",
    "random_array",
    "spdiags",
    "vstack",
]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Numeric, default=Any)
_SCT0 = TypeVar("_SCT0", bound=Numeric)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, *tuple[int, ...]], default=tuple[Any, ...])

_ToArray1D: TypeAlias = Seq[_SCT] | onp.CanArrayND[_SCT]
_ToArray2D: TypeAlias = Seq[Seq[_SCT | int] | onp.CanArrayND[_SCT]] | onp.CanArrayND[_SCT]
_ToSpMatrix: TypeAlias = spmatrix[_SCT] | _ToArray2D[_SCT]
_ToSparse: TypeAlias = _spbase[_SCT] | _ToArray2D[_SCT]

_SpBase: TypeAlias = _spbase[_SCT, _ShapeT] | Any
_SpMatrix: TypeAlias = spmatrix[_SCT] | Any
_SpArray: TypeAlias = sparray[_SCT, _ShapeT] | Any

_SpBase2D: TypeAlias = _SpBase[_SCT, tuple[int, int]]
_SpArray1D: TypeAlias = _SpArray[_SCT, tuple[int]]
_SpArray2D: TypeAlias = _SpArray[_SCT, tuple[int, int]]

_BSRArray: TypeAlias = bsr_array[_SCT]
_CSRArray: TypeAlias = csr_array[_SCT, tuple[int, int]]

_FmtBSR: TypeAlias = Literal["bsr"]
_FmtCOO: TypeAlias = Literal["coo"]
_FmtCSR: TypeAlias = Literal["csr"]
_FmtDIA: TypeAlias = Literal["dia"]
_FmtNonBSR: TypeAlias = Literal["coo", "csc", "csr", "dia", "dok", "lil"]
_FmtNonCOO: TypeAlias = Literal["bsr", "csc", "csr", "dia", "dok", "lil"]
_FmtNonCSR: TypeAlias = Literal["bsr", "coo", "csc", "dia", "dok", "lil"]
_FmtNonDIA: TypeAlias = Literal["bsr", "coo", "csc", "csr", "dok", "lil"]

_DataRVS: TypeAlias = Callable[[int], onp.ArrayND[Numeric]]

_ToBlocks: TypeAlias = Seq[Seq[_spbase[_SCT] | None]] | onp.ArrayND[np.object_]

@type_check_only
class _DataSampler(Protocol):
    def __call__(self, /, *, size: int) -> onp.ArrayND[Numeric]: ...

###

# NOTE: The `overload-overlap` mypy errors are false positives.
@overload  # diagonals: <known>, dtype: None = ..., format: {"dia", None} = ...
def diags_array(  # type: ignore[overload-overlap]
    diagonals: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: None = None,
) -> dia_array[_SCT]: ...
@overload  # diagonals: <known>, dtype: None = ..., format: <otherwise>
def diags_array(
    diagonals: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: None = None,
) -> _SpArray2D[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: int
def diags_array(
    diagonals: onp.ToFloat1D | onp.ToFloat2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: onp.AnyIntDType,
) -> dia_array[np.int_]: ...
@overload  # diagonals: <unknown>, format: <otherwise>, dtype: int
def diags_array(
    diagonals: onp.ToFloat1D | onp.ToFloat2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: onp.AnyIntDType,
) -> _SpArray2D[np.int_]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: float
def diags_array(
    diagonals: onp.ToFloat1D | onp.ToFloat2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: onp.AnyFloat64DType,
) -> dia_array[np.float64]: ...
@overload  # diagonals: <unknown>, format: <otherwise>, dtype: float
def diags_array(
    diagonals: onp.ToFloat1D | onp.ToFloat2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: onp.AnyFloat64DType,
) -> _SpArray2D[np.float64]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: complex
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: onp.AnyComplex128DType,
) -> dia_array[np.complex128]: ...
@overload  # diagonals: <unknown>, format: <otherwise>, dtype: complex
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: onp.AnyComplex128DType,
) -> _SpArray2D[np.complex128]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: onp.ToDType[_SCT],
) -> dia_array[_SCT]: ...
@overload  # diagonals: <unknown>, format: <otherwise>, dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: onp.ToDType[_SCT],
) -> _SpArray2D[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: npt.DTypeLike | None = None,
) -> dia_array: ...
@overload  # diagonals: <unknown>, format: <otherwise>, dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: npt.DTypeLike | None = None,
) -> _SpArray2D: ...

# NOTE: `diags_array` should be prefered over `diags`
@overload  # diagonals: <known>, format: {"dia", None} = ...
def diags(
    diagonals: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: onp.ToDType[_SCT] | None = None,
) -> dia_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: <otherwise> (positional)
def diags(
    diagonals: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    offsets: onp.ToInt | onp.ToInt1D,
    shape: ToShape2D | None,
    format: _FmtNonDIA,
    dtype: onp.ToDType[_SCT] | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # diagonals: <known>, format: <otherwise> (keyword)
def diags(
    diagonals: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    *,
    format: _FmtNonDIA,
    dtype: onp.ToDType[_SCT] | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <known> (positional)
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: ToShape2D | None,
    format: _FmtDIA | None,
    dtype: onp.ToDType[_SCT],
) -> dia_matrix[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <known> (keyword)
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    *,
    dtype: onp.ToDType[_SCT],
) -> dia_matrix[_SCT]: ...
@overload  # diagonals: <unknown>, format: <otherwise> (positional), dtype: <known>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: ToShape2D | None,
    format: _FmtNonDIA,
    dtype: onp.ToDType[_SCT],
) -> _SpMatrix[_SCT]: ...
@overload  # diagonals: <unknown>, format: <otherwise> (keyword), dtype: <known>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    *,
    format: _FmtNonDIA,
    dtype: onp.ToDType[_SCT],
) -> _SpMatrix[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: npt.DTypeLike | None = None,
) -> dia_matrix: ...
@overload  # diagonals: <unknown>, format: <otherwise> (positional), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: ToShape2D | None,
    format: _FmtNonDIA,
    dtype: npt.DTypeLike | None = None,
) -> _SpMatrix: ...
@overload  # diagonals: <unknown>, format: <otherwise> (keyword), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    *,
    format: _FmtNonDIA,
    dtype: npt.DTypeLike | None = None,
) -> _SpMatrix: ...

# NOTE: `diags_array` should be prefered over `spdiags`
@overload
def spdiags(
    data: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: onp.ToJustInt,
    n: onp.ToJustInt,
    format: _FmtDIA | None = None,
) -> dia_matrix[_SCT]: ...
@overload
def spdiags(
    data: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    format: _FmtDIA | None = None,
) -> dia_matrix[_SCT]: ...
@overload
def spdiags(
    data: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: onp.ToJustInt,
    n: onp.ToJustInt,
    format: _FmtNonDIA,
) -> _SpMatrix[_SCT]: ...
@overload
def spdiags(
    data: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    *,
    format: _FmtNonDIA,
) -> _SpMatrix[_SCT]: ...
@overload
def spdiags(
    data: onp.ToComplex1D | onp.ToComplex2D,
    diags: onp.ToInt | onp.ToInt1D,
    m: onp.ToJustInt,
    n: onp.ToJustInt,
    format: SPFormat | None = None,
) -> _SpMatrix: ...
@overload
def spdiags(
    data: onp.ToComplex1D | onp.ToComplex2D,
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    *,
    format: SPFormat | None = None,
) -> _SpMatrix: ...

#
@overload  # dtype like bool, format: None = ...
def identity(n: opt.AnyInt, dtype: onp.AnyBoolDType, format: _FmtDIA | None = None) -> dia_matrix[np.bool_]: ...
@overload  # dtype like int, format: None = ...
def identity(n: opt.AnyInt, dtype: onp.AnyIntDType, format: _FmtDIA | None = None) -> dia_matrix[np.int_]: ...
@overload  # dtype like float (default), format: None = ...
def identity(n: opt.AnyInt, dtype: onp.AnyFloat64DType = "d", format: _FmtDIA | None = None) -> dia_matrix[np.float64]: ...
@overload  # dtype like complex, format: None = ...
def identity(n: opt.AnyInt, dtype: onp.AnyComplex128DType, format: _FmtDIA | None = None) -> dia_matrix[np.complex128]: ...
@overload  # dtype like <known>, format: None = ...
def identity(n: opt.AnyInt, dtype: onp.ToDType[_SCT], format: _FmtDIA | None = None) -> dia_matrix[_SCT]: ...
@overload  # dtype like <unknown>, format: None = ...
def identity(n: opt.AnyInt, dtype: npt.DTypeLike, format: _FmtDIA | None = None) -> dia_matrix[Incomplete]: ...
@overload  # dtype like float, format: <given> (positional)
def identity(n: opt.AnyInt, dtype: onp.AnyFloat64DType, format: _FmtNonDIA) -> _SpMatrix[np.float64]: ...
@overload  # dtype like float (default), format: <given> (keyword)
def identity(n: opt.AnyInt, dtype: onp.AnyFloat64DType = "d", *, format: _FmtNonDIA) -> _SpMatrix[np.float64]: ...
@overload  # dtype like bool, format: <given>
def identity(n: opt.AnyInt, dtype: onp.AnyBoolDType, format: _FmtNonDIA) -> _SpMatrix[np.bool_]: ...
@overload  # dtype like int, format: <given>
def identity(n: opt.AnyInt, dtype: onp.AnyIntDType, format: _FmtNonDIA) -> _SpMatrix[np.int_]: ...
@overload  # dtype like complex, format: <given>
def identity(n: opt.AnyInt, dtype: onp.AnyComplex128DType, format: _FmtNonDIA) -> _SpMatrix[np.complex128]: ...
@overload  # dtype like <known>, format: <given>
def identity(n: opt.AnyInt, dtype: onp.ToDType[_SCT], format: _FmtNonDIA) -> _SpMatrix[_SCT]: ...
@overload  # dtype like <unknown>, fformat: <given>
def identity(n: opt.AnyInt, dtype: npt.DTypeLike, format: _FmtNonDIA) -> _SpMatrix[Incomplete]: ...

#
@overload  # dtype like bool, format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyBoolDType, format: _FmtDIA | None = None
) -> dia_array[np.bool_]: ...
@overload  # dtype like int, format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyIntDType, format: _FmtDIA | None = None
) -> dia_array[np.int_]: ...
@overload  # dtype like float (default), format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtDIA | None = None
) -> dia_array[np.float64]: ...
@overload  # dtype like complex, format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyComplex128DType, format: _FmtDIA | None = None
) -> dia_array[np.complex128]: ...
@overload  # dtype like <known>, format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.ToDType[_SCT], format: _FmtDIA | None = None
) -> dia_array[_SCT]: ...
@overload  # dtype like <unknown>, format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: npt.DTypeLike, format: _FmtDIA | None = None
) -> dia_array[Incomplete]: ...
@overload  # dtype like float (default), format: <given>
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtNonDIA
) -> _SpArray2D[np.float64]: ...
@overload  # dtype like bool, format: <given>
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyBoolDType, format: _FmtNonDIA
) -> _SpArray2D[np.bool_]: ...
@overload  # dtype like int, format: <given>
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyIntDType, format: _FmtNonDIA
) -> _SpArray2D[np.int_]: ...
@overload  # dtype like complex, format: <given>
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyComplex128DType, format: _FmtNonDIA
) -> _SpArray2D[np.complex128]: ...
@overload  # dtype like <known>, format: <given>
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.ToDType[_SCT], format: _FmtNonDIA
) -> _SpArray2D[_SCT]: ...
@overload  # dtype like <unknown>, fformat: <given>
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: npt.DTypeLike, format: _FmtNonDIA
) -> _SpArray2D[Incomplete]: ...

# NOTE: `eye_array` should be prefered over `eye`
@overload  # dtype like float (default), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtDIA | None = None
) -> dia_matrix[np.float64]: ...
@overload  # dtype like bool (positional), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None, k: int, dtype: onp.AnyBoolDType, format: _FmtDIA | None = None
) -> dia_matrix[np.bool_]: ...
@overload  # dtype like bool (keyword), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyBoolDType, format: _FmtDIA | None = None
) -> dia_matrix[np.bool_]: ...
@overload  # dtype like int (positional), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None, k: int, dtype: onp.AnyIntDType, format: _FmtDIA | None = None
) -> dia_matrix[np.int_]: ...
@overload  # dtype like int (keyword), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyIntDType, format: _FmtDIA | None = None
) -> dia_matrix[np.int_]: ...
@overload  # dtype like complex (positional), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None, k: int, dtype: onp.AnyComplex128DType, format: _FmtDIA | None = None
) -> dia_matrix[np.complex128]: ...
@overload  # dtype like complex (keyword), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyComplex128DType, format: _FmtDIA | None = None
) -> dia_matrix[np.complex128]: ...
@overload  # dtype like <known> (positional), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None, k: int, dtype: onp.ToDType[_SCT], format: _FmtDIA | None = None
) -> dia_matrix[_SCT]: ...
@overload  # dtype like <known> (keyword), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.ToDType[_SCT], format: _FmtDIA | None = None
) -> dia_matrix[_SCT]: ...
@overload  # dtype like <unknown> (positional), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None, k: int, dtype: npt.DTypeLike, format: _FmtDIA | None = None
) -> dia_matrix[Incomplete]: ...
@overload  # dtype like <unknown> (keyword), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: npt.DTypeLike, format: _FmtDIA | None = None
) -> dia_matrix[Incomplete]: ...
@overload  # dtype like float (default)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: SPFormat | None = None
) -> _SpMatrix[np.float64]: ...
@overload  # dtype like bool (positional)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None, k: int, dtype: onp.AnyBoolDType, format: SPFormat | None = None
) -> _SpMatrix[np.bool_]: ...
@overload  # dtype like bool (keyword)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyBoolDType, format: SPFormat | None = None
) -> _SpMatrix[np.bool_]: ...
@overload  # dtype like int (positional)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None, k: int, dtype: onp.AnyIntDType, format: SPFormat | None = None
) -> _SpMatrix[np.int_]: ...
@overload  # dtype like int (keyword)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyIntDType, format: SPFormat | None = None
) -> _SpMatrix[np.int_]: ...
@overload  # dtype like complex (positional)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None, k: int, dtype: onp.AnyComplex128DType, format: SPFormat | None = None
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype like complex (keyword)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyComplex128DType, format: SPFormat | None = None
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype like <known> (positional)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None, k: int, dtype: onp.ToDType[_SCT], format: SPFormat | None = None
) -> _SpMatrix[_SCT]: ...
@overload  # dtype like <known> (keyword)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.ToDType[_SCT], format: SPFormat | None = None
) -> _SpMatrix[_SCT]: ...
@overload  # dtype like <unknown> (positional)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None, k: int, dtype: npt.DTypeLike, format: SPFormat | None = None
) -> _SpMatrix[Incomplete]: ...
@overload  # dtype like <unknown> (keyword)
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: npt.DTypeLike, format: SPFormat | None = None
) -> _SpMatrix[Incomplete]: ...

#
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: {"bsr", None} = ...
def kron(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtBSR | None = None) -> bsr_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: <otherwise>
def kron(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtNonBSR) -> _SpMatrix[_SCT]: ...
@overload  # A: sparray, B: sparse, format: {"bsr", None} = ...
def kron(A: sparray[_SCT], B: _ToSparse[_SCT], format: _FmtBSR | None = None) -> _BSRArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: <otherwise>
def kron(A: sparray[_SCT], B: _ToSparse[_SCT], format: _FmtNonBSR) -> _SpArray2D[_SCT]: ...
@overload  # A: sparse, B: sparray, format: {"bsr", None} = ...
def kron(A: _ToSparse[_SCT], B: sparray[_SCT], format: _FmtBSR | None = None) -> _BSRArray[_SCT]: ...
@overload  # A: sparse, B: sparray, format: <otherwise>
def kron(A: _ToSparse[_SCT], B: sparray[_SCT], format: _FmtNonBSR) -> _SpArray2D[_SCT]: ...
@overload  # A: unknown array-like, B: unknown array-like  (catch-all)
def kron(A: onp.ToComplex2D, B: onp.ToComplex2D, format: SPFormat | None = None) -> _SpBase2D[Incomplete]: ...

# NOTE: The `overload-overlap` mypy errors are false positives.
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: {"csr", None} = ...
def kronsum(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtCSR | None = None) -> csr_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: <otherwise>
def kronsum(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtNonCSR) -> _SpMatrix[_SCT]: ...
@overload  # A: sparray, B: sparse, format: {"csr", None} = ...
def kronsum(A: sparray[_SCT], B: _ToSparse[_SCT], format: _FmtCSR | None = None) -> _CSRArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: <otherwise>
def kronsum(A: sparray[_SCT], B: _ToSparse[_SCT], format: _FmtNonCSR) -> _SpArray2D[_SCT]: ...
@overload  # A: sparse, B: sparray, format: {"csr", None} = ...
def kronsum(A: _ToSparse[_SCT], B: sparray[_SCT], format: _FmtCSR | None = None) -> _CSRArray[_SCT]: ...
@overload  # A: sparse, B: sparray, format: <otherwise>
def kronsum(A: _ToSparse[_SCT], B: sparray[_SCT], format: _FmtNonCSR) -> _SpArray2D[_SCT]: ...
@overload  # A: unknown array-like, B: unknown array-like  (catch-all)
def kronsum(A: onp.ToComplex2D, B: onp.ToComplex2D, format: SPFormat | None = None) -> _SpBase2D[Incomplete]: ...

# NOTE: keep in sync with `vstack`
@overload  # sparray, format: <default>, dtype: <default>
def hstack(blocks: Seq[_CanStack[_T]], format: None = None, dtype: None = None) -> _T: ...
@overload  # sparray, format: <default>, dtype: T (keyword)
def hstack(blocks: Seq[_CanStackAs[_SCT0, _T]], format: None = None, *, dtype: onp.ToDType[_SCT0]) -> _T: ...
@overload  # sparray, format: <default>, dtype: bool_ (keyword)
def hstack(blocks: Seq[_CanStackAs[np.bool_, _T]], format: None = None, *, dtype: onp.AnyBoolDType) -> _T: ...
@overload  # sparray, format: <default>, dtype: int_ (keyword)
def hstack(blocks: Seq[_CanStackAs[np.int_, _T]], format: None = None, *, dtype: onp.AnyIntDType) -> _T: ...
@overload  # sparray, format: <default>, dtype: float64 (keyword)
def hstack(blocks: Seq[_CanStackAs[np.float64, _T]], format: None = None, *, dtype: onp.AnyFloat64DType) -> _T: ...
@overload  # sparray, format: <default>, dtype: complex128 (keyword)
def hstack(blocks: Seq[_CanStackAs[np.complex128, _T]], format: None = None, *, dtype: onp.AnyComplex128DType) -> _T: ...
@overload  # sparray, format: <default>, dtype: complex128-like (keyword)
def hstack(blocks: Seq[_CanStackAs[Any, _T]], format: None = None, *, dtype: npt.DTypeLike) -> _T: ...
@overload  # TODO(jorenham): Support for `format=...`
def hstack(blocks: Seq[_spbase], format: SPFormat, dtype: npt.DTypeLike | None = None) -> Incomplete: ...

# NOTE: keep in sync with `vstack`
@overload  # sparray, format: <default>, dtype: <default>
def vstack(blocks: Seq[_CanStack[_T]], format: None = None, dtype: None = None) -> _T: ...
@overload  # sparray, format: <default>, dtype: T (keyword)
def vstack(blocks: Seq[_CanStackAs[_SCT0, _T]], format: None = None, *, dtype: onp.ToDType[_SCT0]) -> _T: ...
@overload  # sparray, format: <default>, dtype: bool_ (keyword)
def vstack(blocks: Seq[_CanStackAs[np.bool_, _T]], format: None = None, *, dtype: onp.AnyBoolDType) -> _T: ...
@overload  # sparray, format: <default>, dtype: int_ (keyword)
def vstack(blocks: Seq[_CanStackAs[np.int_, _T]], format: None = None, *, dtype: onp.AnyIntDType) -> _T: ...
@overload  # sparray, format: <default>, dtype: float64 (keyword)
def vstack(blocks: Seq[_CanStackAs[np.float64, _T]], format: None = None, *, dtype: onp.AnyFloat64DType) -> _T: ...
@overload  # sparray, format: <default>, dtype: complex128 (keyword)
def vstack(blocks: Seq[_CanStackAs[np.complex128, _T]], format: None = None, *, dtype: onp.AnyComplex128DType) -> _T: ...
@overload  # sparray, format: <default>, dtype: complex128-like (keyword)
def vstack(blocks: Seq[_CanStackAs[Any, _T]], format: None = None, *, dtype: npt.DTypeLike) -> _T: ...
@overload  # TODO(jorenham): Support for `format=...`
def vstack(blocks: Seq[_spbase], format: SPFormat, dtype: npt.DTypeLike | None = None) -> Incomplete: ...

_COOArray2D: TypeAlias = coo_array[_SCT, tuple[int, int]]

# TODO(jorenham): Use `_CanStack` here, which requires a way to map matrix types to array types.
@overload  # blocks: <known dtype>, format: <default>, dtype: <default>
def block_array(blocks: _ToBlocks[_SCT], *, format: _FmtCOO | None = None, dtype: None = None) -> _COOArray2D[_SCT]: ...
@overload  # blocks: <unknown dtype>, format: <default>, dtype: <known>
def block_array(blocks: _ToBlocks, *, format: _FmtCOO | None = None, dtype: onp.ToDType[_SCT]) -> _COOArray2D[_SCT]: ...
@overload  # blocks: <unknown dtype>, format: <default>, dtype: <unknown>
def block_array(blocks: _ToBlocks, *, format: _FmtCOO | None = None, dtype: npt.DTypeLike) -> _COOArray2D: ...
@overload  # blocks: <known dtype>, format: <otherwise>, dtype: <default>
def block_array(blocks: _ToBlocks[_SCT], *, format: _FmtNonCOO, dtype: None = None) -> _SpArray2D[_SCT]: ...
@overload  # blocks: <unknown dtype>, format: <otherwise>, dtype: <known>
def block_array(blocks: _ToBlocks, *, format: _FmtNonCOO, dtype: onp.ToDType[_SCT]) -> _SpArray2D[_SCT]: ...
@overload  # blocks: <unknown dtype>, format: <otherwise>, dtype: <unknown>
def block_array(blocks: _ToBlocks, *, format: _FmtNonCOO, dtype: npt.DTypeLike) -> _SpArray2D: ...

# TODO(jorenham): Use `_CanStack` here, which requires a way to map array types to matrix types.
@overload  # blocks: <array, known dtype>, format: <default>, dtype: <default>
def bmat(blocks: Seq[Seq[sparray[_SCT]]], format: _FmtCOO | None = None, dtype: None = None) -> _COOArray2D[_SCT]: ...
@overload  # blocks: <matrix, known dtype>, format: <default>, dtype: <default>
def bmat(blocks: Seq[Seq[spmatrix[_SCT]]], format: _FmtCOO | None = None, dtype: None = None) -> coo_matrix[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <default>, dtype: <known> (positional)
def bmat(blocks: _ToBlocks, format: _FmtCOO | None, dtype: onp.ToDType[_SCT]) -> _COOArray2D[_SCT] | coo_matrix[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <default>, dtype: <known> (keyword)
def bmat(
    blocks: _ToBlocks, format: _FmtCOO | None = None, *, dtype: onp.ToDType[_SCT]
) -> _COOArray2D[_SCT] | coo_matrix[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <default>, dtype: <unknown>
def bmat(
    blocks: _ToBlocks[_SCT], format: _FmtCOO | None = None, dtype: npt.DTypeLike | None = None
) -> _COOArray2D[_SCT] | coo_matrix[_SCT]: ...
@overload  # sparray, blocks: <array, known dtype>, format: <otherwise>, dtype: <default>
def bmat(blocks: Seq[Seq[sparray[_SCT]]], format: SPFormat, dtype: None = None) -> _SpArray2D[_SCT]: ...
@overload  # sparray, blocks: <matrix, known dtype>, format: <otherwise>, dtype: <default>
def bmat(blocks: Seq[Seq[spmatrix[_SCT]]], format: SPFormat, dtype: None = None) -> _SpMatrix[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <otherwise>, dtype: <known>
def bmat(blocks: _ToBlocks, format: SPFormat, dtype: onp.ToDType[_SCT]) -> _SpBase2D[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <otherwise>, dtype: <unknown>
def bmat(blocks: _ToBlocks, format: SPFormat, dtype: npt.DTypeLike) -> _SpBase2D: ...

# TODO(jorenham): Add support for non-COO formats.
@overload  # mats: <array, known dtype>
def block_diag(mats: Iterable[sparray[_SCT]], format: _FmtCOO | None = None, dtype: None = None) -> _COOArray2D[_SCT]: ...
@overload  # mats: <matrix, known dtype>
def block_diag(mats: Iterable[spmatrix[_SCT]], format: _FmtCOO | None = None, dtype: None = None) -> coo_matrix[_SCT]: ...
@overload  # mats: <unknown, known dtype>
def block_diag(
    mats: Iterable[_spbase[_SCT] | onp.ArrayND[_SCT]], format: _FmtCOO | None = None, dtype: None = None
) -> _COOArray2D[_SCT] | coo_matrix[_SCT]: ...
@overload  # mats: <array, unknown dtype>, dtype: <known>  (positional)
def block_diag(mats: Iterable[sparray], format: _FmtCOO | None, dtype: onp.ToDType[_SCT]) -> coo_array[_SCT, tuple[int, int]]: ...
@overload  # mats: <array, unknown dtype>, dtype: <known>  (keyword)
def block_diag(mats: Iterable[sparray], format: _FmtCOO | None = None, *, dtype: onp.ToDType[_SCT]) -> _COOArray2D[_SCT]: ...
@overload  # mats: <matrix, unknown dtype>, dtype: <known>  (positional)
def block_diag(
    mats: Iterable[spmatrix | onp.ArrayND[Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _FmtCOO | None,
    dtype: onp.ToDType[_SCT],
) -> coo_matrix[_SCT]: ...
@overload  # mats: <matrix, unknown dtype>, dtype: <known>  (keyword)
def block_diag(
    mats: Iterable[spmatrix | onp.ArrayND[Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _FmtCOO | None = None,
    *,
    dtype: onp.ToDType[_SCT],
) -> coo_matrix[_SCT]: ...
@overload  # mats: <unknown, unknown dtype>, dtype: <known>  (positional)
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _FmtCOO | None,
    dtype: onp.ToDType[_SCT],
) -> _COOArray2D[_SCT] | coo_matrix[_SCT]: ...
@overload  # mats: <unknown, unknown dtype>, dtype: <known>  (keyword)
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _FmtCOO | None = None,
    *,
    dtype: onp.ToDType[_SCT],
) -> _COOArray2D[_SCT] | coo_matrix[_SCT]: ...
@overload  # catch-all
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _FmtCOO | None = None,
    dtype: npt.DTypeLike | None = None,
) -> _COOArray2D[_SCT] | coo_matrix[Any]: ...
@overload  # catch-all
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: SPFormat | None = None,
    dtype: npt.DTypeLike | None = None,
) -> Incomplete: ...

#
@overload  # shape: T, dtype: <default>, format: <default>
def random_array(
    shape: _ShapeT,
    *,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    dtype: onp.AnyFloat64DType = None,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> coo_array[np.float64, _ShapeT]: ...
@overload  # shape: T, dtype: <known>, format: <default>
def random_array(
    shape: _ShapeT,
    *,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> coo_array[_SCT, _ShapeT]: ...
@overload  # shape: T, dtype: complex, format: <default>
def random_array(
    shape: _ShapeT,
    *,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> coo_array[np.complex128, _ShapeT]: ...
@overload  # shape: T, dtype: <unknown>, format: <default>
def random_array(
    shape: _ShapeT,
    *,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    dtype: npt.DTypeLike,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> coo_array[Any, _ShapeT]: ...
@overload  # shape: T, dtype: <default>
def random_array(
    shape: tuple[int],
    *,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    dtype: onp.AnyFloat64DType = None,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray1D[np.float64]: ...
@overload  # shape: 2d, dtype: <default>
def random_array(
    shape: tuple[int, int],
    *,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    dtype: onp.AnyFloat64DType = None,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray2D[np.float64]: ...
@overload  # shape: 1d, dtype: <known>
def random_array(
    shape: tuple[int],
    *,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray1D[_SCT]: ...
@overload  # shape: 2d, dtype: <known>
def random_array(
    shape: tuple[int, int],
    *,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray2D[_SCT]: ...
@overload  # shape: 1d, dtype: complex
def random_array(
    shape: tuple[int],
    *,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray1D[np.complex128]: ...
@overload  # shape: 2d, dtype: complex
def random_array(
    shape: tuple[int, int],
    *,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray2D[np.complex128]: ...
@overload  # shape: 1d, dtype: <unknown>
def random_array(
    shape: tuple[int],
    *,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    dtype: npt.DTypeLike,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray1D: ...
@overload  # shape: 2d, dtype: <unknown>
def random_array(
    shape: tuple[int, int],
    *,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    dtype: npt.DTypeLike,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray2D: ...

# NOTE: `random_array` should be prefered over `random`
@overload  # dtype: <default>, format: <default>
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    dtype: onp.AnyFloat64DType = None,
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix[np.float64]: ...
@overload  # dtype: <known> (positional), format: <default>
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating,
    format: _FmtCOO,
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix[_SCT]: ...
@overload  # dtype: <known> (keyword), format: <default>
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    *,
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix[_SCT]: ...
@overload  # dtype: complex (positional), format: <default>
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating,
    format: _FmtCOO,
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix[np.complex128]: ...
@overload  # dtype: complex (keyword), format: <default>
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    *,
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix[np.complex128]: ...
@overload  # dtype: <unknown>, format: <default>
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    dtype: npt.DTypeLike | None = None,
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix: ...
@overload  # dtype: <default>
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: SPFormat = ...,
    dtype: onp.AnyFloat64DType = None,
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[np.float64]: ...
@overload  # dtype: <known> (positional)
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating,
    format: SPFormat,
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # dtype: <known> (keyword)
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: SPFormat = ...,
    *,
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # dtype: complex (positional)
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating,
    format: SPFormat,
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype: complex (keyword)
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: SPFormat = ...,
    *,
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype: <unknown>
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: SPFormat = ...,
    dtype: npt.DTypeLike | None = None,
    rng: onp.random.ToRNG | None = None,
    data_rvs: _DataRVS | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix: ...

# NOTE: `random_array` should be prefered over `rand`
@overload  # dtype: <default>, format: <default>
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    dtype: onp.AnyFloat64DType = None,
    rng: onp.random.ToRNG | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix[np.float64]: ...
@overload  # dtype: <known> (positional), format: <default>
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating,
    format: _FmtCOO,
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix[_SCT]: ...
@overload  # dtype: <known> (keyword), format: <default>
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    *,
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix[_SCT]: ...
@overload  # dtype: complex (positional), format: <default>
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating,
    format: _FmtCOO,
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix[np.complex128]: ...
@overload  # dtype: complex (keyword), format: <default>
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    *,
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> coo_matrix[np.complex128]: ...
@overload  # dtype: <default>
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    dtype: onp.AnyFloat64DType = None,
    rng: onp.random.ToRNG | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[np.float64]: ...
@overload  # dtype: <known> (positional)
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating,
    format: SPFormat,
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # dtype: <known> (keyword)
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    *,
    dtype: onp.ToDType[_SCT],
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # dtype: complex (positional)
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating,
    format: SPFormat,
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype: complex (keyword)
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    *,
    dtype: onp.AnyComplex128DType,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype: <unknown>
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating = 0.01,
    format: SPFormat = "coo",
    dtype: npt.DTypeLike | None = None,
    rng: onp.random.ToRNG | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix: ...
