from _typeshed import Incomplete
from collections.abc import Callable, Iterable, Sequence as Seq
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, overload, type_check_only

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc
import optype.typing as opt
from optype.numpy._dtype_attr import f8_code

from ._base import _spbase, sparray
from ._bsr import bsr_array, bsr_matrix
from ._coo import coo_array, coo_matrix
from ._csc import csc_array, csc_matrix
from ._csr import csr_array, csr_matrix
from ._dia import dia_array, dia_matrix
from ._dok import dok_array, dok_matrix
from ._lil import lil_array, lil_matrix
from ._matrix import spmatrix
from ._typing import _CanStack, _CanStackAs, _Format, _ToShape2D

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

_Numeric: TypeAlias = npc.number | np.bool_
_AnyFloat64DTypeNotNone: TypeAlias = type[op.JustFloat] | onp.ToDType[npc.floating64] | f8_code

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=_Numeric, default=Any)
_SCT0 = TypeVar("_SCT0", bound=_Numeric)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, *tuple[int, ...]], default=tuple[Any, ...])

_ToArray1D: TypeAlias = onp.ToArray1D[complex, _SCT]
_ToArray2D: TypeAlias = onp.ToArray2D[complex, _SCT]
_ToSpMatrix: TypeAlias = spmatrix[_SCT] | _ToArray2D[_SCT]
_ToSparse2D: TypeAlias = _spbase[_SCT, tuple[int, int]] | _ToArray2D[_SCT]

_SpBase: TypeAlias = _spbase[_SCT, _ShapeT] | Any
_SpMatrix: TypeAlias = spmatrix[_SCT] | Any
_SpArray: TypeAlias = sparray[_SCT, _ShapeT] | Any

_SpBase2D: TypeAlias = _SpBase[_SCT, tuple[int, int]]
_SpArray1D: TypeAlias = _SpArray[_SCT, tuple[int]]
_SpArray2D: TypeAlias = _SpArray[_SCT, tuple[int, int]]

_BSRArray: TypeAlias = bsr_array[_SCT]
_COOArray2D: TypeAlias = coo_array[_SCT, tuple[int, int]]
_CSCArray: TypeAlias = csc_array[_SCT]
_CSRArray2D: TypeAlias = csr_array[_SCT, tuple[int, int]]
_DIAArray: TypeAlias = dia_array[_SCT]
_DOKArray2D: TypeAlias = dok_array[_SCT, tuple[int, int]]
_LILArray: TypeAlias = lil_array[_SCT]

_FmtBSR: TypeAlias = Literal["bsr"]
_FmtCOO: TypeAlias = Literal["coo"]
_FmtCSC: TypeAlias = Literal["csc"]
_FmtCSR: TypeAlias = Literal["csr"]
_FmtDIA: TypeAlias = Literal["dia"]
_FmtDOK: TypeAlias = Literal["dok"]
_FmtLIL: TypeAlias = Literal["lil"]
_FmtNonCOO: TypeAlias = Literal["bsr", "csc", "csr", "dia", "dok", "lil"]

_Diag: TypeAlias = _ToArray1D[_SCT] | _ToArray2D[_SCT]
_Offsets: TypeAlias = onp.ToInt | onp.ToInt1D

_DataRVS: TypeAlias = Callable[[int], onp.ArrayND[_Numeric]]

_ToBlocks: TypeAlias = Seq[Seq[_spbase[_SCT] | None]] | onp.ArrayND[np.object_]

@type_check_only
class _DataSampler(Protocol):
    def __call__(self, /, *, size: int) -> onp.ArrayND[_Numeric]: ...

###

@overload  # diagonals: <known>, dtype: None = ..., format: {"dia", None} = ...
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: None = None,
) -> _DIAArray[_SCT]: ...
@overload  # diagonals: <known>, format: {"dia", None} = ..., dtype: bool-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: onp.AnyBoolDType,
) -> _DIAArray[np.bool_]: ...
@overload  # diagonals: <known>, format: {"dia", None} = ..., dtype: int-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: onp.AnyIntDType,
) -> _DIAArray[np.int_]: ...
@overload  # diagonals: <known>, format: {"dia", None} = ..., dtype: float64-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: _AnyFloat64DTypeNotNone,
) -> _DIAArray[np.float64]: ...
@overload  # diagonals: <known>, format: {"dia", None} = ..., dtype: complex128-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: onp.AnyComplex128DType,
) -> _DIAArray[np.complex128]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: onp.ToDType[_SCT],
) -> _DIAArray[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: npt.DTypeLike,
) -> _DIAArray: ...

# BSR
@overload  # diagonals: <known>, format: "bsr", dtype: None
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtBSR, dtype: None = None
) -> _BSRArray[_SCT]: ...
@overload  # diagonals: <known>, format: "bsr", dtype: bool-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtBSR, dtype: onp.AnyBoolDType
) -> _BSRArray[np.bool_]: ...
@overload  # diagonals: <known>, format: "bsr", dtype: int-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtBSR, dtype: onp.AnyIntDType
) -> _BSRArray[np.int_]: ...
@overload  # diagonals: <known>, format: "bsr", dtype: float64-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtBSR,
    dtype: _AnyFloat64DTypeNotNone,
) -> _BSRArray[np.float64]: ...
@overload  # diagonals: <known>, format: "bsr", dtype: complex128-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtBSR,
    dtype: onp.AnyComplex128DType,
) -> _BSRArray[np.complex128]: ...
@overload  # diagonals: <unknown>, format: "bsr", dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtBSR,
    dtype: onp.ToDType[_SCT],
) -> _BSRArray[_SCT]: ...
@overload  # diagonals: <unknown>, format: "bsr", dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtBSR,
    dtype: npt.DTypeLike,
) -> _BSRArray: ...

# COO
@overload  # diagonals: <known>, format: "coo", dtype: None
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtCOO, dtype: None = None
) -> _COOArray2D[_SCT]: ...
@overload  # diagonals: <known>, format: "coo", dtype: bool-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtCOO, dtype: onp.AnyBoolDType
) -> _COOArray2D[np.bool_]: ...
@overload  # diagonals: <known>, format: "coo", dtype: int-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtCOO, dtype: onp.AnyIntDType
) -> _COOArray2D[np.int_]: ...
@overload  # diagonals: <known>, format: "coo", dtype: float64-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCOO,
    dtype: _AnyFloat64DTypeNotNone,
) -> _COOArray2D[np.float64]: ...
@overload  # diagonals: <known>, format: "coo", dtype: complex128-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCOO,
    dtype: onp.AnyComplex128DType,
) -> _COOArray2D[np.complex128]: ...
@overload  # diagonals: <unknown>, format: "coo", dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCOO,
    dtype: onp.ToDType[_SCT],
) -> _COOArray2D[_SCT]: ...
@overload  # diagonals: <unknown>, format: "coo", dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCOO,
    dtype: npt.DTypeLike,
) -> _COOArray2D: ...

# CSC
@overload  # diagonals: <known>, format: "csc", dtype: None
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtCSC, dtype: None = None
) -> _CSCArray[_SCT]: ...
@overload  # diagonals: <known>, format: "csc", dtype: bool-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtCSC, dtype: onp.AnyBoolDType
) -> _CSCArray[np.bool_]: ...
@overload  # diagonals: <known>, format: "csc", dtype: int-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtCSC, dtype: onp.AnyIntDType
) -> _CSCArray[np.int_]: ...
@overload  # diagonals: <known>, format: "csc", dtype: float64-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCSC,
    dtype: _AnyFloat64DTypeNotNone,
) -> _CSCArray[np.float64]: ...
@overload  # diagonals: <known>, format: "csc", dtype: complex128-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCSC,
    dtype: onp.AnyComplex128DType,
) -> _CSCArray[np.complex128]: ...
@overload  # diagonals: <unknown>, format: "csc", dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCSC,
    dtype: onp.ToDType[_SCT],
) -> _CSCArray[_SCT]: ...
@overload  # diagonals: <unknown>, format: "csc", dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCSC,
    dtype: npt.DTypeLike,
) -> _CSCArray: ...

# CSR
@overload  # diagonals: <known>, format: "csr", dtype: None
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtCSR, dtype: None = None
) -> _CSRArray2D[_SCT]: ...
@overload  # diagonals: <known>, format: "csr", dtype: bool-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtCSR, dtype: onp.AnyBoolDType
) -> _CSRArray2D[np.bool_]: ...
@overload  # diagonals: <known>, format: "csr", dtype: int-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtCSR, dtype: onp.AnyIntDType
) -> _CSRArray2D[np.int_]: ...
@overload  # diagonals: <known>, format: "csr", dtype: float64-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCSR,
    dtype: _AnyFloat64DTypeNotNone,
) -> _CSRArray2D[np.float64]: ...
@overload  # diagonals: <known>, format: "csr", dtype: complex128-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCSR,
    dtype: onp.AnyComplex128DType,
) -> _CSRArray2D[np.complex128]: ...
@overload  # diagonals: <unknown>, format: "csr", dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCSR,
    dtype: onp.ToDType[_SCT],
) -> _CSRArray2D[_SCT]: ...
@overload  # diagonals: <unknown>, format: "csr", dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtCSR,
    dtype: npt.DTypeLike,
) -> _CSRArray2D: ...

# DOK
@overload  # diagonals: <known>, format: "dok", dtype: None
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtDOK, dtype: None = None
) -> _DOKArray2D[_SCT]: ...
@overload  # diagonals: <known>, format: "dok", dtype: bool-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtDOK, dtype: onp.AnyBoolDType
) -> _DOKArray2D[np.bool_]: ...
@overload  # diagonals: <known>, format: "dok", dtype: int-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtDOK, dtype: onp.AnyIntDType
) -> _DOKArray2D[np.int_]: ...
@overload  # diagonals: <known>, format: "dok", dtype: float64-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDOK,
    dtype: _AnyFloat64DTypeNotNone,
) -> _DOKArray2D[np.float64]: ...
@overload  # diagonals: <known>, format: "dok", dtype: complex128-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDOK,
    dtype: onp.AnyComplex128DType,
) -> _DOKArray2D[np.complex128]: ...
@overload  # diagonals: <unknown>, format: "dok", dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDOK,
    dtype: onp.ToDType[_SCT],
) -> _DOKArray2D[_SCT]: ...
@overload  # diagonals: <unknown>, format: "dok", dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDOK,
    dtype: npt.DTypeLike,
) -> _DOKArray2D: ...

# LIL
@overload  # diagonals: <known>, format: "lil", dtype: None
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtLIL, dtype: None = None
) -> _LILArray[_SCT]: ...
@overload  # diagonals: <known>, format: "lil", dtype: bool-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtLIL, dtype: onp.AnyBoolDType
) -> _LILArray[np.bool_]: ...
@overload  # diagonals: <known>, format: "lil", dtype: int-like
def diags_array(
    diagonals: _Diag[_SCT], /, *, offsets: _Offsets = 0, shape: _ToShape2D | None = None, format: _FmtLIL, dtype: onp.AnyIntDType
) -> _LILArray[np.int_]: ...
@overload  # diagonals: <known>, format: "lil", dtype: float64-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtLIL,
    dtype: _AnyFloat64DTypeNotNone,
) -> _LILArray[np.float64]: ...
@overload  # diagonals: <known>, format: "lil", dtype: complex128-like
def diags_array(
    diagonals: _Diag[_SCT],
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtLIL,
    dtype: onp.AnyComplex128DType,
) -> _LILArray[np.complex128]: ...
@overload  # diagonals: <unknown>, format: "lil", dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtLIL,
    dtype: onp.ToDType[_SCT],
) -> _LILArray[_SCT]: ...
@overload  # diagonals: <unknown>, format: "lil", dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: _Offsets = 0,
    shape: _ToShape2D | None = None,
    format: _FmtLIL,
    dtype: npt.DTypeLike,
) -> _LILArray: ...

# NOTE: `diags_array` should be prefered over `diags`
@overload  # diagonals: <known>, format: {"dia", None} = ...
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: onp.ToDType[_SCT] | None = None,
) -> dia_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "bsr" (positional)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtBSR,
    dtype: onp.ToDType[_SCT] | None = None,
) -> bsr_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "bsr" (keyword)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtBSR,
    dtype: onp.ToDType[_SCT] | None = None,
) -> bsr_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "coo" (positional)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtCOO,
    dtype: onp.ToDType[_SCT] | None = None,
) -> coo_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "coo" (keyword)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtCOO,
    dtype: onp.ToDType[_SCT] | None = None,
) -> coo_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "csr" (positional)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtCSR,
    dtype: onp.ToDType[_SCT] | None = None,
) -> csr_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "csr" (keyword)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtCSR,
    dtype: onp.ToDType[_SCT] | None = None,
) -> csr_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "csc" (positional)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtCSC,
    dtype: onp.ToDType[_SCT] | None = None,
) -> csc_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "csc" (keyword)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtCSC,
    dtype: onp.ToDType[_SCT] | None = None,
) -> csc_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "dok" (positional)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtDOK,
    dtype: onp.ToDType[_SCT] | None = None,
) -> dok_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "dok" (keyword)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtDOK,
    dtype: onp.ToDType[_SCT] | None = None,
) -> dok_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "lil" (positional)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtLIL,
    dtype: onp.ToDType[_SCT] | None = None,
) -> lil_matrix[_SCT]: ...
@overload  # diagonals: <known>, format: "lil" (keyword)
def diags(
    diagonals: _Diag[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtLIL,
    dtype: onp.ToDType[_SCT] | None = None,
) -> lil_matrix[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: npt.DTypeLike | None = None,
) -> dia_matrix: ...
@overload  # diagonals: <unknown>, format: "bsr" (positional), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtBSR,
    dtype: npt.DTypeLike | None = None,
) -> bsr_matrix: ...
@overload  # diagonals: <unknown>, format: "bsr" (keyword), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtBSR,
    dtype: npt.DTypeLike | None = None,
) -> bsr_matrix: ...
@overload  # diagonals: <unknown>, format: "coo" (positional), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtCOO,
    dtype: npt.DTypeLike | None = None,
) -> coo_matrix: ...
@overload  # diagonals: <unknown>, format: "coo" (keyword), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtCOO,
    dtype: npt.DTypeLike | None = None,
) -> coo_matrix: ...
@overload  # diagonals: <unknown>, format: "csr" (positional), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtCSR,
    dtype: npt.DTypeLike | None = None,
) -> csr_matrix: ...
@overload  # diagonals: <unknown>, format: "csr" (keyword), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtCSR,
    dtype: npt.DTypeLike | None = None,
) -> csr_matrix: ...
@overload  # diagonals: <unknown>, format: "csc" (positional), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtCSC,
    dtype: npt.DTypeLike | None = None,
) -> csc_matrix: ...
@overload  # diagonals: <unknown>, format: "csc" (keyword), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtCSC,
    dtype: npt.DTypeLike | None = None,
) -> csc_matrix: ...
@overload  # diagonals: <unknown>, format: "dok" (positional), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtDOK,
    dtype: npt.DTypeLike | None = None,
) -> dok_matrix: ...
@overload  # diagonals: <unknown>, format: "dok" (keyword), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtDOK,
    dtype: npt.DTypeLike | None = None,
) -> dok_matrix: ...
@overload  # diagonals: <unknown>, format: "lil" (positional), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: _ToShape2D | None,
    format: _FmtLIL,
    dtype: npt.DTypeLike | None = None,
) -> lil_matrix: ...
@overload  # diagonals: <unknown>, format: "lil" (keyword), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: _ToShape2D | None = None,
    *,
    format: _FmtLIL,
    dtype: npt.DTypeLike | None = None,
) -> lil_matrix: ...

# NOTE: `diags_array` should be prefered over `spdiags`
@overload
def spdiags(
    data: _Diag[_SCT], diags: onp.ToInt | onp.ToInt1D, m: onp.ToJustInt, n: onp.ToJustInt, format: _FmtDIA | None = None
) -> dia_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    format: _FmtDIA | None = None,
) -> dia_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT], diags: onp.ToInt | onp.ToInt1D, m: onp.ToJustInt, n: onp.ToJustInt, format: _FmtBSR
) -> bsr_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    *,
    format: _FmtBSR,
) -> bsr_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT], diags: onp.ToInt | onp.ToInt1D, m: onp.ToJustInt, n: onp.ToJustInt, format: _FmtCOO
) -> coo_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    *,
    format: _FmtCOO,
) -> coo_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT], diags: onp.ToInt | onp.ToInt1D, m: onp.ToJustInt, n: onp.ToJustInt, format: _FmtCSR
) -> csr_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    *,
    format: _FmtCSR,
) -> csr_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT], diags: onp.ToInt | onp.ToInt1D, m: onp.ToJustInt, n: onp.ToJustInt, format: _FmtCSC
) -> csc_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    *,
    format: _FmtCSC,
) -> csc_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT], diags: onp.ToInt | onp.ToInt1D, m: onp.ToJustInt, n: onp.ToJustInt, format: _FmtDOK
) -> dok_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    *,
    format: _FmtDOK,
) -> dok_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT], diags: onp.ToInt | onp.ToInt1D, m: onp.ToJustInt, n: onp.ToJustInt, format: _FmtLIL
) -> lil_matrix[_SCT]: ...
@overload
def spdiags(
    data: _Diag[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    *,
    format: _FmtLIL,
) -> lil_matrix[_SCT]: ...

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
@overload  # dtype like float (default), format: "bsr" (keyword)
def identity(n: opt.AnyInt, dtype: onp.AnyFloat64DType = "d", *, format: _FmtBSR) -> bsr_matrix[np.float64]: ...
@overload  # dtype like float (default), format: "coo" (keyword)
def identity(n: opt.AnyInt, dtype: onp.AnyFloat64DType = "d", *, format: _FmtCOO) -> coo_matrix[np.float64]: ...
@overload  # dtype like float (default), format: "csc" (keyword)
def identity(n: opt.AnyInt, dtype: onp.AnyFloat64DType = "d", *, format: _FmtCSC) -> csc_matrix[np.float64]: ...
@overload  # dtype like float (default), format: "csr" (keyword)
def identity(n: opt.AnyInt, dtype: onp.AnyFloat64DType = "d", *, format: _FmtCSR) -> csr_matrix[np.float64]: ...
@overload  # dtype like float (default), format: "dok" (keyword)
def identity(n: opt.AnyInt, dtype: onp.AnyFloat64DType = "d", *, format: _FmtDOK) -> dok_matrix[np.float64]: ...
@overload  # dtype like float (default), format: "lil" (keyword)
def identity(n: opt.AnyInt, dtype: onp.AnyFloat64DType = "d", *, format: _FmtLIL) -> lil_matrix[np.float64]: ...
@overload  # dtype like bool, format: "bsr"
def identity(n: opt.AnyInt, dtype: onp.AnyBoolDType, format: _FmtBSR) -> bsr_matrix[np.bool_]: ...
@overload  # dtype like bool, format: "coo"
def identity(n: opt.AnyInt, dtype: onp.AnyBoolDType, format: _FmtCOO) -> coo_matrix[np.bool_]: ...
@overload  # dtype like bool, format: "csc"
def identity(n: opt.AnyInt, dtype: onp.AnyBoolDType, format: _FmtCSC) -> csc_matrix[np.bool_]: ...
@overload  # dtype like bool, format: "csr"
def identity(n: opt.AnyInt, dtype: onp.AnyBoolDType, format: _FmtCSR) -> csr_matrix[np.bool_]: ...
@overload  # dtype like bool, format: "dok"
def identity(n: opt.AnyInt, dtype: onp.AnyBoolDType, format: _FmtDOK) -> dok_matrix[np.bool_]: ...
@overload  # dtype like bool, format: "lil"
def identity(n: opt.AnyInt, dtype: onp.AnyBoolDType, format: _FmtLIL) -> lil_matrix[np.bool_]: ...
@overload  # dtype like int, format: "bsr"
def identity(n: opt.AnyInt, dtype: onp.AnyIntDType, format: _FmtBSR) -> bsr_matrix[np.int_]: ...
@overload  # dtype like int, format: "coo"
def identity(n: opt.AnyInt, dtype: onp.AnyIntDType, format: _FmtCOO) -> coo_matrix[np.int_]: ...
@overload  # dtype like int, format: "csc"
def identity(n: opt.AnyInt, dtype: onp.AnyIntDType, format: _FmtCSC) -> csc_matrix[np.int_]: ...
@overload  # dtype like int, format: "csr"
def identity(n: opt.AnyInt, dtype: onp.AnyIntDType, format: _FmtCSR) -> csr_matrix[np.int_]: ...
@overload  # dtype like int, format: "dok"
def identity(n: opt.AnyInt, dtype: onp.AnyIntDType, format: _FmtDOK) -> dok_matrix[np.int_]: ...
@overload  # dtype like int, format: "lil"
def identity(n: opt.AnyInt, dtype: onp.AnyIntDType, format: _FmtLIL) -> lil_matrix[np.int_]: ...
@overload  # dtype like complex, format: "bsr"
def identity(n: opt.AnyInt, dtype: onp.AnyComplex128DType, format: _FmtBSR) -> bsr_matrix[np.complex128]: ...
@overload  # dtype like complex, format: "coo"
def identity(n: opt.AnyInt, dtype: onp.AnyComplex128DType, format: _FmtCOO) -> coo_matrix[np.complex128]: ...
@overload  # dtype like complex, format: "csc"
def identity(n: opt.AnyInt, dtype: onp.AnyComplex128DType, format: _FmtCSC) -> csc_matrix[np.complex128]: ...
@overload  # dtype like complex, format: "csr"
def identity(n: opt.AnyInt, dtype: onp.AnyComplex128DType, format: _FmtCSR) -> csr_matrix[np.complex128]: ...
@overload  # dtype like complex, format: "dok"
def identity(n: opt.AnyInt, dtype: onp.AnyComplex128DType, format: _FmtDOK) -> dok_matrix[np.complex128]: ...
@overload  # dtype like complex, format: "lil"
def identity(n: opt.AnyInt, dtype: onp.AnyComplex128DType, format: _FmtLIL) -> lil_matrix[np.complex128]: ...
@overload  # dtype like <known>, format: "bsr"
def identity(n: opt.AnyInt, dtype: onp.ToDType[_SCT], format: _FmtBSR) -> bsr_matrix[_SCT]: ...
@overload  # dtype like <known>, format: "coo"
def identity(n: opt.AnyInt, dtype: onp.ToDType[_SCT], format: _FmtCOO) -> coo_matrix[_SCT]: ...
@overload  # dtype like <known>, format: "csc"
def identity(n: opt.AnyInt, dtype: onp.ToDType[_SCT], format: _FmtCSC) -> csc_matrix[_SCT]: ...
@overload  # dtype like <known>, format: "csr"
def identity(n: opt.AnyInt, dtype: onp.ToDType[_SCT], format: _FmtCSR) -> csr_matrix[_SCT]: ...
@overload  # dtype like <known>, format: "dok"
def identity(n: opt.AnyInt, dtype: onp.ToDType[_SCT], format: _FmtDOK) -> dok_matrix[_SCT]: ...
@overload  # dtype like <known>, format: "lil"
def identity(n: opt.AnyInt, dtype: onp.ToDType[_SCT], format: _FmtLIL) -> lil_matrix[_SCT]: ...
@overload  # dtype like <unknown>, format: "bsr"
def identity(n: opt.AnyInt, dtype: npt.DTypeLike, format: _FmtBSR) -> bsr_matrix[Incomplete]: ...
@overload  # dtype like <unknown>, format: "coo"
def identity(n: opt.AnyInt, dtype: npt.DTypeLike, format: _FmtCOO) -> coo_matrix[Incomplete]: ...
@overload  # dtype like <unknown>, format: "csc"
def identity(n: opt.AnyInt, dtype: npt.DTypeLike, format: _FmtCSC) -> csc_matrix[Incomplete]: ...
@overload  # dtype like <unknown>, format: "csr"
def identity(n: opt.AnyInt, dtype: npt.DTypeLike, format: _FmtCSR) -> csr_matrix[Incomplete]: ...
@overload  # dtype like <unknown>, format: "dok"
def identity(n: opt.AnyInt, dtype: npt.DTypeLike, format: _FmtDOK) -> dok_matrix[Incomplete]: ...
@overload  # dtype like <unknown>, format: "lil"
def identity(n: opt.AnyInt, dtype: npt.DTypeLike, format: _FmtLIL) -> lil_matrix[Incomplete]: ...

#
@overload  # dtype like bool, format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyBoolDType, format: _FmtDIA | None = None
) -> _DIAArray[np.bool_]: ...
@overload  # dtype like int, format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyIntDType, format: _FmtDIA | None = None
) -> _DIAArray[np.int_]: ...
@overload  # dtype like float (default), format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtDIA | None = None
) -> _DIAArray[np.float64]: ...
@overload  # dtype like complex, format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyComplex128DType, format: _FmtDIA | None = None
) -> _DIAArray[np.complex128]: ...
@overload  # dtype like <known>, format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.ToDType[_SCT], format: _FmtDIA | None = None
) -> _DIAArray[_SCT]: ...
@overload  # dtype like <unknown>, format: None = ...
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: npt.DTypeLike, format: _FmtDIA | None = None
) -> _DIAArray[Incomplete]: ...
@overload  # dtype like float (default), format: "bsr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtBSR
) -> _BSRArray[np.float64]: ...
@overload  # dtype like float (default), format: "coo"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtCOO
) -> _COOArray2D[np.float64]: ...
@overload  # dtype like float (default), format: "csc"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtCSC
) -> _CSCArray[np.float64]: ...
@overload  # dtype like float (default), format: "csr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtCSR
) -> _CSRArray2D[np.float64]: ...
@overload  # dtype like float (default), format: "dok"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtDOK
) -> _DOKArray2D[np.float64]: ...
@overload  # dtype like float (default), format: "lil"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtLIL
) -> _LILArray[np.float64]: ...
@overload  # dtype like bool, format: "bsr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyBoolDType, format: _FmtBSR
) -> _BSRArray[np.bool_]: ...
@overload  # dtype like bool, format: "coo"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyBoolDType, format: _FmtCOO
) -> _COOArray2D[np.bool_]: ...
@overload  # dtype like bool, format: "csc"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyBoolDType, format: _FmtCSC
) -> _CSCArray[np.bool_]: ...
@overload  # dtype like bool, format: "csr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyBoolDType, format: _FmtCSR
) -> _CSRArray2D[np.bool_]: ...
@overload  # dtype like bool, format: "dok"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyBoolDType, format: _FmtDOK
) -> _DOKArray2D[np.bool_]: ...
@overload  # dtype like bool, format: "lil"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyBoolDType, format: _FmtLIL
) -> _LILArray[np.bool_]: ...
@overload  # dtype like int, format: "bsr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyIntDType, format: _FmtBSR
) -> _BSRArray[np.int_]: ...
@overload  # dtype like int, format: "coo"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyIntDType, format: _FmtCOO
) -> _COOArray2D[np.int_]: ...
@overload  # dtype like int, format: "csc"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyIntDType, format: _FmtCSC
) -> _CSCArray[np.int_]: ...
@overload  # dtype like int, format: "csr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyIntDType, format: _FmtCSR
) -> _CSRArray2D[np.int_]: ...
@overload  # dtype like int, format: "dok"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyIntDType, format: _FmtDOK
) -> _DOKArray2D[np.int_]: ...
@overload  # dtype like int, format: "lil"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyIntDType, format: _FmtLIL
) -> _LILArray[np.int_]: ...
@overload  # dtype like complex, format: "bsr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyComplex128DType, format: _FmtBSR
) -> _BSRArray[np.complex128]: ...
@overload  # dtype like complex, format: "coo"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyComplex128DType, format: _FmtCOO
) -> _COOArray2D[np.complex128]: ...
@overload  # dtype like complex, format: "csc"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyComplex128DType, format: _FmtCSC
) -> _CSCArray[np.complex128]: ...
@overload  # dtype like complex, format: "csr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyComplex128DType, format: _FmtCSR
) -> _CSRArray2D[np.complex128]: ...
@overload  # dtype like complex, format: "dok"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyComplex128DType, format: _FmtDOK
) -> _DOKArray2D[np.complex128]: ...
@overload  # dtype like complex, format: "lil"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.AnyComplex128DType, format: _FmtLIL
) -> _LILArray[np.complex128]: ...
@overload  # dtype like <known>, format: "bsr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.ToDType[_SCT], format: _FmtBSR
) -> _BSRArray[_SCT]: ...
@overload  # dtype like <known>, format: "coo"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.ToDType[_SCT], format: _FmtCOO
) -> _COOArray2D[_SCT]: ...
@overload  # dtype like <known>, format: "csc"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.ToDType[_SCT], format: _FmtCSC
) -> _CSCArray[_SCT]: ...
@overload  # dtype like <known>, format: "csr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.ToDType[_SCT], format: _FmtCSR
) -> _CSRArray2D[_SCT]: ...
@overload  # dtype like <known>, format: "dok"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.ToDType[_SCT], format: _FmtDOK
) -> _DOKArray2D[_SCT]: ...
@overload  # dtype like <known>, format: "lil"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: onp.ToDType[_SCT], format: _FmtLIL
) -> _LILArray[_SCT]: ...
@overload  # dtype like <unknown>, format: "bsr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: npt.DTypeLike, format: _FmtBSR
) -> _BSRArray[Incomplete]: ...
@overload  # dtype like <unknown>, format: "coo"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: npt.DTypeLike, format: _FmtCOO
) -> _COOArray2D[Incomplete]: ...
@overload  # dtype like <unknown>, format: "csc"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: npt.DTypeLike, format: _FmtCSC
) -> _CSCArray[Incomplete]: ...
@overload  # dtype like <unknown>, format: "csr"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: npt.DTypeLike, format: _FmtCSR
) -> _CSRArray2D[Incomplete]: ...
@overload  # dtype like <unknown>, format: "dok"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: npt.DTypeLike, format: _FmtDOK
) -> _DOKArray2D[Incomplete]: ...
@overload  # dtype like <unknown>, format: "lil"
def eye_array(
    m: opt.AnyInt, n: opt.AnyInt | None = None, *, k: int = 0, dtype: npt.DTypeLike, format: _FmtLIL
) -> _LILArray[Incomplete]: ...

# NOTE: `eye_array` should be prefered over `eye`
@overload  # dtype like float (default), default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, dtype: onp.AnyFloat64DType = ..., format: _FmtDIA | None = None
) -> dia_matrix[np.float64]: ...
@overload  # dtype like bool, default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyBoolDType, format: _FmtDIA | None = None
) -> dia_matrix[np.bool_]: ...
@overload  # dtype like int, default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyIntDType, format: _FmtDIA | None = None
) -> dia_matrix[np.int_]: ...
@overload  # dtype like complex, default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyComplex128DType, format: _FmtDIA | None = None
) -> dia_matrix[np.complex128]: ...
@overload  # dtype like <known>, default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.ToDType[_SCT], format: _FmtDIA | None = None
) -> dia_matrix[_SCT]: ...
@overload  # dtype like <unknown>, default format
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: npt.DTypeLike, format: _FmtDIA | None = None
) -> dia_matrix[Incomplete]: ...
@overload  # dtype like float (default), format: "bsr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, dtype: onp.AnyFloat64DType = ..., *, format: _FmtBSR
) -> bsr_matrix[np.float64]: ...
@overload  # dtype like bool, format: "bsr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyBoolDType, format: _FmtBSR
) -> bsr_matrix[np.bool_]: ...
@overload  # dtype like int, format: "bsr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyIntDType, format: _FmtBSR
) -> bsr_matrix[np.int_]: ...
@overload  # dtype like complex, format: "bsr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyComplex128DType, format: _FmtBSR
) -> bsr_matrix[np.complex128]: ...
@overload  # dtype like <known>, format: "bsr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.ToDType[_SCT], format: _FmtBSR
) -> bsr_matrix[_SCT]: ...
@overload  # dtype like <unknown>, format: "bsr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: npt.DTypeLike, format: _FmtBSR
) -> bsr_matrix[Incomplete]: ...
@overload  # dtype like float (default), format: "coo"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, dtype: onp.AnyFloat64DType = ..., *, format: _FmtCOO
) -> coo_matrix[np.float64]: ...
@overload  # dtype like bool, format: "coo"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyBoolDType, format: _FmtCOO
) -> coo_matrix[np.bool_]: ...
@overload  # dtype like int, format: "coo"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyIntDType, format: _FmtCOO
) -> coo_matrix[np.int_]: ...
@overload  # dtype like complex, format: "coo"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyComplex128DType, format: _FmtCOO
) -> coo_matrix[np.complex128]: ...
@overload  # dtype like <known>, format: "coo"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.ToDType[_SCT], format: _FmtCOO
) -> coo_matrix[_SCT]: ...
@overload  # dtype like <unknown>, format: "coo"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: npt.DTypeLike, format: _FmtCOO
) -> coo_matrix[Incomplete]: ...
@overload  # dtype like float (default), format: "csc"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, dtype: onp.AnyFloat64DType = ..., *, format: _FmtCSC
) -> csc_matrix[np.float64]: ...
@overload  # dtype like bool, format: "csc"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyBoolDType, format: _FmtCSC
) -> csc_matrix[np.bool_]: ...
@overload  # dtype like int, format: "csc"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyIntDType, format: _FmtCSC
) -> csc_matrix[np.int_]: ...
@overload  # dtype like complex, format: "csc"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyComplex128DType, format: _FmtCSC
) -> csc_matrix[np.complex128]: ...
@overload  # dtype like <known>, format: "csc"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.ToDType[_SCT], format: _FmtCSC
) -> csc_matrix[_SCT]: ...
@overload  # dtype like <unknown>, format: "csc"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: npt.DTypeLike, format: _FmtCSC
) -> csc_matrix[Incomplete]: ...
@overload  # dtype like float (default), format: "csr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, dtype: onp.AnyFloat64DType = ..., *, format: _FmtCSR
) -> csr_matrix[np.float64]: ...
@overload  # dtype like bool, format: "csr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyBoolDType, format: _FmtCSR
) -> csr_matrix[np.bool_]: ...
@overload  # dtype like int, format: "csr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyIntDType, format: _FmtCSR
) -> csr_matrix[np.int_]: ...
@overload  # dtype like complex, format: "csr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyComplex128DType, format: _FmtCSR
) -> csr_matrix[np.complex128]: ...
@overload  # dtype like <known>, format: "csr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.ToDType[_SCT], format: _FmtCSR
) -> csr_matrix[_SCT]: ...
@overload  # dtype like <unknown>, format: "csr"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: npt.DTypeLike, format: _FmtCSR
) -> csr_matrix[Incomplete]: ...
@overload  # dtype like float (default), format: "dok"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, dtype: onp.AnyFloat64DType = ..., *, format: _FmtDOK
) -> dok_matrix[np.float64]: ...
@overload  # dtype like bool, format: "dok"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyBoolDType, format: _FmtDOK
) -> dok_matrix[np.bool_]: ...
@overload  # dtype like int, format: "dok"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyIntDType, format: _FmtDOK
) -> dok_matrix[np.int_]: ...
@overload  # dtype like complex, format: "dok"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyComplex128DType, format: _FmtDOK
) -> dok_matrix[np.complex128]: ...
@overload  # dtype like <known>, format: "dok"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.ToDType[_SCT], format: _FmtDOK
) -> dok_matrix[_SCT]: ...
@overload  # dtype like <unknown>, format: "dok"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: npt.DTypeLike, format: _FmtDOK
) -> dok_matrix[Incomplete]: ...
@overload  # dtype like float (default), format: "lil"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, dtype: onp.AnyFloat64DType = ..., *, format: _FmtLIL
) -> lil_matrix[np.float64]: ...
@overload  # dtype like bool, format: "lil"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyBoolDType, format: _FmtLIL
) -> lil_matrix[np.bool_]: ...
@overload  # dtype like int, format: "lil"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyIntDType, format: _FmtLIL
) -> lil_matrix[np.int_]: ...
@overload  # dtype like complex, format: "lil"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.AnyComplex128DType, format: _FmtLIL
) -> lil_matrix[np.complex128]: ...
@overload  # dtype like <known>, format: "lil"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: onp.ToDType[_SCT], format: _FmtLIL
) -> lil_matrix[_SCT]: ...
@overload  # dtype like <unknown>, format: "lil"
def eye(
    m: opt.AnyInt, n: opt.AnyInt | None = None, k: int = 0, *, dtype: npt.DTypeLike, format: _FmtLIL
) -> lil_matrix[Incomplete]: ...

#
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: {"bsr", None} = ...
def kron(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtBSR | None = None) -> bsr_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "coo"
def kron(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtCOO) -> coo_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "csc"
def kron(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtCSC) -> csc_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "csr"
def kron(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtCSR) -> csr_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "dia"
def kron(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtDIA) -> dia_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "dok"
def kron(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtDOK) -> dok_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "lil"
def kron(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtLIL) -> lil_matrix[_SCT]: ...
@overload  # A: sparray, B: 2D sparse, format: {"bsr", None} = ...
def kron(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtBSR | None = None) -> _BSRArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "coo"
def kron(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtCOO) -> _COOArray2D[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "csc"
def kron(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtCSC) -> _CSCArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "csr"
def kron(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtCSR) -> _CSRArray2D[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "dia"
def kron(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtDIA) -> _DIAArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "dok"
def kron(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtDOK) -> _DOKArray2D[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "lil"
def kron(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtLIL) -> _LILArray[_SCT]: ...
@overload  # A: sparse, B: sparray, format: {"bsr", None} = ...
def kron(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtBSR | None = None) -> _BSRArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "coo"
def kron(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtCOO) -> _COOArray2D[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "csc"
def kron(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtCSC) -> _CSCArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "csr"
def kron(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtCSR) -> _CSRArray2D[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "dia"
def kron(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtDIA) -> _DIAArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "dok"
def kron(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtDOK) -> _DOKArray2D[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "lil"
def kron(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtLIL) -> _LILArray[_SCT]: ...
@overload  # A: unknown array-like, B: unknown array-like  (catch-all)
def kron(A: onp.ToComplex2D, B: onp.ToComplex2D, format: _Format | None = None) -> _SpBase2D[Incomplete]: ...

# NOTE: The `overload-overlap` mypy errors are false positives.
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: {"csr", None} = ...
def kronsum(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtCSR | None = None) -> csr_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "bsr"
def kronsum(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtBSR) -> bsr_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "coo"
def kronsum(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtCOO) -> coo_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "csc"
def kronsum(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtCSC) -> csc_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "dia"
def kronsum(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtDIA) -> dia_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "dok"
def kronsum(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtDOK) -> dok_matrix[_SCT]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: "lil"
def kronsum(A: _ToSpMatrix[_SCT], B: _ToSpMatrix[_SCT], format: _FmtLIL) -> lil_matrix[_SCT]: ...
@overload  # A: sparray, B: sparse, format: {"csr", None} = ...
def kronsum(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtCSR | None = None) -> _CSRArray2D[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "bsr"
def kronsum(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtBSR) -> _BSRArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "coo"
def kronsum(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtCOO) -> _COOArray2D[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "csc"
def kronsum(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtCSC) -> _CSCArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "dia"
def kronsum(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtDIA) -> _DIAArray[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "dok"
def kronsum(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtDOK) -> _DOKArray2D[_SCT]: ...
@overload  # A: sparray, B: sparse, format: "lil"
def kronsum(A: sparray[_SCT, tuple[int, int]], B: _ToSparse2D[_SCT], format: _FmtLIL) -> _LILArray[_SCT]: ...
@overload  # A: sparse, B: sparray, format: {"csr", None} = ...
def kronsum(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtCSR | None = None) -> _CSRArray2D[_SCT]: ...
@overload  # A: sparse, B: sparray, format: "bsr"
def kronsum(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtBSR) -> _BSRArray[_SCT]: ...
@overload  # A: sparse, B: sparray, format: "coo"
def kronsum(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtCOO) -> _COOArray2D[_SCT]: ...
@overload  # A: sparse, B: sparray, format: "csc"
def kronsum(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtCSC) -> _CSCArray[_SCT]: ...
@overload  # A: sparse, B: sparray, format: "dia"
def kronsum(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtDIA) -> _DIAArray[_SCT]: ...
@overload  # A: sparse, B: sparray, format: "dok"
def kronsum(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtDOK) -> _DOKArray2D[_SCT]: ...
@overload  # A: sparse, B: sparray, format: "lil"
def kronsum(A: _ToSparse2D[_SCT], B: sparray[_SCT, tuple[int, int]], format: _FmtLIL) -> _LILArray[_SCT]: ...
@overload  # A: unknown array-like, B: unknown array-like  (catch-all)
def kronsum(A: onp.ToComplex2D, B: onp.ToComplex2D, format: _Format | None = None) -> _SpBase2D[Incomplete]: ...

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
def hstack(blocks: Seq[_spbase], format: _Format, dtype: npt.DTypeLike | None = None) -> Incomplete: ...

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
def vstack(blocks: Seq[_spbase], format: _Format, dtype: npt.DTypeLike | None = None) -> Incomplete: ...

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
def bmat(blocks: Seq[Seq[sparray[_SCT]]], format: _Format, dtype: None = None) -> _SpArray2D[_SCT]: ...
@overload  # sparray, blocks: <matrix, known dtype>, format: <otherwise>, dtype: <default>
def bmat(blocks: Seq[Seq[spmatrix[_SCT]]], format: _Format, dtype: None = None) -> _SpMatrix[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <otherwise>, dtype: <known>
def bmat(blocks: _ToBlocks, format: _Format, dtype: onp.ToDType[_SCT]) -> _SpBase2D[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <otherwise>, dtype: <unknown>
def bmat(blocks: _ToBlocks, format: _Format, dtype: npt.DTypeLike) -> _SpBase2D: ...

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
    mats: Iterable[spmatrix | onp.ArrayND[_Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _FmtCOO | None,
    dtype: onp.ToDType[_SCT],
) -> coo_matrix[_SCT]: ...
@overload  # mats: <matrix, unknown dtype>, dtype: <known>  (keyword)
def block_diag(
    mats: Iterable[spmatrix | onp.ArrayND[_Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _FmtCOO | None = None,
    *,
    dtype: onp.ToDType[_SCT],
) -> coo_matrix[_SCT]: ...
@overload  # mats: <unknown, unknown dtype>, dtype: <known>  (positional)
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[_Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _FmtCOO | None,
    dtype: onp.ToDType[_SCT],
) -> _COOArray2D[_SCT] | coo_matrix[_SCT]: ...
@overload  # mats: <unknown, unknown dtype>, dtype: <known>  (keyword)
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[_Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _FmtCOO | None = None,
    *,
    dtype: onp.ToDType[_SCT],
) -> _COOArray2D[_SCT] | coo_matrix[_SCT]: ...
@overload  # catch-all
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[_Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _FmtCOO | None = None,
    dtype: npt.DTypeLike | None = None,
) -> _COOArray2D[_SCT] | coo_matrix[Any]: ...
@overload  # catch-all
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[_Numeric] | complex | Seq[onp.ToComplex] | Seq[onp.ToComplex1D]],
    format: _Format | None = None,
    dtype: npt.DTypeLike | None = None,
) -> Incomplete: ...

#
@overload  # shape: T, dtype: <default>, format: <default>
def random_array(
    shape: _ShapeT,
    *,
    density: float | npc.floating = 0.01,
    format: _FmtCOO = "coo",
    dtype: onp.AnyFloat64DType | None = None,
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
    format: _Format = "coo",
    dtype: onp.AnyFloat64DType | None = None,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray1D[np.float64]: ...
@overload  # shape: 2d, dtype: <default>
def random_array(
    shape: tuple[int, int],
    *,
    density: float | npc.floating = 0.01,
    format: _Format = "coo",
    dtype: onp.AnyFloat64DType | None = None,
    rng: onp.random.ToRNG | None = None,
    random_state: onp.random.ToRNG | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray2D[np.float64]: ...
@overload  # shape: 1d, dtype: <known>
def random_array(
    shape: tuple[int],
    *,
    density: float | npc.floating = 0.01,
    format: _Format = "coo",
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
    format: _Format = "coo",
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
    format: _Format = "coo",
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
    format: _Format = "coo",
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
    format: _Format = "coo",
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
    format: _Format = "coo",
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
    dtype: onp.AnyFloat64DType | None = None,
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
    format: _Format = ...,
    dtype: onp.AnyFloat64DType | None = None,
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
    format: _Format,
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
    format: _Format = ...,
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
    format: _Format,
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
    format: _Format = ...,
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
    format: _Format = ...,
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
    dtype: onp.AnyFloat64DType | None = None,
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
    format: _Format = "coo",
    dtype: onp.AnyFloat64DType | None = None,
    rng: onp.random.ToRNG | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix[np.float64]: ...
@overload  # dtype: <known> (positional)
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | npc.floating,
    format: _Format,
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
    format: _Format = "coo",
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
    format: _Format,
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
    format: _Format = "coo",
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
    format: _Format = "coo",
    dtype: npt.DTypeLike | None = None,
    rng: onp.random.ToRNG | None = None,
    *,
    random_state: onp.random.ToRNG | None = None,
) -> _SpMatrix: ...
