# mypy: disable-error-code="misc"
# pyright: reportUnannotatedClassAttribute=false

import abc
from collections.abc import Iterator, Sequence
from typing import Any, Final, Generic, Literal, TypeAlias, overload
from typing_extensions import Self, TypeIs, TypeVar

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import Casting, OrderCF, Untyped
from ._bsr import bsr_array, bsr_matrix
from ._coo import coo_array, coo_matrix
from ._csc import csc_array, csc_matrix
from ._csr import csr_array, csr_matrix
from ._dia import dia_array, dia_matrix
from ._dok import dok_array, dok_matrix
from ._lil import lil_array, lil_matrix
from ._matrix import spmatrix as spmatrix
from ._typing import (
    Complex,
    Float,
    Index1D,
    Int,
    Matrix,
    Scalar,
    SPFormat,
    ToDType,
    ToDTypeBool,
    ToDTypeComplex,
    ToDTypeFloat,
    ToDTypeInt,
    ToShape,
    ToShape1D,
    ToShape2D,
)

__all__ = ["SparseEfficiencyWarning", "SparseWarning", "issparse", "isspmatrix", "sparray"]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Scalar, default=Any)
_SCT_co = TypeVar("_SCT_co", bound=Scalar, default=Scalar, covariant=True)
_SCT_fp = TypeVar("_SCT_fp", bound=Float | Complex)

_ShapeT = TypeVar("_ShapeT", bound=tuple[int] | tuple[int, int], default=tuple[int] | tuple[int, int])
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int] | tuple[int, int], covariant=True)

_Sp1dT = TypeVar("_Sp1dT", bound=_spbase[Any, tuple[int]])
_Sp2dT = TypeVar("_Sp2dT", bound=_spbase[Any, tuple[int, int]])
_SpBoolT = TypeVar("_SpBoolT", bound=_spbase[np.bool_])

_ArrayT = TypeVar("_ArrayT", bound=onp.ArrayND)
_SpMatrixT = TypeVar("_SpMatrixT", bound=spmatrix)

_FromIntT = TypeVar("_FromIntT", bound=_FromInt)
_FromFloatT = TypeVar("_FromFloatT", bound=_FromFloat)
_FromComplexT = TypeVar("_FromComplexT", bound=_FromComplex)

_SpFromInT = TypeVar("_SpFromInT", bound=_spbase[_FromInt])
_SpFromFloatT = TypeVar("_SpFromFloatT", bound=_spbase[_FromFloat])
_SpFromComplexT = TypeVar("_SpFromComplexT", bound=_spbase[_FromComplex])

_FromInt: TypeAlias = Int | Float | Complex
_FromFloat: TypeAlias = Float | Complex
_FromComplex: TypeAlias = Complex

_ToBool: TypeAlias = np.bool_
_ToInt8: TypeAlias = np.bool_ | np.int8
_ToInt: TypeAlias = Int | _ToBool
_ToFloat32: TypeAlias = np.bool_ | Int | np.float32
_ToFloat: TypeAlias = np.bool_ | Int | Float
_ToComplex64: TypeAlias = np.bool_ | Int | Float | np.complex64

_ToSparseFromPy: TypeAlias = Sequence[Sequence[_T]] | Sequence[_T]
_ToSparseFromArrayLike: TypeAlias = onp.CanArrayND[_SCT_co] | _ToSparseFromPy[_SCT_co]

_SpMatrix: TypeAlias = (
    bsr_matrix[_SCT]
    | coo_matrix[_SCT]
    | csc_matrix[_SCT]
    | csr_matrix[_SCT]
    | dia_matrix[_SCT]
    | dok_matrix[_SCT]
    | lil_matrix[_SCT]
)
_SpMatrixOut: TypeAlias = bsr_matrix[_SCT] | csc_matrix[_SCT] | csr_matrix[_SCT]

_SpArray: TypeAlias = (
    bsr_array[_SCT]
    | coo_array[_SCT, _ShapeT]
    | csc_array[_SCT]
    | csr_array[_SCT, _ShapeT]
    | dia_array[_SCT]
    | dok_array[_SCT, _ShapeT]
    | lil_array[_SCT]
)
_SpArray1D: TypeAlias = coo_array[_SCT, tuple[int]] | csr_array[_SCT, tuple[int]] | dok_array[_SCT, tuple[int]]
_SpArray2D: TypeAlias = _SpArray[_SCT, tuple[int, int]]
_SpArrayOut: TypeAlias = bsr_array[_SCT] | csc_array[_SCT] | csr_array[_SCT, _ShapeT]

_DualMatrixLike: TypeAlias = _T | _SCT | _spbase[_SCT]
_DualArrayLike2d: TypeAlias = Sequence[Sequence[_T | _SCT] | onp.CanArrayND[_SCT]] | onp.CanArrayND[_SCT]
_DualArrayLike: TypeAlias = Sequence[_T | _SCT] | _DualArrayLike2d[_T, _SCT]

###

MAXPRINT: Final = 50

class SparseWarning(Warning): ...
class SparseFormatWarning(SparseWarning): ...
class SparseEfficiencyWarning(SparseWarning): ...

class _spbase(Generic[_SCT_co, _ShapeT_co]):
    __array_priority__: float = 10.1
    maxprint: Final[int | None]

    @property
    def ndim(self, /) -> Literal[1, 2]: ...
    @property
    def shape(self, /) -> _ShapeT_co: ...
    @property
    def nnz(self, /) -> int: ...
    @property
    def size(self, /) -> int: ...

    # NOTE: At runtime this isn't abstract, but returns `und` instead.
    @property
    @abc.abstractmethod
    def format(self, /) -> SPFormat: ...

    #
    @property
    def T(self, /) -> Self: ...
    @property
    def real(self, /) -> Self: ...
    @property
    def imag(self, /) -> Self: ...

    # NOTE: In `scipy>=1.15.0` the `maxprint` param will become keyword-only.
    @overload  # shape
    def __init__(self: _spbase[np.float64], /, arg1: ToShape, maxprint: int | None = 50) -> None: ...
    @overload  # sparse
    def __init__(self, /, arg1: _spbase[_SCT_co], maxprint: int | None = 50) -> None: ...
    @overload  # dense array-like
    def __init__(self, /, arg1: _ToSparseFromArrayLike[_SCT_co], maxprint: int | None = 50) -> None: ...
    @overload  # dense array-like bool
    def __init__(self: _spbase[np.bool_], /, arg1: _ToSparseFromPy[bool], maxprint: int | None = 50) -> None: ...
    @overload  # dense array-like int
    def __init__(self: _spbase[np.int_], /, arg1: _ToSparseFromPy[opt.JustInt], maxprint: int | None = 50) -> None: ...
    @overload  # dense array-like float
    def __init__(self: _spbase[np.float64], /, arg1: _ToSparseFromPy[opt.Just[float]], maxprint: int | None = 50) -> None: ...
    @overload  # dense array-like cfloat
    def __init__(
        self: _spbase[np.complex128],
        /,
        arg1: _ToSparseFromPy[opt.Just[complex]],
        maxprint: int | None = 50,
    ) -> None: ...
    @overload  # dense array-like real (pyright is wrong here)
    def __init__(  # pyright: ignore[reportOverlappingOverload]
        self: _spbase[np.float64 | np.int_ | np.bool_],
        /,
        arg1: _ToSparseFromPy[float | int],
        maxprint: int | None = 50,
    ) -> None: ...
    @overload  # dense array-like complex (pyright is wrong here)
    def __init__(  # pyright: ignore[reportOverlappingOverload]
        self: _spbase[np.complex128 | np.float64 | np.int_ | np.bool_],
        /,
        arg1: _ToSparseFromPy[complex | float | int],
        maxprint: int | None = 50,
    ) -> None: ...

    #
    def __bool__(self, /) -> bool: ...

    #
    @overload  # 1-d {csr,dok}_array
    def __iter__(self: csr_array[_SCT, tuple[int]] | dok_array[_SCT, tuple[int]], /) -> Iterator[_SCT]: ...
    @overload  # 2-d {csc,csr}_array
    def __iter__(self: csc_array[_SCT] | csr_array[_SCT, tuple[int, int]], /) -> Iterator[csr_array[_SCT, tuple[int]]]: ...
    @overload  # lil_array
    def __iter__(self: lil_array[_SCT], /) -> Iterator[lil_array[_SCT]]: ...
    @overload  # {csc,csr}_matrix
    def __iter__(self: csc_matrix[_SCT] | csr_matrix[_SCT], /) -> Iterator[csr_matrix[_SCT]]: ...
    @overload  # dok_matrix
    def __iter__(self: dok_matrix[_SCT], /) -> Iterator[dok_matrix[_SCT]]: ...
    @overload  # lil_matrix
    def __iter__(self: lil_matrix[_SCT], /) -> Iterator[lil_matrix[_SCT]]: ...

    #
    @overload
    def __lt__(self: _SpBoolT, other: _spbase[_ToFloat] | onp.ToFloat, /) -> _SpBoolT: ...
    @overload
    def __lt__(self: sparray, other: _spbase[_ToFloat] | onp.ToFloat, /) -> csr_array[np.bool_, _ShapeT_co]: ...
    @overload
    def __lt__(self: spmatrix, other: _spbase[_ToFloat], /) -> csr_matrix[np.bool_]: ...
    #
    @overload
    def __gt__(self: _SpBoolT, other: _spbase[_ToFloat] | onp.ToFloat, /) -> _SpBoolT: ...
    @overload
    def __gt__(self: sparray, other: _spbase[_ToFloat] | onp.ToFloat, /) -> csr_array[np.bool_, _ShapeT_co]: ...
    @overload
    def __gt__(self: spmatrix, other: _spbase[_ToFloat], /) -> csr_matrix[np.bool_]: ...
    #
    @overload
    def __le__(self: _SpBoolT, other: _spbase[_ToFloat] | onp.ToFloat, /) -> _SpBoolT: ...
    @overload
    def __le__(self: sparray, other: _spbase[_ToFloat] | onp.ToFloat, /) -> csr_array[np.bool_, _ShapeT_co]: ...
    @overload
    def __le__(self: spmatrix, other: _spbase[_ToFloat], /) -> csr_matrix[np.bool_]: ...
    #
    @overload
    def __ge__(self: _SpBoolT, other: _spbase[_ToFloat] | onp.ToFloat, /) -> _SpBoolT: ...
    @overload
    def __ge__(self: sparray, other: _spbase[_ToFloat] | onp.ToFloat, /) -> csr_array[np.bool_, _ShapeT_co]: ...
    @overload
    def __ge__(self: spmatrix, other: _spbase[_ToFloat], /) -> _SpMatrixOut[np.bool_]: ...

    #
    def __neg__(self, /) -> Self: ...
    def __abs__(self, /) -> Self: ...
    def __round__(self, /, ndigits: int = 0) -> Self: ...

    # NOTE: only `lil_{array,matrix}` supports non-zero scalar addition (but not subtraction), but upcasts bool + 0 to int_
    @overload  # `0` or sparse of same dtype
    def __add__(self, other: Literal[False, 0] | _spbase[_SCT_co], /) -> Self: ...
    @overload  # sparse[-Int], sparse[+Int8]
    def __add__(self: _SpFromInT, other: _spbase[_ToInt8], /) -> _SpFromInT: ...
    @overload  # sparse[-Float], sparse[+Float32]
    def __add__(self: _SpFromFloatT, other: _spbase[_ToFloat32], /) -> _SpFromFloatT: ...
    @overload  # sparse[-Complex], sparse[+Complex64]
    def __add__(self: _SpFromComplexT, other: _spbase[_ToComplex64], /) -> _SpFromComplexT: ...
    @overload  # spmatrix[-Int], array[+Int8]
    def __add__(self: spmatrix[_FromInt], other: onp.ArrayND[_ToInt8], /) -> Matrix[_SCT_co]: ...
    @overload  # spmatrix[-Float], array[+Float32]
    def __add__(self: spmatrix[_FromFloat], other: onp.ArrayND[_ToFloat32], /) -> Matrix[_SCT_co]: ...
    @overload  # spmatrix[-Complex], array[+Complex64]
    def __add__(self: spmatrix[_FromComplex], other: onp.ArrayND[_ToComplex64], /) -> Matrix[_SCT_co]: ...
    @overload  # sparse[-Int], array[+Int8]
    def __add__(self: _spbase[_FromInt], other: onp.ArrayND[_ToInt8], /) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload  # sparse[-Float], array[+Float32]
    def __add__(self: _spbase[_FromFloat], other: onp.ArrayND[_ToFloat32], /) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload  # sparse[-Complex], arrau[+Complex64]
    def __add__(self: _spbase[_FromComplex], other: onp.ArrayND[_ToComplex64], /) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload  # catch-all
    def __add__(self, other: complex | Scalar | onp.ArrayND[Scalar] | _spbase, /) -> _spbase[Any, Any] | onp.Array[Any, Any]: ...
    __radd__ = __add__

    #
    @overload  # `0` or sparse of same dtype
    def __sub__(self, other: _spbase[_SCT_co], /) -> Self: ...
    @overload  # sparse[-Int], sparse[+Int8]
    def __sub__(self: _SpFromInT, other: _spbase[_ToInt8], /) -> _SpFromInT: ...
    @overload  # sparse[-Float], sparse[+Float32]
    def __sub__(self: _SpFromFloatT, other: _spbase[_ToFloat32], /) -> _SpFromFloatT: ...
    @overload  # sparse[-Complex], sparse[+Complex64]
    def __sub__(self: _SpFromComplexT, other: _spbase[_ToComplex64], /) -> _SpFromComplexT: ...
    @overload  # spmatrix[-Int], array[+Int8]
    def __sub__(self: spmatrix[_FromInt], other: onp.ArrayND[_ToInt8], /) -> Matrix[_SCT_co]: ...
    @overload  # spmatrix[-Float], array[+Float32]
    def __sub__(self: spmatrix[_FromFloat], other: onp.ArrayND[_ToFloat32], /) -> Matrix[_SCT_co]: ...
    @overload  # spmatrix[-Complex], array[+Complex64]
    def __sub__(self: spmatrix[_FromComplex], other: onp.ArrayND[_ToComplex64], /) -> Matrix[_SCT_co]: ...
    @overload  # sparse[-Int], array[+Int8]
    def __sub__(self: _spbase[_FromInt], other: onp.ArrayND[_ToInt8], /) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload  # sparse[-Float], array[+Float32]
    def __sub__(self: _spbase[_FromFloat], other: onp.ArrayND[_ToFloat32], /) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload  # sparse[-Complex], arrau[+Complex64]
    def __sub__(self: _spbase[_FromComplex], other: onp.ArrayND[_ToComplex64], /) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload  # catch-all
    def __sub__(self, other: onp.ArrayND[Scalar] | _spbase, /) -> _spbase[Any, Any] | onp.Array[Any, Any]: ...
    __rsub__ = __sub__

    #
    @overload  # Self[-Bool], /, other: scalar-like +Bool
    def __mul__(self, /, other: bool | _ToBool) -> Self: ...
    @overload  # Self[-Int], /, other: scalar-like +Int
    def __mul__(self: _SpFromInT, /, other: onp.ToInt) -> _SpFromInT: ...
    @overload  # Self[-Float], /, other: scalar-like +Float
    def __mul__(self: _SpFromFloatT, /, other: onp.ToFloat) -> _SpFromFloatT: ...
    @overload  # Self[-Complex], /, other: scalar-like +Complex
    def __mul__(self: _SpFromComplexT, /, other: onp.ToComplex) -> _SpFromComplexT: ...
    @overload  # sparray[-Bool], /, other: sparse +Bool
    def __mul__(self: _SpArray, /, other: _spbase[_ToBool | _SCT_co]) -> _SpArrayOut[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Bool], /, other: array-like +Bool
    def __mul__(self: _SpArray, /, other: _DualArrayLike[bool, _ToBool]) -> coo_array[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Int], /, other: sparse +Int
    def __mul__(self: _SpArray[_FromInt], /, other: _spbase[_ToInt8 | _SCT_co]) -> _SpArrayOut[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Int], /, other: array-like +Int
    def __mul__(self: _SpArray[_FromInt], /, other: _DualArrayLike[bool, _ToInt8]) -> coo_array[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Float], /, other: sparse +Float
    def __mul__(self: _SpArray[_FromFloat], /, other: _spbase[_ToFloat32 | _SCT_co]) -> _SpArrayOut[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Float], /, other: array-like +Float
    def __mul__(self: _SpArray[_FromFloat], /, other: _DualArrayLike[int, _ToFloat32]) -> coo_array[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Complex], /, other: sparse +Complex
    def __mul__(self: _SpArray[_FromComplex], /, other: _spbase[_ToComplex64 | _SCT_co]) -> _SpArrayOut[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Complex], /, other: array-like +Complex
    def __mul__(self: _SpArray[_FromComplex], /, other: _DualArrayLike[int, _ToComplex64]) -> coo_array[_SCT_co, _ShapeT_co]: ...
    @overload  # spmatrix, /, other: spmatrix
    def __mul__(self: _SpMatrixT, /, other: _SpMatrixT) -> _SpMatrixT: ...
    @overload  # spmatrix[-Bool], /, other: sparse +Bool
    def __mul__(self: spmatrix, /, other: _spbase[_ToBool]) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Bool], /, other: array-like +Bool
    def __mul__(self: spmatrix, /, other: _DualArrayLike2d[bool, _ToBool]) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[-Int], /, other: sparse +Int
    def __mul__(self: spmatrix[_FromInt], /, other: _spbase[_ToInt8]) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Int], /, other: array-like +Int
    def __mul__(self: spmatrix[_FromInt], /, other: _DualArrayLike2d[bool, _ToInt8]) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[-Float], /, other: sparse +Float
    def __mul__(self: spmatrix[_FromFloat], /, other: _spbase[_ToFloat32 | _SCT_co]) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Float], /, other: array-like +Float
    def __mul__(self: spmatrix[_FromFloat], /, other: _DualArrayLike2d[int, _ToFloat32]) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[-Complex], /, other: sparse +Complex
    def __mul__(self: spmatrix[_FromComplex], /, other: _spbase[_ToComplex64 | _SCT_co]) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Complex], /, other: array-like +Complex
    def __mul__(self: spmatrix[_FromComplex], /, other: _DualArrayLike2d[float, _ToComplex64]) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[+Bool], /, other: scalar- or matrix-like ~Int
    def __mul__(self: spmatrix[_ToBool], /, other: _DualMatrixLike[opt.JustInt, Int]) -> spmatrix[Int]: ...
    @overload  # spmatrix[+Bool], /, other: array-like ~Int
    def __mul__(self: spmatrix[_ToBool], /, other: _DualArrayLike2d[opt.JustInt, Int]) -> onp.Array2D[Int]: ...
    @overload  # spmatrix[+Int], /, other: scalar- or matrix-like ~Float
    def __mul__(self: spmatrix[_ToInt], /, other: _DualMatrixLike[opt.Just[float], Float]) -> spmatrix[Float]: ...
    @overload  # spmatrix[+Int], /, other: array-like ~Float
    def __mul__(self: spmatrix[_ToInt], /, other: _DualArrayLike2d[opt.Just[float], Float]) -> onp.Array2D[Float]: ...
    @overload  # spmatrix[+Float], /, other: scalar- or matrix-like ~Complex
    def __mul__(self: spmatrix[_ToFloat], /, other: _DualMatrixLike[opt.Just[complex], Complex]) -> spmatrix[Complex]: ...
    @overload  # spmatrix[+Float], /, other: array-like ~Complex
    def __mul__(self: spmatrix[_ToFloat], /, other: _DualArrayLike2d[opt.Just[complex], Complex]) -> onp.Array2D[Complex]: ...
    @overload  # Self[+Bool], /, other: -Int
    def __mul__(self: _spbase[_ToBool], /, other: _FromIntT) -> _spbase[_FromIntT, _ShapeT_co]: ...
    @overload  # Self[+Int], /, other: -Float
    def __mul__(self: _spbase[_ToInt], /, other: _FromFloatT) -> _spbase[_FromFloatT, _ShapeT_co]: ...
    @overload  # Self[+Float], /, other: -Complex
    def __mul__(self: _spbase[_ToFloat], /, other: _FromComplexT) -> _spbase[_FromComplexT, _ShapeT_co]: ...
    @overload  # catch-all
    def __mul__(self, /, other: _DualArrayLike[complex, Scalar] | _spbase) -> _spbase[Any, Any] | onp.Array[Any, Any]: ...
    multiply = __mul__
    __rmul__ = __mul__

    #
    @overload  # sparray[-Bool], other: sparse +Bool
    def __matmul__(self: _SpArray, other: _spbase[_ToBool | _SCT_co], /) -> _SpArrayOut[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Bool], other: array-like +Bool
    def __matmul__(self: _SpArray, other: _DualArrayLike[bool, _ToBool], /) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Int], other: sparse +Int
    def __matmul__(self: _SpArray[_FromInt], other: _spbase[_ToInt8 | _SCT_co], /) -> _SpArrayOut[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Int], other: array-like +Int
    def __matmul__(self: _SpArray[_FromInt], other: _DualArrayLike[bool, _ToInt8], /) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Float], other: sparse +Float
    def __matmul__(self: _SpArray[_FromFloat], other: _spbase[_ToFloat32 | _SCT_co], /) -> _SpArrayOut[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Float], other: array-like +Float
    def __matmul__(self: _SpArray[_FromFloat], other: _DualArrayLike[int, _ToFloat32], /) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Complex], other: sparse +Complex
    def __matmul__(
        self: _SpArray[_FromComplex],
        other: _spbase[_ToComplex64 | _SCT_co],
        /,
    ) -> _SpArrayOut[_SCT_co, _ShapeT_co]: ...
    @overload  # sparray[-Complex], other: array-like +Complex
    def __matmul__(
        self: _SpArray[_FromComplex],
        other: _DualArrayLike[int, _ToComplex64],
        /,
    ) -> onp.ArrayND[_SCT_co, _ShapeT_co]: ...
    @overload  # spmatrix, other: spmatrix
    def __matmul__(self: _SpMatrixT, other: _SpMatrixT, /) -> _SpMatrixT: ...
    @overload  # spmatrix[-Bool], other: sparse +Bool
    def __matmul__(self: spmatrix, other: _spbase[_ToBool], /) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Bool], other: array-like +Bool
    def __matmul__(self: spmatrix, other: _DualArrayLike2d[bool, _ToBool], /) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[-Int], other: sparse +Int
    def __matmul__(self: spmatrix[_FromInt], other: _spbase[_ToInt8], /) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Int], other: array-like +Int
    def __matmul__(self: spmatrix[_FromInt], other: _DualArrayLike2d[bool, _ToInt8], /) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[-Float], other: sparse +Float
    def __matmul__(self: spmatrix[_FromFloat], other: _spbase[_ToFloat32 | _SCT_co], /) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Float], other: array-like +Float
    def __matmul__(self: spmatrix[_FromFloat], other: _DualArrayLike2d[int, _ToFloat32], /) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[-Complex], other: sparse +Complex
    def __matmul__(self: spmatrix[_FromComplex], other: _spbase[_ToComplex64 | _SCT_co], /) -> _SpMatrixOut[_SCT_co]: ...
    @overload  # spmatrix[-Complex], other: array-like +Complex
    def __matmul__(self: spmatrix[_FromComplex], other: _DualArrayLike2d[float, _ToComplex64], /) -> onp.Array2D[_SCT_co]: ...
    @overload  # spmatrix[+Bool], other: scalar- or matrix-like ~Int
    def __matmul__(self: spmatrix[_ToBool], other: _spbase[Int], /) -> _SpMatrixOut[Int]: ...
    @overload  # spmatrix[+Bool], other: array-like ~Int
    def __matmul__(self: spmatrix[_ToBool], other: _DualArrayLike2d[opt.JustInt, Int], /) -> onp.Array2D[Int]: ...
    @overload  # spmatrix[+Int], other: scalar- or matrix-like ~Float
    def __matmul__(self: spmatrix[_ToInt], other: _spbase[Float], /) -> _SpMatrixOut[Float]: ...
    @overload  # spmatrix[+Int], other: array-like ~Float
    def __matmul__(self: spmatrix[_ToInt], other: _DualArrayLike2d[opt.Just[float], Float], /) -> onp.Array2D[Float]: ...
    @overload  # spmatrix[+Float], other: scalar- or matrix-like ~Complex
    def __matmul__(self: spmatrix[_ToFloat], other: _spbase[Complex], /) -> _SpMatrixOut[Complex]: ...
    @overload  # spmatrix[+Float], other: array-like ~Complex
    def __matmul__(self: spmatrix[_ToFloat], other: _DualArrayLike2d[opt.Just[complex], Complex], /) -> onp.Array2D[Complex]: ...
    @overload  # catch-all
    def __matmul__(self, other: _DualArrayLike[complex, Scalar] | _spbase, /) -> _spbase[Any, Any] | onp.Array[Any, Any]: ...
    __rmatmul__ = __matmul__
    #
    def __truediv__(self, rhs: Untyped, /) -> Untyped: ...
    __div__ = __truediv__

    #
    def __pow__(self, rhs: op.CanIndex, /) -> Self: ...

    #
    def dot(self, /, other: Untyped) -> Untyped: ...

    #
    def maximum(self, /, other: Untyped) -> Untyped: ...
    def minimum(self, /, other: Untyped) -> Untyped: ...

    #
    @overload
    def power(self, /, n: op.CanIndex, dtype: ToDType[_SCT_co] | None = None) -> Self: ...
    @overload
    def power(self, /, n: op.CanIndex, dtype: ToDTypeBool) -> _spbase[np.bool_]: ...
    @overload
    def power(self, /, n: op.CanIndex, dtype: ToDTypeInt) -> _spbase[np.int_]: ...
    @overload
    def power(self, /, n: op.CanIndex, dtype: ToDTypeFloat) -> _spbase[np.float64]: ...
    @overload
    def power(self, /, n: op.CanIndex, dtype: ToDTypeComplex) -> _spbase[np.complex128]: ...
    @overload
    def power(self, /, n: op.CanIndex, dtype: ToDType[_SCT]) -> _spbase[_SCT]: ...

    #
    def nonzero(self, /) -> tuple[Index1D, Index1D]: ...
    def count_nonzero(self, /) -> int: ...
    def conjugate(self, /, copy: bool = True) -> Self: ...
    def conj(self, /, copy: bool = True) -> Self: ...
    def transpose(self, /, axes: None = None, copy: bool = False) -> Self: ...

    #
    def diagonal(self, /, k: int = 0) -> onp.Array1D[_SCT_co]: ...  # only if 2-d
    def trace(self, /, offset: int = 0) -> _SCT_co: ...

    #
    def sum(self, /, axis: op.CanIndex | None = None, dtype: Untyped | None = None, out: Untyped | None = None) -> Untyped: ...
    #
    @overload  # out: array
    def mean(self, /, axis: op.CanIndex | None = None, dtype: npt.DTypeLike | None = None, *, out: _ArrayT) -> _ArrayT: ...
    @overload  # axis: None = ..., dtype: <known>  (keyword)
    def mean(self, /, axis: None = None, *, dtype: ToDType[_SCT], out: None = None) -> _SCT: ...
    @overload  # Self[+Int], axis: None = ...
    def mean(self: _spbase[np.bool_ | Int], /, axis: None = None, dtype: None = None, out: None = None) -> np.float64: ...
    @overload  # spmatrix[+Int], axis: index
    def mean(
        self: spmatrix[np.bool_ | Int],
        /,
        axis: op.CanIndex,
        dtype: None = None,
        out: None = None,
    ) -> Matrix[np.float64]: ...
    @overload  # Self[Float | Complex], axis: None = ...
    def mean(self: _spbase[_SCT_fp], /, axis: None = None, dtype: None = None, out: None = None) -> _SCT_fp: ...
    @overload  # spmatrix[Float | Complex], axis: index
    def mean(self: spmatrix[_SCT_fp], /, axis: op.CanIndex, dtype: None = None, out: None = None) -> Matrix[_SCT_fp]: ...
    @overload  # spmatrix, axis: index, dtype: <unknown>
    def mean(self: spmatrix, /, axis: op.CanIndex, dtype: npt.DTypeLike, out: None = None) -> Matrix[Any]: ...
    @overload  # dtype: <unknown>  (keyword)
    def mean(self, /, axis: op.CanIndex | None = None, *, dtype: npt.DTypeLike, out: None = None) -> Any: ...  # noqa: ANN401

    #
    def copy(self, /) -> Self: ...

    #
    @overload
    def reshape(self: _Sp1dT, shape: ToShape1D, /, *, order: OrderCF = "C", copy: bool = False) -> _Sp1dT: ...
    @overload
    def reshape(self: _Sp2dT, shape: ToShape2D, /, *, order: OrderCF = "C", copy: bool = False) -> _Sp2dT: ...

    # NOTE: the following two ignored errors won't cause any problems (when using the public API)
    @overload  # current type
    def astype(  # pyright: ignore[reportOverlappingOverload]
        self,
        /,
        dtype: ToDType[_SCT_co],
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> Self: ...
    @overload  # known type -> sparray
    def astype(  # pyright: ignore[reportOverlappingOverload]
        self: bsr_array,
        /,
        dtype: ToDType[_SCT],
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> bsr_array[_SCT]: ...
    @overload
    def astype(
        self: coo_array,
        /,
        dtype: ToDType[_SCT],
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> coo_array[_SCT, _ShapeT_co]: ...
    @overload
    def astype(
        self: csc_array,
        /,
        dtype: ToDType[_SCT],
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> csc_array[_SCT]: ...
    @overload
    def astype(
        self: csr_array,
        /,
        dtype: ToDType[_SCT],
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> csr_array[_SCT, _ShapeT_co]: ...
    @overload
    def astype(
        self: dia_array,
        /,
        dtype: ToDType[_SCT],
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> dia_array[_SCT]: ...
    @overload
    def astype(
        self: dok_array,
        /,
        dtype: ToDType[_SCT],
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> dok_array[_SCT, _ShapeT_co]: ...
    @overload
    def astype(self: lil_array, /, dtype: ToDType[_SCT], casting: Casting = "unsafe", copy: bool = True) -> lil_array[_SCT]: ...
    @overload  # known type -> spmatrix
    def astype(self: bsr_matrix, /, dtype: ToDType[_SCT], casting: Casting = "unsafe", copy: bool = True) -> bsr_matrix[_SCT]: ...
    @overload
    def astype(self: coo_matrix, /, dtype: ToDType[_SCT], casting: Casting = "unsafe", copy: bool = True) -> coo_matrix[_SCT]: ...
    @overload
    def astype(self: csc_matrix, /, dtype: ToDType[_SCT], casting: Casting = "unsafe", copy: bool = True) -> csc_matrix[_SCT]: ...
    @overload
    def astype(self: csr_matrix, /, dtype: ToDType[_SCT], casting: Casting = "unsafe", copy: bool = True) -> csr_matrix[_SCT]: ...
    @overload
    def astype(self: dia_matrix, /, dtype: ToDType[_SCT], casting: Casting = "unsafe", copy: bool = True) -> dia_matrix[_SCT]: ...
    @overload
    def astype(self: dok_matrix, /, dtype: ToDType[_SCT], casting: Casting = "unsafe", copy: bool = True) -> dok_matrix[_SCT]: ...
    @overload
    def astype(self: lil_matrix, /, dtype: ToDType[_SCT], casting: Casting = "unsafe", copy: bool = True) -> lil_matrix[_SCT]: ...
    @overload  # dtype-like -> 1d sparray
    def astype(
        self: _spbase[Any, tuple[int]],
        /,
        dtype: ToDTypeBool,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpArray1D[np.bool_]: ...
    @overload
    def astype(
        self: _spbase[Any, tuple[int]],
        /,
        dtype: ToDTypeInt,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpArray1D[np.int_]: ...
    @overload
    def astype(
        self: _spbase[Any, tuple[int]],
        /,
        dtype: ToDTypeFloat | None,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpArray1D[np.float64]: ...
    @overload
    def astype(
        self: _spbase[Any, tuple[int]],
        /,
        dtype: ToDTypeComplex,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpArray1D[np.complex128]: ...
    @overload  # dtype-like -> 2d sparray
    def astype(
        self: sparray,
        /,
        dtype: ToDTypeBool,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpArray2D[np.bool_]: ...
    @overload
    def astype(
        self: sparray,
        /,
        dtype: ToDTypeInt,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpArray2D[np.int_]: ...
    @overload
    def astype(
        self: sparray,
        /,
        dtype: ToDTypeFloat | None,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpArray2D[np.float64]: ...
    @overload
    def astype(
        self: sparray,
        /,
        dtype: ToDTypeComplex,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpArray2D[np.complex128]: ...
    @overload  # dtype-like -> spmatrix
    def astype(
        self: spmatrix,
        /,
        dtype: ToDTypeBool,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpMatrix[np.bool_]: ...
    @overload
    def astype(
        self: spmatrix,
        /,
        dtype: ToDTypeInt,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpMatrix[np.int_]: ...
    @overload
    def astype(
        self: spmatrix,
        /,
        dtype: ToDTypeFloat | None,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpMatrix[np.float64]: ...
    @overload
    def astype(
        self: spmatrix,
        /,
        dtype: ToDTypeComplex,
        casting: Casting = "unsafe",
        copy: bool = True,
    ) -> _SpMatrix[np.complex128]: ...
    @overload  # catch-all
    def astype(self, /, dtype: npt.DTypeLike, casting: Casting = "unsafe", copy: bool = True) -> _spbase[Any]: ...

    #
    @overload
    def asformat(self: sparray, /, format: Literal["bsr"], copy: bool = False) -> bsr_array[_SCT_co]: ...
    @overload
    def asformat(self: spmatrix, /, format: Literal["bsr"], copy: bool = False) -> bsr_matrix[_SCT_co]: ...
    @overload
    def asformat(self: sparray, /, format: Literal["coo"], copy: bool = False) -> coo_array[_SCT_co, _ShapeT_co]: ...
    @overload
    def asformat(self: spmatrix, /, format: Literal["coo"], copy: bool = False) -> coo_matrix[_SCT_co]: ...
    @overload
    def asformat(self: sparray, /, format: Literal["csc"], copy: bool = False) -> csc_array[_SCT_co]: ...
    @overload
    def asformat(self: spmatrix, /, format: Literal["csc"], copy: bool = False) -> csc_matrix[_SCT_co]: ...
    @overload
    def asformat(self: sparray, /, format: Literal["csr"], copy: bool = False) -> csr_array[_SCT_co, _ShapeT_co]: ...
    @overload
    def asformat(self: spmatrix, /, format: Literal["csr"], copy: bool = False) -> csr_matrix[_SCT_co]: ...
    @overload
    def asformat(self: sparray, /, format: Literal["dia"], copy: bool = False) -> dia_array[_SCT_co]: ...
    @overload
    def asformat(self: spmatrix, /, format: Literal["dia"], copy: bool = False) -> dia_matrix[_SCT_co]: ...
    @overload
    def asformat(self: sparray, /, format: Literal["dok"], copy: bool = False) -> dok_array[_SCT_co, _ShapeT_co]: ...
    @overload
    def asformat(self: spmatrix, /, format: Literal["dok"], copy: bool = False) -> dok_matrix[_SCT_co]: ...
    @overload
    def asformat(self: sparray, /, format: Literal["lil"], copy: bool = False) -> lil_array[_SCT_co]: ...
    @overload
    def asformat(self: spmatrix, /, format: Literal["lil"], copy: bool = False) -> lil_matrix[_SCT_co]: ...

    #
    @overload  # self: spmatrix, out: None
    def todense(self: spmatrix, /, order: OrderCF | None = None, out: None = None) -> Matrix[_SCT_co]: ...
    @overload  # self: spmatrix, out: array (positional)
    def todense(self: spmatrix, /, order: OrderCF | None, out: onp.ArrayND[_SCT]) -> Matrix[_SCT]: ...
    @overload  # self: spmatrix, out: array (keyword)
    def todense(self: spmatrix, /, order: OrderCF | None = None, *, out: onp.ArrayND[_SCT]) -> Matrix[_SCT]: ...
    @overload  # out: None
    def todense(self, /, order: OrderCF | None = None, out: None = None) -> onp.Array[_ShapeT_co, _SCT_co]: ...
    @overload  # out: array (positional)
    def todense(self, /, order: OrderCF | None, out: _ArrayT) -> _ArrayT: ...
    @overload  # out: array (keyword)
    def todense(self, /, order: OrderCF | None = None, *, out: _ArrayT) -> _ArrayT: ...

    #
    @overload  # out: None
    def toarray(self, /, order: OrderCF | None = None, out: None = None) -> onp.Array2D[_SCT_co]: ...
    @overload  # out: array (positional)
    def toarray(self, /, order: OrderCF | None, out: _ArrayT) -> _ArrayT: ...
    @overload  # out: array  (keyword)
    def toarray(self, /, order: OrderCF | None = None, *, out: _ArrayT) -> _ArrayT: ...

    #
    @overload
    def tobsr(self: sparray, /, blocksize: tuple[int, int] | None = None, copy: bool = False) -> bsr_array[_SCT_co]: ...
    @overload
    def tobsr(self: spmatrix, /, blocksize: tuple[int, int] | None = None, copy: bool = False) -> bsr_matrix[_SCT_co]: ...
    #
    @overload
    def tocoo(self: sparray, /, copy: bool = False) -> coo_array[_SCT_co, _ShapeT_co]: ...
    @overload
    def tocoo(self: spmatrix, /, copy: bool = False) -> coo_matrix[_SCT_co]: ...
    #
    @overload
    def tocsc(self: sparray, /, copy: bool = False) -> csc_array[_SCT_co]: ...
    @overload
    def tocsc(self: spmatrix, /, copy: bool = False) -> csc_matrix[_SCT_co]: ...
    #
    @overload
    def tocsr(self: sparray, /, copy: bool = False) -> csr_array[_SCT_co, _ShapeT_co]: ...
    @overload
    def tocsr(self: spmatrix, /, copy: bool = False) -> csr_matrix[_SCT_co]: ...
    #
    @overload
    def todia(self: sparray, /, copy: bool = False) -> dia_array[_SCT_co]: ...
    @overload
    def todia(self: spmatrix, /, copy: bool = False) -> dia_matrix[_SCT_co]: ...
    #
    @overload
    def todok(self: sparray, /, copy: bool = False) -> dok_array[_SCT_co, _ShapeT_co]: ...
    @overload
    def todok(self: spmatrix, /, copy: bool = False) -> dok_matrix[_SCT_co]: ...
    #
    @overload
    def tolil(self: sparray, /, copy: bool = False) -> lil_array[_SCT_co]: ...
    @overload
    def tolil(self: spmatrix, /, copy: bool = False) -> lil_matrix[_SCT_co]: ...

    #
    def resize(self, /, shape: ToShape) -> None: ...
    def setdiag(self, /, values: onp.ToComplex1D, k: int = 0) -> None: ...

class sparray: ...

def issparse(x: object) -> TypeIs[_spbase]: ...
def isspmatrix(x: object) -> TypeIs[spmatrix]: ...
