# mypy: disable-error-code="misc"

from collections.abc import Sequence
from typing import Any, ClassVar, Generic, Literal, Self, overload, override, type_check_only
from typing_extensions import TypeIs, TypeVar

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from ._base import _CanAsComplex128, _CanAsFloat64, _spbase, sparray
from ._bsr import bsr_array, bsr_matrix
from ._coo import coo_array, coo_matrix
from ._csc import csc_array, csc_matrix
from ._csr import csr_array, csr_matrix
from ._data import _data_matrix
from ._dok import dok_matrix
from ._lil import lil_matrix
from ._matrix import spmatrix
from ._typing import _CanAsAny, _ToShape2D, _ToShapeND

__all__ = ["dia_array", "dia_matrix", "isspmatrix_dia"]

type _2D = tuple[int, int]  # ruff: ignore[snake-case-type-alias]

type _ToInt8 = np.bool | np.int8
type _ToInt16 = np.bool | npc.integer8 | npc.integer16
type _ToInt = npc.integer | np.bool
type _ToFloat32 = np.float32 | _ToInt
type _ToFloat = npc.floating | _ToInt
type _ToComplex64 = np.complex64 | _ToFloat
type _Scalar = npc.number | np.bool

type _ToMatrixPy[T] = Sequence[T] | Sequence[Sequence[T]]
type _ToMatrix[ST: _Scalar] = _spbase[ST] | onp.CanArrayND[ST] | Sequence[onp.CanArrayND[ST]] | _ToMatrixPy[ST]
type _ToData[ST: _Scalar] = tuple[onp.ArrayND[ST] | Sequence[onp.CanArray1D[ST]], onp.ToJustInt | onp.ToJustInt1D]

type _SparseLike[T, ST: _Scalar] = T | ST | _spbase[ST]
type _To2DLike[T, ST: _Scalar] = Sequence[T | ST] | Sequence[Sequence[T | ST] | onp.CanArrayND[ST]] | onp.CanArrayND[ST]

type _SpMatrix[ST: _Scalar] = (
    bsr_matrix[ST] | coo_matrix[ST] | csc_matrix[ST] | csr_matrix[ST] | dia_matrix[ST] | dok_matrix[ST] | lil_matrix[ST]
)
type _SpMatrixOut[ST: _Scalar] = bsr_matrix[ST] | csc_matrix[ST] | csr_matrix[ST]
type _SpArrayOut[ST: _Scalar] = bsr_array[ST] | csc_array[ST] | csr_array[ST]

_ScalarT_co = TypeVar("_ScalarT_co", bound=_Scalar, default=Any, covariant=True)

_IntT = TypeVar("_IntT", bound=npc.integer, default=np.int_)
_FloatT = TypeVar("_FloatT", bound=npc.floating, default=np.float64)
_Float64T = TypeVar("_Float64T", bound=np.float64 | npc.floating80, default=np.float64)
_ComplexT = TypeVar("_ComplexT", bound=npc.complexfloating, default=np.complex128)
_Complex64T = TypeVar("_Complex64T", bound=npc.complexfloating, default=np.complex64)
_Complex128T = TypeVar("_Complex128T", bound=np.complex128 | npc.complexfloating160, default=np.complex128)

###

class _dia_base(_data_matrix[_ScalarT_co, tuple[int, int]], Generic[_ScalarT_co]):
    _format: ClassVar = "dia"

    data: onp.Array2D[_ScalarT_co]
    offsets: onp.Array1D[np.int32]

    @property
    @override
    def format(self, /) -> Literal["dia"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...

    #
    def __init__(
        self,
        /,
        arg1: onp.ToComplex2D,
        shape: _ToShape2D | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

    # NOTE: since scipy 1.16, `dia * dia` and `dia @ dia` (and so `multiply` and `dot`) return `dia`
    # NOTE: keep in sync with `_spbase.__mul__` and `multiply`
    @override  # type: ignore[override]
    @overload  # sparse, scalar +bool
    def __mul__(self, /, other: bool | np.bool) -> Self: ...
    @overload  # sparse T@number, scalar +int
    def __mul__[SelfT: _dia_base[npc.number]](self: SelfT, /, other: onp.ToInt) -> SelfT: ...
    @overload  # sparse T@inexact, scalar +float
    def __mul__[SelfT: _dia_base[npc.inexact]](self: SelfT, /, other: onp.ToFloat) -> SelfT: ...
    @overload  # sparse T@complex, scalar +complex
    def __mul__[SelfT: _dia_base[npc.complexfloating]](self: SelfT, /, other: onp.ToComplex) -> SelfT: ...
    @overload  # sparse +int, scalar ~float | ~f64
    def __mul__[OutT](self: _CanAsFloat64[_ToInt, OutT], /, other: op.JustFloat | np.float64) -> OutT: ...
    @overload  # sparse ~f64 | +int, scalar ~complex | ~c128
    def __mul__[OutT](self: _CanAsComplex128[np.float64 | _ToInt, OutT], /, other: op.JustComplex | np.complex128) -> OutT: ...
    @overload  # sparse, scalar ~int | ~float | ~complex
    def __mul__[OutT](self: _CanAsAny[OutT], /, other: op.JustInt | op.JustFloat | op.JustComplex) -> OutT: ...
    @overload  # dia T@+int, dia T | +i8
    def __mul__[SelfT: _dia_base[_ToInt]](self: SelfT, /, other: _dia_base[_ToInt8 | _ScalarT_co]) -> SelfT: ...
    @overload  # dia T@float, dia T | +f32
    def __mul__[SelfT: _dia_base[npc.floating]](self: SelfT, /, other: _dia_base[_ToFloat32 | _ScalarT_co]) -> SelfT: ...
    @overload  # dia T@complex, dia T | +c64
    def __mul__[SelfT: _dia_base[npc.complexfloating]](self: SelfT, /, other: _dia_base[_ToComplex64 | _ScalarT_co]) -> SelfT: ...
    @overload  # sparray, dense +bool
    def __mul__[ST: _Scalar](self: sparray[ST, _2D], /, other: _To2DLike[bool, np.bool]) -> coo_array[ST, _2D]: ...
    @overload  # sparray T@+int, sparse T | +i8
    def __mul__[ST: _ToInt](self: sparray[ST], /, other: _spbase[_ToInt8 | ST]) -> _SpArrayOut[ST]: ...
    @overload  # sparray T@number, dense +i8
    def __mul__[ST: npc.number](self: sparray[ST, _2D], /, other: _To2DLike[bool, _ToInt8]) -> coo_array[ST, _2D]: ...
    @overload  # sparray ~i64, dense ~i64
    def __mul__(self: sparray[np.int64, _2D], /, other: _To2DLike[int, np.int64]) -> coo_array[np.int64, _2D]: ...
    @overload  # sparray T@float, sparse T | +f32
    def __mul__[ST: npc.floating](self: sparray[ST], /, other: _spbase[_ToFloat32 | ST]) -> _SpArrayOut[ST]: ...
    @overload  # sparray T@inexact, dense +f32
    def __mul__[ST: npc.inexact](self: sparray[ST, _2D], /, other: _To2DLike[int, _ToFloat32]) -> coo_array[ST, _2D]: ...
    @overload  # sparray ~f64, dense ~f64
    def __mul__(self: sparray[np.float64, _2D], /, other: _To2DLike[float, np.float64]) -> coo_array[np.float64, _2D]: ...
    @overload  # sparray T@complex, sparse T | +c64
    def __mul__[ST: npc.complexfloating](self: sparray[ST], /, other: _spbase[_ToComplex64 | ST]) -> _SpArrayOut[ST]: ...
    @overload  # sparray T@complex, dense +c64
    def __mul__[ST: npc.complexfloating](
        self: sparray[ST, _2D], /, other: _To2DLike[int, _ToComplex64]
    ) -> coo_array[ST, _2D]: ...
    @overload  # sparray ~c128, dense ~c128
    def __mul__(
        self: sparray[np.complex128, _2D], /, other: _To2DLike[complex, np.complex128]
    ) -> coo_array[np.complex128, _2D]: ...
    @overload  # spmatrix bool, scalar | sparse int
    def __mul__(self: spmatrix[np.bool], /, other: _SparseLike[op.JustInt, npc.integer]) -> _SpMatrix[np.int_ | Any]: ...
    @overload  # spmatrix bool, dense int
    def __mul__(self: spmatrix[np.bool], /, other: _To2DLike[op.JustInt, npc.integer]) -> onp.ArrayND[np.int_ | Any]: ...
    @overload  # spmatrix +int, scalar | sparse float
    def __mul__(self: spmatrix[_ToInt], /, other: _SparseLike[op.JustFloat, npc.floating]) -> _SpMatrix[np.float64 | Any]: ...
    @overload  # spmatrix +int, dense float
    def __mul__(self: spmatrix[_ToInt], /, other: _To2DLike[op.JustFloat, npc.floating]) -> onp.ArrayND[np.float64 | Any]: ...
    @overload  # spmatrix +float, scalar | sparse complex
    def __mul__(
        self: spmatrix[_ToFloat], /, other: _SparseLike[op.JustComplex, npc.complexfloating]
    ) -> _SpMatrix[np.complex128 | Any]: ...
    @overload  # spmatrix +float, dense complex
    def __mul__(
        self: spmatrix[_ToFloat], /, other: _To2DLike[op.JustComplex, npc.complexfloating]
    ) -> onp.ArrayND[np.complex128 | Any]: ...
    @overload  # spmatrix T, sparse bool
    def __mul__[ST: _Scalar](self: spmatrix[ST], /, other: _spbase[np.bool]) -> _SpMatrixOut[ST]: ...
    @overload  # spmatrix T, dense +bool
    def __mul__[ST: _Scalar](self: spmatrix[ST], /, other: _To2DLike[bool, np.bool]) -> onp.ArrayND[ST]: ...
    @overload  # spmatrix T@+int, sparse +i8
    def __mul__[ST: _ToInt](self: spmatrix[ST], /, other: _spbase[_ToInt8]) -> _SpMatrixOut[ST]: ...
    @overload  # spmatrix T@number, dense +i8
    def __mul__[ST: npc.number](self: spmatrix[ST], /, other: _To2DLike[bool, _ToInt8]) -> onp.ArrayND[ST]: ...
    @overload  # spmatrix ~i64, dense ~i64
    def __mul__(self: spmatrix[np.int64], /, other: _To2DLike[int, np.int64]) -> onp.ArrayND[np.int64]: ...
    @overload  # spmatrix T@float, sparse T | +f32
    def __mul__[ST: npc.floating](self: spmatrix[ST], /, other: _spbase[_ToFloat32 | ST]) -> _SpMatrixOut[ST]: ...
    @overload  # spmatrix T@inexact, dense +f32
    def __mul__[ST: npc.inexact](self: spmatrix[ST], /, other: _To2DLike[int, _ToFloat32]) -> onp.ArrayND[ST]: ...
    @overload  # spmatrix ~f64, dense ~f64
    def __mul__(self: spmatrix[np.float64], /, other: _To2DLike[float, np.float64]) -> onp.ArrayND[np.float64]: ...
    @overload  # spmatrix T@complex, sparse T | +c64
    def __mul__[ST: npc.complexfloating](self: spmatrix[ST], /, other: _spbase[_ToComplex64 | ST]) -> _SpMatrixOut[ST]: ...
    @overload  # spmatrix T@complex, dense +c64
    def __mul__[ST: npc.complexfloating](self: spmatrix[ST], /, other: _To2DLike[int, _ToComplex64]) -> onp.ArrayND[ST]: ...
    @overload  # spmatrix ~c128, dense ~c128
    def __mul__(self: spmatrix[np.complex128], /, other: _To2DLike[complex, np.complex128]) -> onp.ArrayND[np.complex128]: ...
    @overload  # sparse bool, scalar T@number
    def __mul__[ST: npc.number](self: _dia_base[np.bool], /, other: ST) -> _spbase[ST, _2D]: ...
    @overload  # sparse +int, scalar T@inexact
    def __mul__[ST: npc.inexact](self: _dia_base[_ToInt], /, other: ST) -> _spbase[ST, _2D]: ...
    @overload  # sparse +float, scalar T@complex
    def __mul__[ST: npc.complexfloating](self: _dia_base[_ToFloat], /, other: ST) -> _spbase[ST, _2D]: ...
    @overload  # fallback
    def __mul__(self, /, other: _To2DLike[complex, _Scalar] | _spbase) -> _spbase[Any] | onp.ArrayND[Any]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    # NOTE: without this, mypy uses `_spbase.__rmul__` for `dia * dia` with different scalar types
    __rmul__ = __mul__  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

    # NOTE: keep in sync with `_spbase.multiply` and `__mul__`
    @override
    @overload  # sparse, scalar +bool
    def multiply(self, /, other: bool | np.bool) -> Self: ...
    @overload  # sparse T@number, scalar +int
    def multiply[SelfT: _dia_base[npc.number]](self: SelfT, /, other: onp.ToInt) -> SelfT: ...
    @overload  # sparse T@inexact, scalar +float
    def multiply[SelfT: _dia_base[npc.inexact]](self: SelfT, /, other: onp.ToFloat) -> SelfT: ...
    @overload  # sparse T@complex, scalar +complex
    def multiply[SelfT: _dia_base[npc.complexfloating]](self: SelfT, /, other: onp.ToComplex) -> SelfT: ...
    @overload  # sparse +int, scalar ~float | ~f64
    def multiply[OutT](self: _CanAsFloat64[_ToInt, OutT], /, other: op.JustFloat | np.float64) -> OutT: ...
    @overload  # sparse ~f64 | +int, scalar ~complex | ~c128
    def multiply[OutT](self: _CanAsComplex128[np.float64 | _ToInt, OutT], /, other: op.JustComplex | np.complex128) -> OutT: ...
    @overload  # dia T@+int, dia T | +i8
    def multiply[SelfT: _dia_base[_ToInt]](self: SelfT, /, other: _dia_base[_ToInt8 | _ScalarT_co]) -> SelfT: ...
    @overload  # dia T@float, dia T | +f32
    def multiply[SelfT: _dia_base[npc.floating]](self: SelfT, /, other: _dia_base[_ToFloat32 | _ScalarT_co]) -> SelfT: ...
    @overload  # dia T@complex, dia T | +c64
    def multiply[SelfT: _dia_base[npc.complexfloating]](
        self: SelfT, /, other: _dia_base[_ToComplex64 | _ScalarT_co]
    ) -> SelfT: ...
    @overload  # sparray, dense +bool
    def multiply[ST: _Scalar](self: sparray[ST, _2D], /, other: _To2DLike[bool, np.bool]) -> coo_array[ST, _2D]: ...
    @overload  # sparray T@+int, sparse T | +i8
    def multiply[ST: _ToInt](self: sparray[ST], /, other: _spbase[_ToInt8 | ST]) -> _SpArrayOut[ST]: ...
    @overload  # sparray T@number, dense +i8
    def multiply[ST: npc.number](self: sparray[ST, _2D], /, other: _To2DLike[bool, _ToInt8]) -> coo_array[ST, _2D]: ...
    @overload  # sparray ~i64, dense ~i64
    def multiply(self: sparray[np.int64, _2D], /, other: _To2DLike[int, np.int64]) -> coo_array[np.int64, _2D]: ...
    @overload  # sparray T@float, sparse T | +f32
    def multiply[ST: npc.floating](self: sparray[ST], /, other: _spbase[_ToFloat32 | ST]) -> _SpArrayOut[ST]: ...
    @overload  # sparray T@inexact, dense +f32
    def multiply[ST: npc.inexact](self: sparray[ST, _2D], /, other: _To2DLike[int, _ToFloat32]) -> coo_array[ST, _2D]: ...
    @overload  # sparray ~f64, dense ~f64
    def multiply(self: sparray[np.float64, _2D], /, other: _To2DLike[float, np.float64]) -> coo_array[np.float64, _2D]: ...
    @overload  # sparray T@complex, sparse T | +c64
    def multiply[ST: npc.complexfloating](self: sparray[ST], /, other: _spbase[_ToComplex64 | ST]) -> _SpArrayOut[ST]: ...
    @overload  # sparray T@complex, dense +c64
    def multiply[ST: npc.complexfloating](
        self: sparray[ST, _2D], /, other: _To2DLike[int, _ToComplex64]
    ) -> coo_array[ST, _2D]: ...
    @overload  # sparray ~c128, dense ~c128
    def multiply(
        self: sparray[np.complex128, _2D], /, other: _To2DLike[complex, np.complex128]
    ) -> coo_array[np.complex128, _2D]: ...
    @overload  # spmatrix bool, scalar | sparse int
    def multiply(self: spmatrix[np.bool], /, other: _SparseLike[op.JustInt, _IntT]) -> _SpMatrix[_IntT]: ...
    @overload  # spmatrix bool, dense int
    def multiply(self: spmatrix[np.bool], /, other: _To2DLike[op.JustInt, _IntT]) -> coo_matrix[_IntT]: ...
    @overload  # spmatrix +i16, scalar | sparse float
    def multiply(self: spmatrix[_ToInt16], /, other: _SparseLike[op.JustFloat, _FloatT]) -> _SpMatrix[_FloatT]: ...
    @overload  # spmatrix +i16, dense float
    def multiply(self: spmatrix[_ToInt16], /, other: _To2DLike[op.JustFloat, _FloatT]) -> coo_matrix[_FloatT]: ...
    @overload  # spmatrix i32 | i64, scalar | sparse float
    def multiply(
        self: spmatrix[npc.integer32 | npc.integer64],
        /,
        other: _SparseLike[op.JustFloat | np.float32 | _spbase[np.float32], _Float64T],
    ) -> _SpMatrix[_Float64T]: ...
    @overload  # spmatrix i32 | i64, dense float
    def multiply(
        self: spmatrix[npc.integer32 | npc.integer64],
        /,
        other: _To2DLike[op.JustFloat | np.float32, _Float64T] | onp.CanArrayND[np.float32],
    ) -> coo_matrix[_Float64T]: ...
    @overload  # spmatrix f32, scalar | sparse complex
    def multiply(self: spmatrix[np.float32], /, other: _SparseLike[op.JustComplex, _Complex64T]) -> _SpMatrix[_Complex64T]: ...
    @overload  # spmatrix +i16, scalar | sparse complex
    def multiply(self: spmatrix[_ToInt16], /, other: _SparseLike[op.JustComplex, _ComplexT]) -> _SpMatrix[_ComplexT]: ...
    @overload  # spmatrix f32 | +i16, dense complex
    def multiply(
        self: spmatrix[np.float32 | _ToInt16], /, other: _To2DLike[op.JustComplex, _ComplexT]
    ) -> coo_matrix[_ComplexT]: ...
    @overload  # spmatrix f64 | i32 | i64, scalar | sparse complex
    def multiply(
        self: spmatrix[np.float64 | npc.integer32 | npc.integer64],
        /,
        other: _SparseLike[op.JustComplex | np.complex64 | _spbase[np.complex64], _Complex128T],
    ) -> _SpMatrix[_Complex128T]: ...
    @overload  # spmatrix f64 | i32 | i64, dense complex
    def multiply(
        self: spmatrix[np.float64 | npc.integer32 | npc.integer64],
        /,
        other: _To2DLike[op.JustComplex | np.complex64, _Complex128T] | onp.CanArrayND[np.complex64],
    ) -> coo_matrix[_Complex128T]: ...
    @overload  # spmatrix f80, scalar | sparse | dense complex
    def multiply(
        self: spmatrix[npc.floating80],
        /,
        other: _SparseLike[op.JustComplex, npc.complexfloating] | _To2DLike[op.JustComplex, npc.complexfloating],
    ) -> _SpMatrix[np.clongdouble]: ...
    @overload  # spmatrix T, sparse bool
    def multiply[ST: _Scalar](self: spmatrix[ST], /, other: _spbase[np.bool]) -> _SpMatrixOut[ST]: ...
    @overload  # spmatrix T, dense +bool
    def multiply[ST: _Scalar](self: spmatrix[ST], /, other: _To2DLike[bool, np.bool]) -> coo_matrix[ST]: ...
    @overload  # spmatrix T@+int, sparse +i8
    def multiply[ST: _ToInt](self: spmatrix[ST], /, other: _spbase[_ToInt8]) -> _SpMatrixOut[ST]: ...
    @overload  # spmatrix T@+int, dense +i8
    def multiply[ST: _ToInt](self: spmatrix[ST], /, other: _To2DLike[bool, _ToInt8]) -> coo_matrix[ST]: ...
    @overload  # spmatrix ~i64, dense ~i64
    def multiply(self: spmatrix[np.int64], /, other: _To2DLike[int, np.int64]) -> coo_matrix[np.int64]: ...
    @overload  # spmatrix T@float, sparse T | +f32
    def multiply[ST: npc.floating](self: spmatrix[ST], /, other: _spbase[_ToFloat32 | ST]) -> _SpMatrixOut[ST]: ...
    @overload  # spmatrix T@float, dense +f32
    def multiply[ST: npc.floating](self: spmatrix[ST], /, other: _To2DLike[int, _ToFloat32]) -> coo_matrix[ST]: ...
    @overload  # spmatrix ~f64, dense ~f64
    def multiply(self: spmatrix[np.float64], /, other: _To2DLike[float, np.float64]) -> coo_matrix[np.float64]: ...
    @overload  # spmatrix T@complex, sparse T | +c64
    def multiply[ST: npc.complexfloating](self: spmatrix[ST], /, other: _spbase[_ToComplex64 | ST]) -> _SpMatrixOut[ST]: ...
    @overload  # spmatrix T@complex, dense +c64
    def multiply[ST: npc.complexfloating](self: spmatrix[ST], /, other: _To2DLike[int, _ToComplex64]) -> coo_matrix[ST]: ...
    @overload  # spmatrix ~c128, dense ~c128
    def multiply(self: spmatrix[np.complex128], /, other: _To2DLike[complex, np.complex128]) -> coo_matrix[np.complex128]: ...
    @overload  # sparse bool, scalar T@number
    def multiply[ST: npc.number](self: _dia_base[np.bool], /, other: ST) -> _spbase[ST, _2D]: ...
    @overload  # sparse +int, scalar T@inexact
    def multiply[ST: npc.inexact](self: _dia_base[_ToInt], /, other: ST) -> _spbase[ST, _2D]: ...
    @overload  # sparse +float, scalar T@complex
    def multiply[ST: npc.complexfloating](self: _dia_base[_ToFloat], /, other: ST) -> _spbase[ST, _2D]: ...
    @overload  # sparse, scalar ~int | ~float | ~complex
    def multiply[OutT](self: _CanAsAny[OutT], /, other: op.JustInt | op.JustFloat | op.JustComplex) -> OutT: ...
    @overload  # fallback
    def multiply(self, /, other: _To2DLike[complex, _Scalar] | _spbase) -> _spbase[Any]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    # NOTE: keep in sync with `_spbase.__matmul__` and `dot`
    @override  # type: ignore[override]
    @overload  # dia T@+int, dia T | +i8
    def __matmul__[SpT: _dia_base[_ToInt]](self: SpT, other: _dia_base[_ToInt8 | _ScalarT_co], /) -> SpT: ...
    @overload  # dia T@float, dia T | +f32
    def __matmul__[SpT: _dia_base[npc.floating]](self: SpT, other: _dia_base[_ToFloat32 | _ScalarT_co], /) -> SpT: ...
    @overload  # dia T@complex, dia T | +c64
    def __matmul__[SpT: _dia_base[npc.complexfloating]](self: SpT, other: _dia_base[_ToComplex64 | _ScalarT_co], /) -> SpT: ...
    @overload  # sparray T@+int, sparse T | +i8
    def __matmul__[ST: _ToInt](self: sparray[ST], other: _spbase[_ToInt8 | ST], /) -> _SpArrayOut[ST]: ...
    @overload  # sparray T@float, sparse T | +f32
    def __matmul__[ST: npc.floating](self: sparray[ST], other: _spbase[_ToFloat32 | ST], /) -> _SpArrayOut[ST]: ...
    @overload  # sparray T@complex, sparse T | +c64
    def __matmul__[ST: npc.complexfloating](self: sparray[ST], other: _spbase[_ToComplex64 | ST], /) -> _SpArrayOut[ST]: ...
    @overload  # spmatrix bool, sparse int
    def __matmul__(self: spmatrix[np.bool], other: _spbase[npc.integer], /) -> _SpMatrixOut[np.int_ | Any]: ...
    @overload  # spmatrix bool, dense int
    def __matmul__(self: spmatrix[np.bool], other: _To2DLike[op.JustInt, npc.integer], /) -> onp.ArrayND[np.int_ | Any]: ...
    @overload  # spmatrix +int, sparse float
    def __matmul__(self: spmatrix[_ToInt], other: _spbase[npc.floating], /) -> _SpMatrixOut[np.float64 | Any]: ...
    @overload  # spmatrix +int, dense float
    def __matmul__(self: spmatrix[_ToInt], other: _To2DLike[op.JustFloat, npc.floating], /) -> onp.ArrayND[np.float64 | Any]: ...
    @overload  # spmatrix +float, sparse complex
    def __matmul__(self: spmatrix[_ToFloat], other: _spbase[npc.complexfloating], /) -> _SpMatrixOut[np.complex128 | Any]: ...
    @overload  # spmatrix +float, dense complex
    def __matmul__(
        self: spmatrix[_ToFloat], other: _To2DLike[op.JustComplex, npc.complexfloating], /
    ) -> onp.ArrayND[np.complex128 | Any]: ...
    @overload  # spmatrix T, sparse bool
    def __matmul__[ST: _Scalar](self: spmatrix[ST], other: _spbase[np.bool], /) -> _SpMatrixOut[ST]: ...
    @overload  # sparse T, dense +bool
    def __matmul__[ST: _Scalar](self: _dia_base[ST], other: _To2DLike[bool, np.bool], /) -> onp.ArrayND[ST]: ...
    @overload  # spmatrix T@+int, sparse +i8
    def __matmul__[ST: _ToInt](self: spmatrix[ST], other: _spbase[_ToInt8], /) -> _SpMatrixOut[ST]: ...
    @overload  # sparse T@number, dense +i8
    def __matmul__[ST: npc.number](self: _dia_base[ST], other: _To2DLike[bool, _ToInt8], /) -> onp.ArrayND[ST]: ...
    @overload  # sparse ~i64, dense ~i64
    def __matmul__(self: _dia_base[np.int64], other: _To2DLike[int, np.int64], /) -> onp.ArrayND[np.int64]: ...
    @overload  # spmatrix T@float, sparse T | +f32
    def __matmul__[ST: npc.floating](self: spmatrix[ST], other: _spbase[_ToFloat32 | ST], /) -> _SpMatrixOut[ST]: ...
    @overload  # sparse T@inexact, dense +f32
    def __matmul__[ST: npc.inexact](self: _dia_base[ST], other: _To2DLike[int, _ToFloat32], /) -> onp.ArrayND[ST]: ...
    @overload  # sparse ~f64, dense ~f64
    def __matmul__(self: _dia_base[np.float64], other: _To2DLike[float, np.float64], /) -> onp.ArrayND[np.float64]: ...
    @overload  # spmatrix T@complex, sparse T | +c64
    def __matmul__[ST: npc.complexfloating](self: spmatrix[ST], other: _spbase[_ToComplex64 | ST], /) -> _SpMatrixOut[ST]: ...
    @overload  # sparse T@complex, dense +c64
    def __matmul__[ST: npc.complexfloating](self: _dia_base[ST], other: _To2DLike[int, _ToComplex64], /) -> onp.ArrayND[ST]: ...
    @overload  # sparse ~c128, dense ~c128
    def __matmul__(self: _dia_base[np.complex128], other: _To2DLike[complex, np.complex128], /) -> onp.ArrayND[np.complex128]: ...
    @overload  # fallback
    def __matmul__(self, other: _To2DLike[complex, _Scalar] | _spbase, /) -> _spbase[Any] | onp.ArrayND[Any]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    # NOTE: without this, mypy uses `_spbase.__rmatmul__` for `dia @ dia` with different scalar types
    __rmatmul__ = __matmul__

    # NOTE: keep in sync with `_spbase.dot` and `__matmul__`
    @override  # type: ignore[override]
    @overload  # sparse, scalar +bool
    def dot(self, /, other: bool | np.bool) -> Self: ...
    @overload  # sparse T@number, scalar +int
    def dot[SelfT: _dia_base[npc.number]](self: SelfT, /, other: onp.ToInt) -> SelfT: ...
    @overload  # sparse T@inexact, scalar +float
    def dot[SelfT: _dia_base[npc.inexact]](self: SelfT, /, other: onp.ToFloat) -> SelfT: ...
    @overload  # sparse T@complex, scalar +complex
    def dot[SelfT: _dia_base[npc.complexfloating]](self: SelfT, /, other: onp.ToComplex) -> SelfT: ...
    @overload  # sparse +int, scalar ~float | ~f64
    def dot[OutT](self: _CanAsFloat64[_ToInt, OutT], /, other: op.JustFloat | np.float64) -> OutT: ...
    @overload  # sparse ~f64 | +int, scalar ~complex | ~c128
    def dot[OutT](self: _CanAsComplex128[np.float64 | _ToInt, OutT], /, other: op.JustComplex | np.complex128) -> OutT: ...
    @overload  # sparse, scalar ~int | ~float | ~complex
    def dot[OutT](self: _CanAsAny[OutT], /, other: op.JustInt | op.JustFloat | op.JustComplex) -> OutT: ...
    @overload  # dia T@+int, dia T | +i8
    def dot[SelfT: _dia_base[_ToInt]](self: SelfT, /, other: _dia_base[_ToInt8 | _ScalarT_co]) -> SelfT: ...
    @overload  # dia T@float, dia T | +f32
    def dot[SelfT: _dia_base[npc.floating]](self: SelfT, /, other: _dia_base[_ToFloat32 | _ScalarT_co]) -> SelfT: ...
    @overload  # dia T@complex, dia T | +c64
    def dot[SelfT: _dia_base[npc.complexfloating]](self: SelfT, /, other: _dia_base[_ToComplex64 | _ScalarT_co]) -> SelfT: ...
    @overload  # sparray T@+int, sparse T | +i8
    def dot[ST: _ToInt](self: sparray[ST], /, other: _spbase[_ToInt8 | ST]) -> _SpArrayOut[ST]: ...
    @overload  # sparray T@float, sparse T | +f32
    def dot[ST: npc.floating](self: sparray[ST], /, other: _spbase[_ToFloat32 | ST]) -> _SpArrayOut[ST]: ...
    @overload  # sparray T@complex, sparse T | +c64
    def dot[ST: npc.complexfloating](self: sparray[ST], /, other: _spbase[_ToComplex64 | ST]) -> _SpArrayOut[ST]: ...
    @overload  # spmatrix bool, sparse int
    def dot(self: spmatrix[np.bool], /, other: _spbase[npc.integer]) -> _SpMatrixOut[np.int_ | Any]: ...
    @overload  # spmatrix bool, dense int
    def dot(self: spmatrix[np.bool], /, other: _To2DLike[op.JustInt, npc.integer]) -> onp.ArrayND[np.int_ | Any]: ...
    @overload  # spmatrix +int, sparse float
    def dot(self: spmatrix[_ToInt], /, other: _spbase[npc.floating]) -> _SpMatrixOut[np.float64 | Any]: ...
    @overload  # spmatrix +int, dense float
    def dot(self: spmatrix[_ToInt], /, other: _To2DLike[op.JustFloat, npc.floating]) -> onp.ArrayND[np.float64 | Any]: ...
    @overload  # spmatrix +float, sparse complex
    def dot(self: spmatrix[_ToFloat], /, other: _spbase[npc.complexfloating]) -> _SpMatrixOut[np.complex128 | Any]: ...
    @overload  # spmatrix +float, dense complex
    def dot(
        self: spmatrix[_ToFloat], /, other: _To2DLike[op.JustComplex, npc.complexfloating]
    ) -> onp.ArrayND[np.complex128 | Any]: ...
    @overload  # spmatrix T, sparse bool
    def dot[ST: _Scalar](self: spmatrix[ST], /, other: _spbase[np.bool]) -> _SpMatrixOut[ST]: ...
    @overload  # sparse T, dense +bool
    def dot[ST: _Scalar](self: _dia_base[ST], /, other: _To2DLike[bool, np.bool]) -> onp.ArrayND[ST]: ...
    @overload  # spmatrix T@+int, sparse +i8
    def dot[ST: _ToInt](self: spmatrix[ST], /, other: _spbase[_ToInt8]) -> _SpMatrixOut[ST]: ...
    @overload  # sparse T@number, dense +i8
    def dot[ST: npc.number](self: _dia_base[ST], /, other: _To2DLike[bool, _ToInt8]) -> onp.ArrayND[ST]: ...
    @overload  # sparse ~i64, dense ~i64
    def dot(self: _dia_base[np.int64], /, other: _To2DLike[int, np.int64]) -> onp.ArrayND[np.int64]: ...
    @overload  # spmatrix T@float, sparse T | +f32
    def dot[ST: npc.floating](self: spmatrix[ST], /, other: _spbase[_ToFloat32 | ST]) -> _SpMatrixOut[ST]: ...
    @overload  # sparse T@inexact, dense +f32
    def dot[ST: npc.inexact](self: _dia_base[ST], /, other: _To2DLike[int, _ToFloat32]) -> onp.ArrayND[ST]: ...
    @overload  # sparse ~f64, dense ~f64
    def dot(self: _dia_base[np.float64], /, other: _To2DLike[float, np.float64]) -> onp.ArrayND[np.float64]: ...
    @overload  # spmatrix T@complex, sparse T | +c64
    def dot[ST: npc.complexfloating](self: spmatrix[ST], /, other: _spbase[_ToComplex64 | ST]) -> _SpMatrixOut[ST]: ...
    @overload  # sparse T@complex, dense +c64
    def dot[ST: npc.complexfloating](self: _dia_base[ST], /, other: _To2DLike[int, _ToComplex64]) -> onp.ArrayND[ST]: ...
    @overload  # sparse ~c128, dense ~c128
    def dot(self: _dia_base[np.complex128], /, other: _To2DLike[complex, np.complex128]) -> onp.ArrayND[np.complex128]: ...
    @overload  # fallback
    def dot(self, /, other: _To2DLike[complex, _Scalar] | _spbase) -> _spbase[Any] | onp.ArrayND[Any]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

class dia_array(_dia_base[_ScalarT_co], sparray[_ScalarT_co, tuple[int, int]], Generic[_ScalarT_co]):
    # NOTE: These four methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> coo_array[_ScalarT_co, tuple[int, int]]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__[ST: _Scalar](self, sctype: ST, /) -> coo_array[ST, tuple[int, int]]: ...
    @override
    @type_check_only
    def __assoc_as_float32__(self, /) -> dia_array[np.float32]: ...
    @override
    @type_check_only
    def __assoc_as_float64__(self, /) -> dia_array[np.float64]: ...
    @type_check_only
    def __assoc_as_complex128__(self, /) -> dia_array[np.complex128]: ...
    @override
    @type_check_only
    def __assoc_as_any__(self, /) -> dia_array[Any]: ...

    # NOTE: keep in sync with `dia_matrix.__init__`
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_ScalarT_co] | _ToData[_ScalarT_co],
        shape: _ToShapeND | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: float64-like | None
    def __init__(
        self: dia_array[np.float64],
        /,
        arg1: _ToShape2D,
        shape: _ToShapeND | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: bool-like
    def __init__(
        self: dia_array[np.bool],
        /,
        arg1: _ToShape2D,
        shape: _ToShapeND | None = None,
        *,
        dtype: onp.AnyBoolDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: int-like
    def __init__(
        self: dia_array[np.int64],
        /,
        arg1: _ToShape2D,
        shape: _ToShapeND | None = None,
        *,
        dtype: onp.AnyIntDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: complex128-like
    def __init__(
        self: dia_array[np.complex128],
        /,
        arg1: _ToShape2D,
        shape: _ToShapeND | None = None,
        *,
        dtype: onp.AnyComplex128DType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: <known>
    def __init__(
        self,
        /,
        arg1: _ToShape2D,
        shape: _ToShapeND | None = None,
        *,
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: bool-like | None
    def __init__(
        self: dia_array[np.bool],
        /,
        arg1: _ToMatrixPy[bool],
        shape: _ToShapeND | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: int-like | None
    def __init__(
        self: dia_array[np.int_],
        /,
        arg1: _ToMatrixPy[op.JustInt],
        shape: _ToShapeND | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: float64-like | None
    def __init__(
        self: dia_array[np.float64],
        /,
        arg1: _ToMatrixPy[op.JustFloat],
        shape: _ToShapeND | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: complex128-like | None
    def __init__(
        self: dia_array[np.complex128],
        /,
        arg1: _ToMatrixPy[op.JustComplex],
        shape: _ToShapeND | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (positional)
    def __init__(
        self,
        /,
        arg1: onp.ToComplex2D,
        shape: _ToShapeND | None,
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (keyword)
    def __init__(
        self,
        /,
        arg1: onp.ToComplex2D,
        shape: _ToShapeND | None = None,
        *,
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...

# pyrefly: ignore[inconsistent-inheritance]
class dia_matrix(_dia_base[_ScalarT_co], spmatrix[_ScalarT_co], Generic[_ScalarT_co]):
    # NOTE: These four methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> coo_matrix[_ScalarT_co]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__[ST: _Scalar](self, sctype: ST, /) -> coo_matrix[ST]: ...
    @override
    @type_check_only
    def __assoc_as_float32__(self, /) -> dia_matrix[np.float32]: ...
    @override
    @type_check_only
    def __assoc_as_float64__(self, /) -> dia_matrix[np.float64]: ...
    @type_check_only
    def __assoc_as_complex128__(self, /) -> dia_matrix[np.complex128]: ...
    @override
    @type_check_only
    def __assoc_as_any__(self, /) -> dia_matrix[Any]: ...

    # NOTE: keep in sync with `dia_array.__init__`
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_ScalarT_co] | _ToData[_ScalarT_co],
        shape: _ToShapeND | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: dia_matrix[np.float64],
        /,
        arg1: _ToShape2D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: bool-like
    def __init__(
        self: dia_matrix[np.bool],
        /,
        arg1: _ToShape2D,
        shape: _ToShapeND | None = None,
        *,
        dtype: onp.AnyBoolDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: <known>
    def __init__(
        self,
        /,
        arg1: _ToShape2D,
        shape: _ToShapeND | None = None,
        *,
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: bool-like | None
    def __init__(
        self: dia_matrix[np.bool],
        /,
        arg1: _ToMatrixPy[bool],
        shape: _ToShapeND | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: int-like | None
    def __init__(
        self: dia_matrix[np.int_],
        /,
        arg1: _ToMatrixPy[op.JustInt],
        shape: _ToShapeND | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: float64-like | None
    def __init__(
        self: dia_matrix[np.float64],
        /,
        arg1: _ToMatrixPy[op.JustFloat],
        shape: _ToShapeND | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: complex128-like | None
    def __init__(
        self: dia_matrix[np.complex128],
        /,
        arg1: _ToMatrixPy[op.JustComplex],
        shape: _ToShapeND | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (positional)
    def __init__(
        self,
        /,
        arg1: onp.ToComplex2D,
        shape: _ToShapeND | None,
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (keyword)
    def __init__(
        self,
        /,
        arg1: onp.ToComplex2D,
        shape: _ToShapeND | None = None,
        *,
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <unknown>
    def __init__(
        self,
        /,
        arg1: onp.ToComplex2D,
        shape: _ToShapeND | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

def isspmatrix_dia(x: object) -> TypeIs[dia_matrix]: ...
