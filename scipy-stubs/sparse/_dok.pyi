# NOTE: Adding `@override` to `@overload`ed methods will crash stubtest (mypy 1.13.0)
# mypy: disable-error-code="misc, override"

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, ClassVar, Generic, Literal, Never, Self, SupportsIndex, overload, override, type_check_only
from typing_extensions import TypeIs, TypeVar

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from ._base import _spbase, sparray
from ._coo import coo_array, coo_matrix
from ._index import IndexMixin
from ._matrix import spmatrix
from ._typing import _ToShape1D, _ToShape2D

__all__ = ["dok_array", "dok_matrix", "isspmatrix_dok"]

###

type _1D = tuple[int]  # ruff: ignore[snake-case-type-alias]
type _2D = tuple[int, int]  # ruff: ignore[snake-case-type-alias]
type _Scalar = npc.number | np.bool

# workaround for the typing-spec non-conformance regarding overload behavior of mypy and pyright
type _NoD = tuple[Never] | tuple[Never, Never]
type _AnyD = tuple[Any, ...]

type _ToMatrix[ST: _Scalar] = _spbase[ST] | onp.CanArrayND[ST] | Sequence[onp.CanArrayND[ST]] | _ToMatrixPy[ST]
type _ToMatrixPy[T] = Sequence[T] | Sequence[Sequence[T]]

type _ToKey1D = onp.ToJustInt | tuple[onp.ToJustInt]
type _ToKey2D = tuple[onp.ToJustInt, onp.ToJustInt]

type _ToKeys1 = Iterable[_ToKey1D]
type _ToKeys2 = Iterable[_ToKey2D]
type _ToKeys = Iterable[_ToKey1D | _ToKey2D]

_ScalarT_co = TypeVar("_ScalarT_co", bound=_Scalar, default=Any, covariant=True)
_ShapeT_co = TypeVar("_ShapeT_co", bound=_1D | _2D, default=_2D, covariant=True)

###

class _dok_base(  # pyright: ignore[reportIncompatibleMethodOverride]  # ty:ignore[invalid-method-override]
    _spbase[_ScalarT_co, _ShapeT_co],
    IndexMixin[_ScalarT_co, _ShapeT_co],
    dict[tuple[Any, ...], _ScalarT_co | Any],  # pyrefly:ignore[invalid-variance] # ty:ignore[invalid-generic-class]
    Generic[_ScalarT_co, _ShapeT_co],
):
    _format: ClassVar = "dok"
    _allow_nd: ClassVar = 1, 2

    dtype: np.dtype[_ScalarT_co]

    @property
    @override
    def format(self, /) -> Literal["dok"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[1, 2]: ...
    @property
    @override
    def shape(self, /) -> _ShapeT_co: ...

    #
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: _ToShape1D | _ToShape2D | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

    #
    @override
    def __len__(self, /) -> int: ...
    @override
    def __delitem__[ShapeT: _1D | _2D](self: _dok_base[Any, ShapeT], key: ShapeT, /) -> None: ...  # pyrefly: ignore[bad-override]  # ty:ignore[invalid-method-override]

    #
    @override
    @overload
    def setdefault[ST: _Scalar, ShapeT: _1D | _2D, T](  # pyrefly: ignore[bad-override]
        self: _dok_base[ST, ShapeT], key: ShapeT, default: T, /
    ) -> ST | T: ...
    @overload
    def setdefault[ST: _Scalar, ShapeT: _1D | _2D](  # pyright: ignore[reportIncompatibleMethodOverride]
        self: _dok_base[ST, ShapeT], key: ShapeT, default: None = None, /
    ) -> ST | None: ...  # ty:ignore[invalid-method-override]

    #
    @override
    @overload
    def get[ST: _Scalar, ShapeT: _1D | _2D, T](  # pyrefly: ignore[bad-override]
        self: _dok_base[ST, ShapeT], /, key: ShapeT, default: T
    ) -> ST | T: ...
    @overload
    def get[ST: _Scalar, ShapeT: _1D | _2D](self: _dok_base[ST, ShapeT], /, key: ShapeT, default: float = 0.0) -> ST | float: ...  # pyright: ignore[reportIncompatibleMethodOverride] # ty: ignore[invalid-method-override]

    #
    @override
    def __or__(self, other: Never, /) -> Never: ...  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
    @override
    def __ror__(self, other: Never, /) -> Never: ...  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
    @override
    def __ior__(self, other: Never, /) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
    @override
    def update(self, /, val: Never) -> Never: ...  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]

    #
    @override
    @overload
    def count_nonzero(self, /, axis: None = None) -> np.intp: ...
    @overload
    def count_nonzero(self: _dok_base[Any, tuple[int, int]], /, axis: SupportsIndex) -> onp.Array1D[np.intp]: ...

    #
    @override
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.bool, _2D]], iterable: _ToKeys2, v: bool, /) -> _dok_base[np.bool, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.bool, _1D]], iterable: _ToKeys1, v: bool, /) -> _dok_base[np.bool, _1D]: ...
    @overload
    @classmethod
    def fromkeys[ST: _Scalar](cls: type[_dok_base[ST, _2D]], iterable: _ToKeys2, v: ST, /) -> _dok_base[ST, _2D]: ...
    @overload
    @classmethod
    def fromkeys[ST: _Scalar](cls: type[_dok_base[ST, _1D]], iterable: _ToKeys1, v: ST, /) -> _dok_base[ST, _1D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.int_, _2D]], iterable: _ToKeys2, v: op.JustInt = 1, /) -> _dok_base[np.int_, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.int_, _1D]], iterable: _ToKeys1, v: op.JustInt = 1, /) -> _dok_base[np.int_, _1D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.float64, _2D]], iterable: _ToKeys2, v: op.JustFloat, /) -> _dok_base[np.float64, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.float64, _1D]], iterable: _ToKeys1, v: op.JustFloat, /) -> _dok_base[np.float64, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[_dok_base[np.complex128, _NoD]], iterable: _ToKeys, v: op.JustComplex, /
    ) -> _dok_base[np.complex128, _AnyD]: ...
    @overload
    @classmethod
    def fromkeys[SelfT: _dok_base[np.float64, _2D]](cls: type[SelfT], iterable: _ToKeys2, v: op.JustComplex, /) -> SelfT: ...
    @overload
    @classmethod
    def fromkeys(  # pyright:ignore[reportIncompatibleMethodOverride] # ty:ignore[invalid-method-override]q
        cls: type[_dok_base[np.complex128, _1D]], iterable: _ToKeys1, v: op.JustComplex, /
    ) -> _dok_base[np.complex128, _1D]: ...

#
class dok_array(_dok_base[_ScalarT_co, _ShapeT_co], sparray[_ScalarT_co, _ShapeT_co], Generic[_ScalarT_co, _ShapeT_co]):  # ty:ignore[invalid-method-override]
    # NOTE: These four methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> coo_array[_ScalarT_co, _2D]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__[ST: _Scalar](self, sctype: ST, /) -> coo_array[ST, _2D]: ...
    @type_check_only
    def __assoc_as_float32__(self, /) -> dok_array[np.float32, _ShapeT_co]: ...
    @type_check_only
    def __assoc_as_float64__(self, /) -> dok_array[np.float64, _ShapeT_co]: ...
    @override
    @type_check_only
    def __assoc_as_any__(self, /) -> dok_array[Any, _ShapeT_co]: ...

    # NOTE: keep the 2d overloads in sync with `dok_matrix.__init__`
    # TODO(jorenham): Overloads for specific shape types.
    @overload  # sparse or dense (know dtype & shape), dtype: None
    def __init__(
        self,
        /,
        arg1: _spbase[_ScalarT_co, _ShapeT_co] | onp.CanArrayND[_ScalarT_co, _ShapeT_co],
        shape: _ShapeT_co | None = None,
        dtype: onp.ToDType[_ScalarT_co] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like (know dtype), dtype: None
    def __init__[ST: _Scalar](
        self: dok_array[ST, _1D],
        /,
        arg1: Sequence[ST],
        shape: _ToShape1D | None = None,
        dtype: onp.ToDType[ST] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like (know dtype), dtype: None
    def __init__[ST: _Scalar](
        self: dok_array[ST, _2D],
        /,
        arg1: Sequence[Sequence[ST] | onp.CanArrayND[ST]],  # assumes max. 2-d
        shape: _ToShape2D | None = None,
        dtype: onp.ToDType[ST] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like (known dtype), dtype: None
    def __init__[ST: _Scalar](
        self: dok_array[ST, _AnyD],
        /,
        arg1: _ToMatrix[ST],
        shape: _ToShape1D | _ToShape2D | None = None,
        dtype: onp.ToDType[ST] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: float64-like | None
    def __init__(
        self: dok_array[np.float64, _1D],
        /,
        arg1: _ToShape1D,
        shape: _ToShape1D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: float64-like | None
    def __init__(
        self: dok_array[np.float64, _2D],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: bool-like
    def __init__(
        self: dok_array[np.bool, _1D],
        /,
        arg1: _ToShape1D,
        shape: _ToShape1D | None = None,
        *,
        dtype: onp.AnyBoolDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: bool-like
    def __init__(
        self: dok_array[np.bool, _2D],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.AnyBoolDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: int-like
    def __init__(
        self: dok_array[np.int64, _1D],
        /,
        arg1: _ToShape1D,
        shape: _ToShape1D | None = None,
        *,
        dtype: onp.AnyIntDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: int-like
    def __init__(
        self: dok_array[np.int64, _2D],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.AnyIntDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: complex128-like
    def __init__(
        self: dok_array[np.complex128, _1D],
        /,
        arg1: _ToShape1D,
        shape: _ToShape1D | None = None,
        *,
        dtype: onp.AnyComplex128DType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: complex128-like
    def __init__(
        self: dok_array[np.complex128, _2D],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.AnyComplex128DType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: <known>
    def __init__[ST: _Scalar](
        self: dok_array[ST, _1D],
        /,
        arg1: _ToShape1D,
        shape: _ToShape1D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: <known>
    def __init__[ST: _Scalar](
        self: dok_array[ST, _2D],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like bool, dtype: bool-like | None
    def __init__(
        self: dok_array[np.bool, _1D],
        /,
        arg1: onp.ToJustBoolStrict1D,
        shape: _ToShape1D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like bool, dtype: bool-like | None
    def __init__(
        self: dok_array[np.bool, _2D],
        /,
        arg1: onp.ToJustBoolStrict2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like ~int, dtype: int-like | None
    def __init__(
        self: dok_array[np.int64, _1D],
        /,
        arg1: onp.ToJustInt64Strict1D,
        shape: _ToShape1D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~int, dtype: int-like | None
    def __init__(
        self: dok_array[np.int64, _2D],
        /,
        arg1: onp.ToJustInt64Strict2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like ~float, dtype: float64-like | None
    def __init__(
        self: dok_array[np.float64, _1D],
        /,
        arg1: onp.ToJustFloat64Strict1D,
        shape: _ToShape1D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~float, dtype: float64-like | None
    def __init__(
        self: dok_array[np.float64, _2D],
        /,
        arg1: onp.ToJustFloat64Strict2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like ~complex, dtype: complex128-like | None
    def __init__(
        self: dok_array[np.complex128, _1D],
        /,
        arg1: onp.ToJustComplex128Strict1D,
        shape: _ToShape1D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~complex, dtype: complex128-like | None
    def __init__(
        self: dok_array[np.complex128, _2D],
        /,
        arg1: onp.ToJustComplex128Strict2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (positional)
    def __init__[ST: _Scalar](
        self: dok_array[ST, _1D],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: _ToShape1D | None,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (keyword)
    def __init__[ST: _Scalar](
        self: dok_array[ST, _1D],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: _ToShape1D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (positional)
    def __init__[ST: _Scalar](
        self: dok_array[ST, _2D],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: _ToShape2D | None,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (keyword)
    def __init__[ST: _Scalar](
        self: dok_array[ST, _2D],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <unknown>
    def __init__(
        self: dok_array[Any, _AnyD],
        /,
        arg1: onp.ToComplex1D | onp.ToComplex2D,
        shape: _ToShape1D | _ToShape2D | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

    #
    @override
    @overload
    # pyrefly: ignore[bad-override]
    def __getitem__(self, key: onp.CanArrayND[np.bool | npc.integer] | list[int] | slice, /) -> Self: ...
    @overload
    def __getitem__[ST: _Scalar, ShapeT: _1D | _2D](
        self: dok_array[ST, ShapeT], key: _spbase[np.bool, ShapeT], /
    ) -> dok_array[ST, ShapeT]: ...
    @overload
    def __getitem__[ST: _Scalar](self: dok_array[ST, _NoD], key: _ToKey1D, /) -> Any: ...
    @overload
    def __getitem__[ST: _Scalar](self: dok_array[ST, _2D], key: _ToKey2D, /) -> ST: ...
    @overload
    def __getitem__[ST: _Scalar](self: dok_array[ST, _1D], key: _ToKey1D, /) -> ST: ...
    @overload
    def __getitem__[ST: _Scalar](self: dok_array[ST, _2D], key: _ToKey1D, /) -> coo_array[ST, _1D]: ...  # pyright:ignore[reportIncompatibleMethodOverride] # ty:ignore[invalid-method-override]

    # NOTE: This horrible code duplication is required due to the lack of higher-kinded typing (HKT) support.
    # https://github.com/python/typing/issues/548
    @override
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.bool, _NoD]], iterable: _ToKeys, v: bool, /) -> dok_array[np.bool, _AnyD]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.bool, _2D]], iterable: _ToKeys2, v: bool, /) -> dok_array[np.bool, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.bool, _1D]], iterable: _ToKeys1, v: bool, /) -> dok_array[np.bool, _1D]: ...
    @overload
    @classmethod
    def fromkeys[ST: _Scalar](cls: type[dok_array[ST, _NoD]], iterable: _ToKeys, v: ST, /) -> dok_array[ST, _AnyD]: ...
    @overload
    @classmethod
    def fromkeys[ST: _Scalar](cls: type[dok_array[ST, _2D]], iterable: _ToKeys2, v: ST, /) -> dok_array[ST, _2D]: ...
    @overload
    @classmethod
    def fromkeys[ST: _Scalar](cls: type[dok_array[ST, _1D]], iterable: _ToKeys1, v: ST, /) -> dok_array[ST, _1D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.int_, _NoD]], iterable: _ToKeys, v: op.JustInt = 1, /) -> dok_array[np.int_, _AnyD]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.int_, _2D]], iterable: _ToKeys2, v: op.JustInt = 1, /) -> dok_array[np.int_, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.int_, _1D]], iterable: _ToKeys1, v: op.JustInt = 1, /) -> dok_array[np.int_, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.float64, _NoD]], iterable: _ToKeys, v: op.JustFloat, /
    ) -> dok_array[np.float64, _AnyD]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.float64, _2D]], iterable: _ToKeys2, v: op.JustFloat, /) -> dok_array[np.float64, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.float64, _1D]], iterable: _ToKeys1, v: op.JustFloat, /) -> dok_array[np.float64, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.complex128, _NoD]], iterable: _ToKeys, v: op.JustComplex, /
    ) -> dok_array[np.complex128, _AnyD]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.complex128, _2D]], iterable: _ToKeys2, v: op.JustComplex, /
    ) -> dok_array[np.complex128, _2D]: ...
    @overload
    @classmethod
    def fromkeys(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type[dok_array[np.complex128, _1D]], iterable: _ToKeys1, v: op.JustComplex, /
    ) -> dok_array[np.complex128, _1D]: ...

#
class dok_matrix(_dok_base[_ScalarT_co, _2D], spmatrix[_ScalarT_co], Generic[_ScalarT_co]):  # ty:ignore[invalid-method-override]
    # NOTE: These four methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> coo_matrix[_ScalarT_co]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__[ST: _Scalar](self, sctype: ST, /) -> coo_matrix[ST]: ...
    @type_check_only
    def __assoc_as_float32__(self, /) -> dok_matrix[np.float32]: ...
    @type_check_only
    def __assoc_as_float64__(self, /) -> dok_matrix[np.float64]: ...
    @override
    @type_check_only
    def __assoc_as_any__(self, /) -> dok_matrix[Any]: ...

    #
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...

    # NOTE: keep the in sync with `dok_array.__init__`
    @overload  # matrix-like (known dtype), dtype: None
    def __init__[ST: _Scalar](
        self: dok_matrix[ST],  # this self annotation works around a mypy bug
        /,
        arg1: _ToMatrix[ST],
        shape: _ToShape2D | None = None,
        dtype: onp.ToDType[ST] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: dok_matrix[np.float64],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: bool-like
    def __init__(
        self: dok_matrix[np.bool],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.AnyBoolDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: <known>
    def __init__[ST: _Scalar](
        self: dok_matrix[ST],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: bool-like | None
    def __init__(
        self: dok_matrix[np.bool],
        /,
        arg1: onp.ToJustBoolStrict2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: int-like | None
    def __init__(
        self: dok_matrix[np.int64],
        /,
        arg1: onp.ToJustInt64Strict2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: float64-like | None
    def __init__(
        self: dok_matrix[np.float64],
        /,
        arg1: onp.ToJustFloat64Strict2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: complex128-like | None
    def __init__(
        self: dok_matrix[np.complex128],
        /,
        arg1: onp.ToJustComplex128Strict2D,
        shape: _ToShape2D | None = None,
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
        shape: _ToShape2D | None,
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
        shape: _ToShape2D | None = None,
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
        shape: _ToShape2D | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

    #
    @override
    @overload
    def __getitem__(  # pyrefly: ignore[bad-override]
        self, key: _ToKey1D | onp.CanArrayND[np.bool | npc.integer] | _spbase[np.bool, _2D] | list[int] | slice, /
    ) -> Self: ...
    @overload
    def __getitem__(self, key: _ToKey2D, /) -> _ScalarT_co: ...  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]

    #
    @override
    def __reversed__(self, /) -> Iterator[tuple[int, int]]: ...

    #
    @override
    @overload
    @classmethod
    def fromkeys(cls: type[dok_matrix[np.bool]], iterable: _ToKeys2, v: bool, /) -> dok_matrix[np.bool]: ...
    @overload
    @classmethod
    def fromkeys[ST: _Scalar](cls: type[dok_matrix[ST]], iterable: _ToKeys2, v: ST, /) -> dok_matrix[ST]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_matrix[np.int_]], iterable: _ToKeys2, v: op.JustInt = 1, /) -> dok_matrix[np.int_]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_matrix[np.float64]], iterable: _ToKeys2, v: op.JustFloat, /) -> dok_matrix[np.float64]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_matrix[np.complex128]], iterable: _ToKeys2, v: op.JustComplex, /) -> dok_matrix[np.complex128]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

#
def isspmatrix_dok(x: object) -> TypeIs[dok_matrix]: ...
