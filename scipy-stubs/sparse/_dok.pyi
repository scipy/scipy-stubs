# NOTE: Adding `@override` to `@overload`ed methods will crash stubtest (mypy 1.13.0)
# mypy: disable-error-code="misc, override"

from collections.abc import Iterable, Sequence
from typing import Any, ClassVar, Generic, Literal, Never, Self, TypeAlias, overload, type_check_only
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp

from ._base import _spbase, sparray
from ._coo import coo_array, coo_matrix
from ._index import IndexMixin
from ._matrix import spmatrix
from ._typing import Numeric, ToShape2D, ToShapeMin1D

__all__ = ["dok_array", "dok_matrix", "isspmatrix_dok"]

###

_T = TypeVar("_T")
_ScalarT = TypeVar("_ScalarT", bound=Numeric)
_ScalarT_co = TypeVar("_ScalarT_co", bound=Numeric, default=Any, covariant=True)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int, int], covariant=True)

_1D: TypeAlias = tuple[int]  # noqa: PYI042
_2D: TypeAlias = tuple[int, int]  # noqa: PYI042
# workaround for the typing-spec non-conformance regarding overload behavior of mypy and pyright
_NeitherD: TypeAlias = tuple[Never] | tuple[Never, Never]

_ToMatrix: TypeAlias = _spbase[_ScalarT] | onp.CanArrayND[_ScalarT] | Sequence[onp.CanArrayND[_ScalarT]] | _ToMatrixPy[_ScalarT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]

_ToKey1D: TypeAlias = onp.ToJustInt | tuple[onp.ToJustInt]
_ToKey2D: TypeAlias = tuple[onp.ToJustInt, onp.ToJustInt]

_ToKeys1: TypeAlias = Iterable[_ToKey1D]
_ToKeys2: TypeAlias = Iterable[_ToKey2D]
_ToKeys: TypeAlias = Iterable[_ToKey1D | _ToKey2D]

_C2T = TypeVar("_C2T", bound=_dok_base[np.float64, _2D])

###

class _dok_base(  # pyright: ignore[reportIncompatibleMethodOverride]
    _spbase[_ScalarT_co, _ShapeT_co],
    IndexMixin[_ScalarT_co, _ShapeT_co],
    dict[tuple[Any, ...], _ScalarT_co | Any],
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
        shape: ToShapeMin1D | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

    #
    @override
    def __len__(self, /) -> int: ...

    #
    @overload
    def __delitem__(self: _dok_base[Any, _2D], key: _ToKey2D, /) -> None: ...
    @overload
    def __delitem__(self: _dok_base[Any, _1D], key: _ToKey1D, /) -> None: ...
    @overload
    def __delitem__(self, key: _ToKey1D | _ToKey2D, /) -> None: ...

    #
    @override
    def __or__(self, other: Never, /) -> Never: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __ror__(self, other: Never, /) -> Never: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __ior__(self, other: Never, /) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    def count_nonzero(self, /, axis: None = None) -> int: ...
    @overload
    def count_nonzero(self, /, axis: op.CanIndex) -> onp.Array1D[np.intp]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @override
    def update(self, /, val: Never) -> Never: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    def setdefault(self: _dok_base[Any, _2D], key: _ToKey2D, default: _T, /) -> _ScalarT_co | _T: ...
    @overload
    def setdefault(self: _dok_base[Any, _2D], key: _ToKey2D, default: None = None, /) -> _ScalarT_co | None: ...
    @overload
    def setdefault(self: _dok_base[Any, _1D], key: _ToKey1D, default: _T, /) -> _ScalarT_co | _T: ...
    @overload
    def setdefault(self: _dok_base[Any, _1D], key: _ToKey1D, default: None = None, /) -> _ScalarT_co | None: ...
    @overload
    def setdefault(self, key: _ToKey1D | _ToKey2D, default: _T, /) -> _ScalarT_co | _T: ...
    @overload
    def setdefault(self, key: _ToKey1D | _ToKey2D, default: None = None, /) -> _ScalarT_co | None: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    def get(self: _dok_base[Any, _2D], /, key: _ToKey2D, default: _T) -> _ScalarT_co | _T: ...
    @overload
    def get(self: _dok_base[Any, _2D], /, key: _ToKey2D, default: float = 0.0) -> _ScalarT_co | float: ...
    @overload
    def get(self: _dok_base[Any, _1D], /, key: _ToKey1D, default: _T) -> _ScalarT_co | _T: ...
    @overload
    def get(self: _dok_base[Any, _1D], /, key: _ToKey1D, default: float = 0.0) -> _ScalarT_co | float: ...
    @overload
    def get(self, /, key: _ToKey1D | _ToKey2D, default: _T) -> _ScalarT_co | _T: ...
    @overload
    def get(self, /, key: _ToKey1D | _ToKey2D, default: float = 0.0) -> _ScalarT_co | float: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.bool_, _2D]], iterable: _ToKeys2, v: onp.ToBool, /) -> _dok_base[np.bool_, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[np.bool_, _1D]], iterable: _ToKeys1, v: onp.ToBool, /) -> _dok_base[np.bool_, _1D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[_ScalarT, _2D]], iterable: _ToKeys2, v: _ScalarT, /) -> _dok_base[_ScalarT, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_dok_base[_ScalarT, _1D]], iterable: _ToKeys1, v: _ScalarT, /) -> _dok_base[_ScalarT, _1D]: ...
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
        cls: type[_dok_base[np.complex128, _NeitherD]], iterable: _ToKeys, v: op.JustComplex, /
    ) -> _dok_base[np.complex128, tuple[Any, ...]]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[_C2T], iterable: _ToKeys2, v: op.JustComplex, /) -> _C2T: ...
    @overload
    @classmethod
    def fromkeys(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls: type[_dok_base[np.complex128, _1D]], iterable: _ToKeys1, v: op.JustComplex, /
    ) -> _dok_base[np.complex128, _1D]: ...

#
class dok_array(_dok_base[_ScalarT_co, _ShapeT_co], sparray[_ScalarT_co, _ShapeT_co], Generic[_ScalarT_co, _ShapeT_co]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> coo_array[_ScalarT_co, tuple[int, int]]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _ScalarT, /) -> coo_array[_ScalarT, tuple[int, int]]: ...

    # NOTE: keep the 2d overloads in sync with `dok_matrix.__init__`
    # TODO(jorenham): Overloads for specific shape types.
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_ScalarT_co],
        shape: ToShapeMin1D | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: dok_array[np.float64],
        /,
        arg1: ToShapeMin1D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: dok_array[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: dok_array[np.int_],
        /,
        arg1: _ToMatrixPy[op.JustInt],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: dok_array[np.float64],
        /,
        arg1: _ToMatrixPy[op.JustFloat],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: dok_array[np.complex128],
        /,
        arg1: _ToMatrixPy[op.JustComplex],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (positional)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShapeMin1D | None,
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (keyword)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShapeMin1D | None = None,
        *,
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <unknown>
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShapeMin1D | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

    # NOTE: This horrible code duplication is required due to the lack of higher-kinded typing (HKT) support.
    # https://github.com/python/typing/issues/548
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.bool_, _NeitherD]], iterable: _ToKeys, v: onp.ToBool, /
    ) -> dok_array[np.bool_, tuple[Any, ...]]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.bool_, _2D]], iterable: _ToKeys2, v: onp.ToBool, /) -> dok_array[np.bool_, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.bool_, _1D]], iterable: _ToKeys1, v: onp.ToBool, /) -> dok_array[np.bool_, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[_ScalarT, _NeitherD]], iterable: _ToKeys, v: _ScalarT, /
    ) -> dok_array[_ScalarT, tuple[Any, ...]]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[_ScalarT, _2D]], iterable: _ToKeys2, v: _ScalarT, /) -> dok_array[_ScalarT, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[_ScalarT, _1D]], iterable: _ToKeys1, v: _ScalarT, /) -> dok_array[_ScalarT, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.int_, _NeitherD]], iterable: _ToKeys, v: op.JustInt = 1, /
    ) -> dok_array[np.int_, tuple[Any, ...]]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.int_, _2D]], iterable: _ToKeys2, v: op.JustInt = 1, /) -> dok_array[np.int_, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.int_, _1D]], iterable: _ToKeys1, v: op.JustInt = 1, /) -> dok_array[np.int_, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.float64, _NeitherD]], iterable: _ToKeys, v: op.JustFloat, /
    ) -> dok_array[np.float64, tuple[Any, ...]]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.float64, _2D]], iterable: _ToKeys2, v: op.JustFloat, /) -> dok_array[np.float64, _2D]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_array[np.float64, _1D]], iterable: _ToKeys1, v: op.JustFloat, /) -> dok_array[np.float64, _1D]: ...
    @overload
    @classmethod
    def fromkeys(
        cls: type[dok_array[np.complex128, _NeitherD]], iterable: _ToKeys, v: op.JustComplex, /
    ) -> dok_array[np.complex128, tuple[Any, ...]]: ...
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
class dok_matrix(_dok_base[_ScalarT_co, _2D], spmatrix[_ScalarT_co], Generic[_ScalarT_co]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> coo_matrix[_ScalarT_co]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _ScalarT, /) -> coo_matrix[_ScalarT]: ...

    # NOTE: keep the in sync with `dok_array.__init__`
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_ScalarT_co],
        shape: ToShape2D | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: dok_matrix[np.float64],
        /,
        arg1: ToShape2D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: dok_matrix[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: dok_matrix[np.int_],
        /,
        arg1: _ToMatrixPy[op.JustInt],
        shape: ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: dok_matrix[np.float64],
        /,
        arg1: _ToMatrixPy[op.JustFloat],
        shape: ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: dok_matrix[np.complex128],
        /,
        arg1: _ToMatrixPy[op.JustComplex],
        shape: ToShape2D | None = None,
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
        shape: ToShape2D | None,
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
        shape: ToShape2D | None = None,
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
        shape: ToShape2D | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

    #
    @override
    def get(self, /, key: _ToKey2D, default: onp.ToComplex = 0.0) -> _ScalarT_co: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def setdefault(self, key: _ToKey2D, default: onp.ToComplex | None = None, /) -> _ScalarT_co: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    @classmethod
    def fromkeys(cls: type[dok_matrix[np.bool_]], iterable: _ToKeys2, v: onp.ToBool, /) -> dok_matrix[np.bool_]: ...
    @overload
    @classmethod
    def fromkeys(cls: type[dok_matrix[_ScalarT]], iterable: _ToKeys2, v: _ScalarT, /) -> dok_matrix[_ScalarT]: ...
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
