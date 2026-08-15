# mypy: disable-error-code="explicit-override"

from collections.abc import Sequence
from types import EllipsisType
from typing import Any, ClassVar, Generic, Literal, Never, Self, SupportsIndex, overload, override, type_check_only
from typing_extensions import TypeIs, TypeVar

import numpy as np
import numpy.typing as npt
import numpy_typing_compat as nptc
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from ._base import _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._matrix import spmatrix
from ._typing import _ToShape1D, _ToShape2D

__all__ = ["coo_array", "coo_matrix", "isspmatrix_coo"]

###

type _Scalar = npc.number | np.bool

type _ToData[ST: _Scalar] = tuple[
    onp.ArrayND[ST],
    tuple[onp.ArrayND[npc.integer], onp.ArrayND[npc.integer]] | tuple[onp.ArrayND[npc.integer]],
]  # fmt: skip

type _ScalarOrDense[ST: _Scalar] = onp.ArrayND[ST] | ST
type _JustND[T] = onp.SequenceND[op.Just[T]]

type _Axes = int | tuple[Sequence[int], Sequence[int]]
type _ToShapeMin3D = tuple[SupportsIndex, SupportsIndex, SupportsIndex, *tuple[SupportsIndex, ...]]  # ndim > 2

type _IndexSlice = slice | EllipsisType | list[int] | _spbase[np.intp] | onp.ArrayND[np.intp]

_ScalarT_co = TypeVar("_ScalarT_co", bound=_Scalar, default=Any, covariant=True)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int, *tuple[int, ...]], default=tuple[Any, ...], covariant=True)

###

class _coo_base(_data_matrix[_ScalarT_co, _ShapeT_co], _minmax_mixin[_ScalarT_co, _ShapeT_co], Generic[_ScalarT_co, _ShapeT_co]):
    _format: ClassVar = "coo"
    _allow_nd: ClassVar[Sequence[int]] = ...  # range(1, 65)

    data: onp.Array1D[_ScalarT_co]
    coords: tuple[onp.Array1D[np.int32], ...]  # len(coords) == ndim
    has_canonical_format: bool

    @property
    @override
    def format(self, /) -> Literal["coo"]: ...
    #
    @property
    @override
    def shape(self, /) -> _ShapeT_co: ...
    #
    @property
    def row(self, /) -> onp.Array1D[np.int32]: ...
    @row.setter
    def row(self, row: onp.ToInt1D, /) -> None: ...
    #
    @property
    def col(self, /) -> onp.Array1D[np.int32]: ...
    @col.setter
    def col(self, col: onp.ToInt1D, /) -> None: ...

    #
    def __init__(
        self,
        /,
        arg1: onp.ToComplex1D | onp.ToComplex2D,
        shape: tuple[SupportsIndex, *tuple[SupportsIndex, ...]] | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

    #
    def sum_duplicates(self, /) -> None: ...
    def eliminate_zeros(self, /) -> None: ...

    # NOTE: all combinations (self dtype, other dtype, self shape, other shape, self: array|matrix, other dense|sparse, axes)
    #   would  result in more overloads than that mypy has bugs (i.e. >1_200).
    # NOTE: due to a bug in `axes`, only `int` can be used at the moment (passing a 2-tuple or 2-list raises `TypeError`)
    @overload
    def tensordot[ST: _Scalar](
        self, /, other: _spbase[ST], axes: _Axes = 2
    ) -> _ScalarT_co | ST | coo_array[_ScalarT_co | ST]: ...
    @overload
    def tensordot[ST: _Scalar](
        self, /, other: onp.ArrayND[ST] | onp.SequenceND[onp.ArrayND[ST]] | onp.SequenceND[ST], axes: _Axes = 2
    ) -> _ScalarOrDense[_ScalarT_co | ST]: ...
    @overload
    def tensordot(self, /, other: onp.SequenceND[bool], axes: _Axes = 2) -> _ScalarOrDense[_ScalarT_co]: ...
    @overload
    def tensordot(
        self: _spbase[np.bool | npc.integer8 | npc.integer16 | np.int32 | np.int_], /, other: _JustND[int], axes: _Axes = 2
    ) -> _ScalarOrDense[np.int_]: ...
    @overload
    def tensordot(
        self: _spbase[np.bool | npc.integer | np.float32 | np.float64], /, other: _JustND[float], axes: _Axes = 2
    ) -> _ScalarOrDense[np.float64]: ...
    @overload
    def tensordot(
        self: _spbase[np.bool | npc.integer | npc.inexact32 | npc.inexact64], /, other: _JustND[complex], axes: _Axes = 2
    ) -> _ScalarOrDense[np.complex128]: ...
    @overload
    def tensordot[ST: np.complex128 | npc.complexfloating160](
        self: _spbase[ST], /, other: _JustND[complex], axes: _Axes = 2
    ) -> _ScalarOrDense[ST]: ...
    @overload
    def tensordot[ST: npc.inexact64 | npc.inexact80](
        self: _spbase[ST], /, other: _JustND[float], axes: _Axes = 2
    ) -> _ScalarOrDense[ST]: ...
    @overload
    def tensordot[ST: np.uint32 | npc.number64 | npc.inexact80](
        self: _spbase[ST], /, other: _JustND[int], axes: _Axes = 2
    ) -> _ScalarOrDense[ST]: ...

    #
    @overload
    def __getitem__(self, key: _IndexSlice | tuple[()], /) -> Self: ...
    @overload
    def __getitem__[ST: _Scalar](self: _coo_base[ST, tuple[int]], key: int, /) -> ST: ...
    @overload
    def __getitem__[ST: _Scalar](self: _coo_base[ST, tuple[int]], key: None, /) -> coo_array[ST, tuple[int, int]]: ...
    @overload
    def __getitem__[ST: _Scalar](
        self: _coo_base[ST, tuple[int, int]], key: int | tuple[int, _IndexSlice] | tuple[_IndexSlice, int], /
    ) -> coo_array[ST, tuple[int]]: ...
    @overload
    def __getitem__[ST: _Scalar](self: _coo_base[ST, tuple[int, int]], key: tuple[int, int], /) -> ST: ...
    @overload
    def __getitem__[ST: _Scalar](
        self: _coo_base[ST, tuple[int, int, int, *tuple[int, ...]]],
        key: tuple[int | _IndexSlice | None, *tuple[int | _IndexSlice | None, ...]] | int | None,
        /,
    ) -> coo_array[ST]: ...

    #
    def __setitem__(
        self,
        key: tuple[int | _IndexSlice | None, ...] | int | _IndexSlice | None,
        val: onp.ToComplex | onp.ToComplexND | _spbase,
        /,
    ) -> None: ...

class coo_array(_coo_base[_ScalarT_co, _ShapeT_co], sparray[_ScalarT_co, _ShapeT_co], Generic[_ScalarT_co, _ShapeT_co]):
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
    def __assoc_as_float32__(self, /) -> coo_array[np.float32, _ShapeT_co]: ...
    @override
    @type_check_only
    def __assoc_as_float64__(self, /) -> coo_array[np.float64, _ShapeT_co]: ...
    @override
    @type_check_only
    def __assoc_as_any__(self, /) -> coo_array[Any, _ShapeT_co]: ...

    # NOTE: keep in sync with `coo_array.__init__`
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _spbase[_ScalarT_co, _ShapeT_co] | nptc.CanArray[_ShapeT_co, np.dtype[_ScalarT_co]],
        shape: _ShapeT_co | None = None,
        dtype: onp.ToDType[_ScalarT_co] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToData[_ScalarT_co],
        shape: _ShapeT_co | None = None,
        dtype: onp.ToDType[_ScalarT_co] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # >2-d shape-like, dtype: float64-like | None
    def __init__(
        self: coo_array[np.float64, onp.AtLeast3D],
        /,
        arg1: _ToShapeMin3D,
        shape: _ToShapeMin3D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: bool-like
    def __init__(
        self: coo_array[np.bool, tuple[int]],
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
        self: coo_array[np.bool, tuple[int, int]],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.AnyBoolDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # >2-d shape-like, dtype: bool-like
    def __init__(
        self: coo_array[np.bool, onp.AtLeast3D],
        /,
        arg1: _ToShapeMin3D,
        shape: _ToShapeMin3D | None = None,
        *,
        dtype: onp.AnyBoolDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: int-like
    def __init__(
        self: coo_array[np.int64, tuple[int]],
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
        self: coo_array[np.int64, tuple[int, int]],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.AnyIntDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # >2-d shape-like, dtype: int-like
    def __init__(
        self: coo_array[np.int64, onp.AtLeast3D],
        /,
        arg1: _ToShapeMin3D,
        shape: _ToShapeMin3D | None = None,
        *,
        dtype: onp.AnyIntDType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: float64-like | None
    def __init__(
        self: coo_array[np.float64, tuple[int]],
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
        self: coo_array[np.float64, tuple[int, int]],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: complex128-like
    def __init__(
        self: coo_array[np.complex128, tuple[int]],
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
        self: coo_array[np.complex128, tuple[int, int]],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.AnyComplex128DType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # >2-d shape-like, dtype: complex128-like
    def __init__(
        self: coo_array[np.complex128, onp.AtLeast3D],
        /,
        arg1: _ToShapeMin3D,
        shape: _ToShapeMin3D | None = None,
        *,
        dtype: onp.AnyComplex64DType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: <known>
    def __init__[ST: _Scalar](
        self: coo_array[ST, tuple[int]],
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
        self: coo_array[ST, tuple[int, int]],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # >2-d shape-like, dtype: <known>
    def __init__[ST: _Scalar](
        self: coo_array[ST, onp.AtLeast3D],
        /,
        arg1: _ToShapeMin3D,
        shape: _ToShapeMin3D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # vector-like builtins.bool, dtype: bool-like | None
    def __init__(
        self: coo_array[np.bool, tuple[int]],
        /,
        arg1: Sequence[bool],
        shape: _ToShape1D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: bool-like | None
    def __init__(
        self: coo_array[np.bool, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[bool]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # vector-like builtins.int, dtype: int-like | None
    def __init__(
        self: coo_array[np.int_, tuple[int]],
        /,
        arg1: list[int],
        shape: _ToShape1D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: int-like | None
    def __init__(
        self: coo_array[np.int_, tuple[int, int]],
        /,
        arg1: Sequence[list[int]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # vector-like builtins.float, dtype: float64-like | None
    def __init__(
        self: coo_array[np.float64, tuple[int]],
        /,
        arg1: list[float],
        shape: _ToShape1D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: float64-like | None
    def __init__(
        self: coo_array[np.float64, tuple[int, int]],
        /,
        arg1: Sequence[list[float]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: complex128-like | None
    def __init__(
        self: coo_array[np.complex128, tuple[int]],
        /,
        arg1: list[complex],
        shape: _ToShape1D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: complex128-like | None
    def __init__(
        self: coo_array[np.complex128, tuple[int, int]],
        /,
        arg1: Sequence[list[complex]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (positional)
    def __init__[ST: _Scalar](
        self: coo_array[ST, tuple[int]],
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
        self: coo_array[ST, tuple[int]],
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
        self: coo_array[ST, tuple[int, int]],
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
        self: coo_array[ST, tuple[int, int]],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # >2-D, dtype: <known> (positional)
    def __init__[ST: _Scalar](
        self: coo_array[ST, onp.AtLeast3D],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: _ToShapeMin3D | None,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # >2-D, dtype: <known> (keyword)
    def __init__[ST: _Scalar](
        self: coo_array[ST, onp.AtLeast3D],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: _ToShapeMin3D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: tuple[SupportsIndex, *tuple[SupportsIndex, ...]] | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

class coo_matrix(_coo_base[_ScalarT_co, tuple[int, int]], spmatrix[_ScalarT_co], Generic[_ScalarT_co]):  # type:ignore[misc]
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
    def __assoc_as_float32__(self, /) -> coo_matrix[np.float32]: ...
    @override
    @type_check_only
    def __assoc_as_float64__(self, /) -> coo_matrix[np.float64]: ...
    @override
    @type_check_only
    def __assoc_as_any__(self, /) -> coo_matrix[Any]: ...

    #
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...

    # NOTE: keep in sync with `coo_array.__init__` (minus the 1d overloads)
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self: coo_matrix,
        /,
        arg1: _spbase[_ScalarT_co, _ShapeT_co] | onp.CanArrayND[_ScalarT_co],
        shape: _ShapeT_co | None = None,
        dtype: onp.ToDType[_ScalarT_co] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToData[_ScalarT_co],
        shape: _ShapeT_co | None = None,
        dtype: onp.ToDType[_ScalarT_co] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: coo_matrix[np.float64],
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
        self: coo_matrix[np.bool],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
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
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: bool-like | None
    def __init__(
        self: coo_matrix[np.bool],
        /,
        arg1: Sequence[Sequence[bool]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: int-like | None
    def __init__(
        self: coo_matrix[np.int_],
        /,
        arg1: Sequence[list[int]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: float64-like | None
    def __init__(
        self: coo_matrix[np.float64],
        /,
        arg1: Sequence[list[float]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: complex128-like | None
    def __init__(
        self: coo_matrix[np.complex128],
        /,
        arg1: Sequence[list[complex]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (positional)
    def __init__[ST: _Scalar](
        self: coo_matrix[ST],
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
        self: coo_matrix[ST],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload
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
    def getnnz(self, /, axis: None = None) -> int: ...
    @overload
    def getnnz(self, /, axis: SupportsIndex) -> onp.Array1D[np.int32]: ...

    #
    def __setstate__(self, state: dict[str, Any], /) -> None: ...

    #
    @override
    # pyrefly: ignore [bad-override]
    def __getitem__(self, key: Never, /) -> Never: ...  # type:ignore[override] # pyright:ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]
    @override
    # pyrefly: ignore [bad-override]
    def __setitem__(self, key: Never, x: Never, /) -> None: ...  # type:ignore[override] # pyright:ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]

def isspmatrix_coo(x: object) -> TypeIs[coo_matrix]: ...
