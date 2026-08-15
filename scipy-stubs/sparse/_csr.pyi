from typing import Any, ClassVar, Generic, Literal, Never, SupportsIndex, overload, override, type_check_only
from typing_extensions import TypeIs, TypeVar

import numpy as np
import numpy.typing as npt
import numpy_typing_compat as nptc
import optype.numpy as onp
import optype.numpy.compat as npc

from ._base import _spbase, sparray
from ._compressed import _cs_matrix
from ._csc import csc_array, csc_matrix
from ._matrix import spmatrix
from ._typing import _ToShape1D, _ToShape2D

__all__ = ["csr_array", "csr_matrix", "isspmatrix_csr"]

###

# workaround for the typing-spec non-conformance regarding overload behavior of mypy and pyright
type _NeitherD = tuple[Never] | tuple[Never, Never]

type _Scalar = npc.number | np.bool

type _ToMatrixPy[T] = list[T] | list[list[T]]
type _ToMatrix[ST: _Scalar] = _spbase[ST] | onp.CanArrayND[ST] | list[onp.ArrayND[ST]] | _ToMatrixPy[ST]

type _ToData[T] = (
    tuple[T, tuple[onp.ToJustInt1D, onp.ToJustInt1D]]  # (data, (row_ind, col_ind))
    | tuple[T, onp.ToJustInt1D, onp.ToJustInt1D]  # (data, indices, indptr)
)

_ScalarT_co = TypeVar("_ScalarT_co", bound=_Scalar, default=Any, covariant=True)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int, int], covariant=True)

###

class _csr_base(_cs_matrix[_ScalarT_co, _ShapeT_co], Generic[_ScalarT_co, _ShapeT_co]):
    _format: ClassVar = "csr"
    _allow_nd: ClassVar = 1, 2

    @property
    @override
    def ndim(self, /) -> Literal[1, 2]: ...
    @property
    @override
    def format(self, /) -> Literal["csr"]: ...

    #
    @override
    @overload
    def count_nonzero(self, /, axis: None = None) -> np.intp: ...
    @overload
    def count_nonzero(self: _csr_base[Any, _NeitherD], /, axis: SupportsIndex) -> onp.Array1D[np.intp] | Any: ...
    @overload
    def count_nonzero(self: csr_array[Any, tuple[int]], /, axis: SupportsIndex) -> np.intp: ...  # type: ignore[misc]
    @overload
    def count_nonzero(self: _csr_base[Any, tuple[int, int]], /, axis: SupportsIndex) -> onp.Array1D[np.intp]: ...
    @overload
    def count_nonzero(self: csr_array[Any, Any], /, axis: SupportsIndex) -> onp.Array1D[np.intp] | Any: ...  # type: ignore[misc]

class csr_array(_csr_base[_ScalarT_co, _ShapeT_co], sparray[_ScalarT_co, _ShapeT_co], Generic[_ScalarT_co, _ShapeT_co]):
    # NOTE: These four methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> csr_array[_ScalarT_co, tuple[int, int]]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__[ST: _Scalar](self, sctype: ST, /) -> csr_array[ST, tuple[int, int]]: ...
    @override
    @type_check_only
    def __assoc_as_float32__(self, /) -> csr_array[np.float32, _ShapeT_co]: ...
    @override
    @type_check_only
    def __assoc_as_float64__(self, /) -> csr_array[np.float64, _ShapeT_co]: ...
    @override
    @type_check_only
    def __assoc_as_any__(self, /) -> csr_array[Any, _ShapeT_co]: ...

    #
    @overload  # sparse or dense (know dtype & shape)
    def __init__(
        self,
        /,
        arg1: _spbase[_ScalarT_co, _ShapeT_co] | nptc.CanArray[_ShapeT_co, np.dtype[_ScalarT_co]],
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like (know dtype)
    def __init__[ST: _Scalar](
        self: csr_array[ST, tuple[int]],
        /,
        arg1: list[ST],
        shape: _ToShape1D | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like (know dtype)
    def __init__[ST: _Scalar](
        self: csr_array[ST, tuple[int, int]],
        /,
        arg1: onp.ToArray2D[ST, ST],
        shape: _ToShape2D | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like (know dtype)
    def __init__[ST: _Scalar](
        self: csr_array[ST, tuple[int, int]],
        /,
        arg1: _ToData[onp.ToArray1D[ST, ST]],
        shape: _ToShape2D | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like (known dtype)
    def __init__[ST: _Scalar](
        self: csr_array[ST, tuple[Any, ...]],
        /,
        arg1: _ToMatrix[ST],
        shape: _ToShape1D | _ToShape2D | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: float64-like | None
    def __init__(
        self: csr_array[np.float64, tuple[int]],
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
        self: csr_array[np.float64, tuple[int, int]],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: bool-like (keyword)
    def __init__(
        self: csr_array[np.bool, tuple[int]],
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
        self: csr_array[np.bool, tuple[int, int]],
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
        self: csr_array[np.int64, tuple[int]],
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
        self: csr_array[np.int64, tuple[int, int]],
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
        self: csr_array[np.complex128, tuple[int]],
        /,
        arg1: _ToShape1D,
        shape: _ToShape1D | None = None,
        *,
        dtype: onp.AnyComplex128DType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: <known>
    def __init__[ST: _Scalar](
        self: csr_array[ST, tuple[int]],
        /,
        arg1: _ToShape1D,
        shape: _ToShape1D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: complex128-like
    def __init__(
        self: csr_array[np.complex128, tuple[int, int]],
        /,
        arg1: _ToShape2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.AnyComplex128DType,
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: <known>
    def __init__[ST: _Scalar](
        self: csr_array[ST, tuple[int, int]],
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
        self: csr_array[np.bool, tuple[int]],
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
        self: csr_array[np.bool, tuple[int, int]],
        /,
        arg1: onp.ToJustBoolStrict2D | _ToData[list[bool]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like ~int, dtype: int-like | None
    def __init__(
        self: csr_array[np.int64, tuple[int]],
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
        self: csr_array[np.int64, tuple[int, int]],
        /,
        arg1: onp.ToJustInt64Strict2D | _ToData[list[int]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like ~float, dtype: float64-like | None
    def __init__(
        self: csr_array[np.float64, tuple[int]],
        /,
        arg1: onp.ToJustFloat64Strict1D,
        shape: _ToShape1D | _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~float, dtype: float64-like | None
    def __init__(
        self: csr_array[np.float64, tuple[int, int]],
        /,
        arg1: onp.ToJustFloat64Strict2D | _ToData[list[float]],
        shape: _ToShape1D | _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like ~complex, dtype: complex128-like | None
    def __init__(
        self: csr_array[np.complex128, tuple[int]],
        /,
        arg1: onp.ToJustComplex128Strict1D,
        shape: _ToShape1D | _ToShape2D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~complex, dtype: complex128-like | None
    def __init__(
        self: csr_array[np.complex128, tuple[int, int]],
        /,
        arg1: onp.ToJustComplex128Strict2D | _ToData[list[complex]],
        shape: _ToShape1D | _ToShape2D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-D, dtype: <known>
    def __init__[ST: _Scalar](
        self: csr_array[ST, tuple[int]],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: _ToShape1D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known>
    def __init__[ST: _Scalar](
        self: csr_array[ST, tuple[int, int]],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: _ToShape2D | None = None,
        *,
        dtype: onp.ToDType[ST],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # shape: known
    def __init__(
        self,
        /,
        arg1: onp.ToComplex1D | onp.ToComplex2D,
        shape: _ToShape1D | _ToShape2D | None = None,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

    #
    @override  # type: ignore[override]
    @overload
    def transpose[ST: _Scalar](  # pyrefly: ignore[bad-override]
        self: csr_array[ST, tuple[int, int]], /, axes: tuple[Literal[1, -1], Literal[0]] | None = None, copy: bool = False
    ) -> csc_array[ST]: ...
    @overload
    def transpose[ST: _Scalar](  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]
        self: csr_array[ST, tuple[int]], /, axes: None = None, copy: bool = False
    ) -> csr_array[ST, tuple[int]]: ...

class csr_matrix(_csr_base[_ScalarT_co], spmatrix[_ScalarT_co], Generic[_ScalarT_co]):  # type: ignore[misc]
    # NOTE: These four methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> csr_matrix[_ScalarT_co]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__[ST: _Scalar](self, sctype: ST, /) -> csr_matrix[ST]: ...
    @override
    @type_check_only
    def __assoc_as_float32__(self, /) -> csr_matrix[np.float32]: ...
    @override
    @type_check_only
    def __assoc_as_float64__(self, /) -> csr_matrix[np.float64]: ...
    @override
    @type_check_only
    def __assoc_as_any__(self, /) -> csr_matrix[Any]: ...

    # NOTE: keep in sync with `csc_matrix.__init__`
    @overload  # matrix-like (known dtype)
    def __init__[ST: _Scalar](
        self: csr_matrix[ST],  # this self annotation works around a mypy bug
        /,
        arg1: _ToMatrix[ST],
        shape: _ToShape2D | None = None,
        dtype: onp.ToDType[ST] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like (known dtype)
    def __init__[ST: _Scalar](
        self: csr_matrix[ST],  # this self annotation works around a mypy bug
        /,
        arg1: _ToData[onp.ToArray1D[ST, ST]],
        shape: _ToShape2D | None = None,
        dtype: onp.ToDType[ST] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like
    def __init__(
        self: csr_matrix[np.float64],
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
        self: csr_matrix[np.bool],
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
    @overload  # 2-d array-like bool, dtype: bool-like | None
    def __init__(
        self: csr_matrix[np.bool],
        /,
        arg1: onp.ToJustBoolStrict2D | _ToData[list[bool]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~int, dtype: int-like | None
    def __init__(
        self: csr_matrix[np.int64],
        /,
        arg1: onp.ToJustInt64Strict2D | _ToData[list[int]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~float, dtype: float64-like | None
    def __init__(
        self: csr_matrix[np.float64],
        /,
        arg1: onp.ToJustFloat64Strict2D | _ToData[list[float]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~complex, dtype: complex128-like | None
    def __init__(
        self: csr_matrix[np.complex128],
        /,
        arg1: onp.ToJustComplex128Strict2D | _ToData[list[complex]],
        shape: _ToShape2D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (positional)
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
    @overload  # 2-D, dtype: <known> (keyword)
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
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...

    #
    @override
    def transpose(  # type: ignore[override] # pyrefly: ignore [bad-override]
        self, /, axes: tuple[Literal[1, -1], Literal[0]] | None = None, copy: bool = False
    ) -> csc_matrix[_ScalarT_co]: ...

    #
    @override
    @overload
    def getnnz(self, /, axis: None = None) -> int: ...
    @overload
    def getnnz(self, /, axis: Literal[0, 1, -1, -2]) -> onp.Array1D[np.intp]: ...

def isspmatrix_csr(x: object) -> TypeIs[csr_matrix]: ...
