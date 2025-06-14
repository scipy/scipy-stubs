from collections.abc import Sequence
from typing import Any, ClassVar, Generic, Literal, Never, TypeAlias, overload, type_check_only
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from ._base import _spbase, sparray
from ._compressed import _cs_matrix
from ._matrix import spmatrix
from ._typing import Index1D, Numeric, ToShape1D, ToShape2D, ToShapeMin1D

__all__ = ["csr_array", "csr_matrix", "isspmatrix_csr"]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Numeric, default=Any)
_SCT0 = TypeVar("_SCT0", bound=Numeric)
_AsSCT = TypeVar("_AsSCT", bound=Numeric)
_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int] | tuple[int, int], default=tuple[int, int], covariant=True)

# workaround for the typing-spec non-conformance regarding overload behavior of mypy and pyright
_NeitherD: TypeAlias = tuple[Never] | tuple[Never, Never]

_ToMatrix: TypeAlias = _spbase[_SCT] | onp.CanArrayND[_SCT] | Sequence[onp.CanArrayND[_SCT]] | _ToMatrixPy[_SCT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]

_ToData2B: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[npc.integer]]  # bsr
_ToData2C: TypeAlias = tuple[onp.ArrayND[_SCT], tuple[onp.ArrayND[npc.integer], onp.ArrayND[npc.integer]]]  # csc, csr
_ToData2: TypeAlias = _ToData2B[_SCT] | _ToData2C[_SCT]
_ToData3: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[npc.integer], onp.ArrayND[npc.integer]]
_ToData: TypeAlias = _ToData2[_SCT] | _ToData3[_SCT]

###

class _csr_base(_cs_matrix[_SCT, _ShapeT_co], Generic[_SCT, _ShapeT_co]):
    _format: ClassVar = "csr"
    _allow_nd: ClassVar = 1, 2

    @property
    @override
    def ndim(self, /) -> Literal[1, 2]: ...
    @property
    @override
    def format(self, /) -> Literal["csr"]: ...

    #
    @overload
    def count_nonzero(self, /, axis: None = None) -> np.intp: ...
    @overload
    def count_nonzero(self: _csr_base[Any, _NeitherD], /, axis: op.CanIndex) -> onp.Array1D[np.intp] | Any: ...  # noqa: ANN401
    @overload
    def count_nonzero(self: csr_array[Any, tuple[int]], /, axis: op.CanIndex) -> np.intp: ...  # type: ignore[misc]
    @overload
    def count_nonzero(self: _csr_base[Any], /, axis: op.CanIndex) -> onp.Array1D[np.intp]: ...
    @overload
    def count_nonzero(self: csr_array[Any, Any], /, axis: op.CanIndex) -> onp.Array1D[np.intp] | Any: ...  # type: ignore[misc]  # noqa: ANN401

class csr_array(_csr_base[_SCT, _ShapeT_co], sparray[_SCT, _ShapeT_co], Generic[_SCT, _ShapeT_co]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> csr_array[_SCT, tuple[int, int]]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _AsSCT, /) -> csr_array[_AsSCT, tuple[int, int]]: ...

    #
    @overload  # sparse or dense (know dtype & shape), dtype: None
    def __init__(
        self,
        /,
        arg1: _spbase[_SCT, _ShapeT_co] | onp.CanArrayND[_SCT, _ShapeT_co],
        shape: _ShapeT_co | None = None,
        dtype: onp.ToDType[_SCT] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like (know dtype), dtype: None
    def __init__(
        self: csr_array[_SCT0, tuple[int]],
        /,
        arg1: Sequence[_SCT0],
        shape: ToShape1D | None = None,
        dtype: onp.ToDType[_SCT0] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like (know dtype), dtype: None
    def __init__(
        self: csr_array[_SCT0, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[_SCT0]] | Sequence[onp.CanArrayND[_SCT0]],  # assumes max. 2-d
        shape: ToShape2D | None = None,
        dtype: onp.ToDType[_SCT0] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT] | _ToData[_SCT],
        shape: ToShapeMin1D | None = None,
        dtype: onp.ToDType[_SCT] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d shape-like, dtype: None
    def __init__(
        self: csr_array[np.float64, tuple[int]],
        /,
        arg1: ToShape1D,
        shape: ToShape1D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: csr_array[np.float64, tuple[int, int]],
        /,
        arg1: ToShape2D,
        shape: ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like bool, dtype: type[bool] | None
    def __init__(
        self: csr_array[np.bool_, tuple[int]],
        /,
        arg1: Sequence[bool],
        shape: ToShape1D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like bool, dtype: type[bool] | None
    def __init__(
        self: csr_array[np.bool_, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[bool]],
        shape: ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like ~int, dtype: type[int] | None
    def __init__(
        self: csr_array[np.int_, tuple[int]],
        /,
        arg1: Sequence[op.JustInt],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~int, dtype: type[int] | None
    def __init__(
        self: csr_array[np.int_, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[op.JustInt]],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like ~float, dtype: type[float] | None
    def __init__(
        self: csr_array[np.float64, tuple[int]],
        /,
        arg1: Sequence[op.JustFloat],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~float, dtype: type[float] | None
    def __init__(
        self: csr_array[np.float64, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[op.JustFloat]],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-d array-like ~complex, dtype: type[complex] | None
    def __init__(
        self: csr_array[np.complex128, tuple[int]],
        /,
        arg1: Sequence[op.JustComplex],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~complex, dtype: type[complex] | None
    def __init__(
        self: csr_array[np.complex128, tuple[int, int]],
        /,
        arg1: Sequence[Sequence[op.JustComplex]],
        shape: ToShapeMin1D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (positional)
    def __init__(
        self: csr_array[_SCT0, tuple[int]],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: ToShape1D | None,
        dtype: onp.ToDType[_SCT0],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 1-D, dtype: <known> (keyword)
    def __init__(
        self: csr_array[_SCT0, tuple[int]],
        /,
        arg1: onp.ToComplexStrict1D,
        shape: ToShape1D | None = None,
        *,
        dtype: onp.ToDType[_SCT0],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (positional)
    def __init__(
        self: csr_array[_SCT0, tuple[int, int]],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShape2D | None,
        dtype: onp.ToDType[_SCT0],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (keyword)
    def __init__(
        self: csr_array[_SCT0, tuple[int, int]],
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShape2D | None = None,
        *,
        dtype: onp.ToDType[_SCT0],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # shape: known
    def __init__(
        self,
        /,
        arg1: onp.ToComplex1D | onp.ToComplex2D,
        shape: ToShapeMin1D | None = None,
        dtype: npt.DTypeLike | None = ...,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...

class csr_matrix(_csr_base[_SCT], spmatrix[_SCT], Generic[_SCT]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> csr_matrix[_SCT]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _AsSCT, /) -> csr_matrix[_AsSCT]: ...

    # NOTE: keep in sync with `csc_matrix.__init__`
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT],
        shape: ToShape2D | None = None,
        dtype: onp.ToDType[_SCT] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: csr_matrix[np.float64],
        /,
        arg1: ToShape2D,
        shape: ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like bool, dtype: type[bool] | None
    def __init__(
        self: csr_matrix[np.bool_],
        /,
        arg1: Sequence[Sequence[bool]],
        shape: ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~int, dtype: type[int] | None
    def __init__(
        self: csr_matrix[np.int_],
        /,
        arg1: Sequence[Sequence[op.JustInt]],
        shape: ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~float, dtype: type[float] | None
    def __init__(
        self: csr_matrix[np.float64],
        /,
        arg1: Sequence[Sequence[op.JustFloat]],
        shape: ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d array-like ~complex, dtype: type[complex] | None
    def __init__(
        self: csr_matrix[np.complex128],
        /,
        arg1: Sequence[Sequence[op.JustComplex]],
        shape: ToShape2D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (positional)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShape2D | None,
        dtype: onp.ToDType[_SCT],
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-D, dtype: <known> (keyword)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexStrict2D,
        shape: ToShape2D | None = None,
        *,
        dtype: onp.ToDType[_SCT],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...

    #
    @overload
    def getnnz(self, /, axis: None = None) -> int: ...
    @overload
    def getnnz(self, /, axis: op.CanIndex) -> Index1D: ...

def isspmatrix_csr(x: object) -> TypeIs[csr_matrix]: ...
