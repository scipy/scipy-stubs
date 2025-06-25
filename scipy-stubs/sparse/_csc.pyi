from collections.abc import Sequence
from typing import Any, ClassVar, Generic, Literal, TypeAlias, overload, type_check_only
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp

from ._base import _spbase, sparray
from ._compressed import _cs_matrix
from ._csr import _csr_base, csr_array, csr_matrix
from ._matrix import spmatrix
from ._typing import Index1D, Numeric, ToShape2D

__all__ = ["csc_array", "csc_matrix", "isspmatrix_csc"]

_T = TypeVar("_T")
_ScalarT = TypeVar("_ScalarT", bound=Numeric)
_ScalarT_co = TypeVar("_ScalarT_co", bound=Numeric, default=Any, covariant=True)

_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]
_ToMatrix: TypeAlias = _spbase[_ScalarT] | onp.CanArrayND[_ScalarT] | Sequence[onp.CanArrayND[_ScalarT]] | _ToMatrixPy[_ScalarT]

###

class _csc_base(_cs_matrix[_ScalarT_co, tuple[int, int]], Generic[_ScalarT_co]):
    _format: ClassVar = "csc"

    @property
    @override
    def format(self, /) -> Literal["csc"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...

    #
    @override
    def transpose(  # type: ignore[override]
        self, /, axes: tuple[Literal[1, -1], Literal[0]] | None = None, copy: bool = False
    ) -> _csr_base[_ScalarT_co, tuple[int, int]]: ...

    #
    @override
    @overload
    def count_nonzero(self, /, axis: None = None) -> np.intp: ...
    @overload
    def count_nonzero(self, /, axis: op.CanIndex) -> onp.Array1D[np.intp]: ...

class csc_array(_csc_base[_ScalarT_co], sparray[_ScalarT_co, tuple[int, int]], Generic[_ScalarT_co]):
    # NOTE: These four methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> csc_array[_ScalarT_co]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _ScalarT, /) -> csc_array[_ScalarT]: ...
    @override
    @type_check_only
    def __assoc_as_float32__(self, /) -> csc_array[np.float32]: ...
    @override
    @type_check_only
    def __assoc_as_float64__(self, /) -> csc_array[np.float64]: ...

    # NOTE: keep in sync with `csc_matrix.__init__`
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_ScalarT_co],
        shape: ToShape2D | None = None,
        dtype: onp.ToDType[_ScalarT_co] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: csc_array[np.float64],
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
        self: csc_array[np.bool_],
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
        self: csc_array[np.int_],
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
        self: csc_array[np.float64],
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
        self: csc_array[np.complex128],
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
        dtype: onp.ToDType[_ScalarT_co],
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
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...

    #
    @override
    def transpose(  # type: ignore[override]
        self, /, axes: tuple[Literal[1, -1], Literal[0]] | None = None, copy: bool = False
    ) -> csr_array[_ScalarT_co, tuple[int, int]]: ...

class csc_matrix(_csc_base[_ScalarT_co], spmatrix[_ScalarT_co], Generic[_ScalarT_co]):
    # NOTE: These four methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> csc_matrix[_ScalarT_co]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _ScalarT, /) -> csc_matrix[_ScalarT]: ...
    @override
    @type_check_only
    def __assoc_as_float32__(self, /) -> csc_matrix[np.float32]: ...
    @override
    @type_check_only
    def __assoc_as_float64__(self, /) -> csc_matrix[np.float64]: ...

    # NOTE: keep in sync with `csc_array.__init__`
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_ScalarT_co],
        shape: ToShape2D | None = None,
        dtype: onp.ToDType[_ScalarT_co] | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: csc_matrix[np.float64],
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
        self: csc_matrix[np.bool_],
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
        self: csc_matrix[np.int_],
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
        self: csc_matrix[np.float64],
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
        self: csc_matrix[np.complex128],
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
        dtype: onp.ToDType[_ScalarT_co],
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
        dtype: onp.ToDType[_ScalarT_co],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...

    #
    @override
    def transpose(  # type: ignore[override]
        self, /, axes: tuple[Literal[1, -1], Literal[0]] | None = None, copy: bool = False
    ) -> csr_matrix[_ScalarT_co]: ...

    #
    @override
    @overload
    def getnnz(self, /, axis: None = None) -> int: ...
    @overload
    def getnnz(self, /, axis: op.CanIndex) -> Index1D: ...

def isspmatrix_csc(x: object) -> TypeIs[csc_matrix]: ...
