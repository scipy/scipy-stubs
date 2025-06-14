from collections.abc import Sequence
from typing import Any, ClassVar, Generic, Literal, TypeAlias, overload, type_check_only
from typing_extensions import TypeIs, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

from ._base import _spbase, sparray
from ._compressed import _cs_matrix
from ._coo import coo_array, coo_matrix
from ._data import _minmax_mixin
from ._matrix import spmatrix
from ._typing import Numeric, ToShape2D

__all__ = ["bsr_array", "bsr_matrix", "isspmatrix_bsr"]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Numeric, default=Any)
_AsSCT = TypeVar("_AsSCT", bound=Numeric, default=Any)

_ToMatrix: TypeAlias = _spbase[_SCT] | onp.CanArrayND[_SCT] | Sequence[onp.CanArrayND[_SCT]] | _ToMatrixPy[_SCT]
_ToMatrixPy: TypeAlias = Sequence[_T] | Sequence[Sequence[_T]]

_ToData2: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[npc.integer]]
_ToData3: TypeAlias = tuple[onp.ArrayND[_SCT], onp.ArrayND[npc.integer], onp.ArrayND[npc.integer]]
_ToData: TypeAlias = _ToData2[_SCT] | _ToData3[_SCT]

###

class _bsr_base(_cs_matrix[_SCT, tuple[int, int]], _minmax_mixin[_SCT, tuple[int, int]], Generic[_SCT]):
    _format: ClassVar = "bsr"

    data: onp.Array3D[_SCT]

    @property
    @override
    def format(self, /) -> Literal["bsr"]: ...
    @property
    @override
    def ndim(self, /) -> Literal[2]: ...
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...
    @property
    def blocksize(self, /) -> tuple[int, int]: ...

class bsr_array(_bsr_base[_SCT], sparray[_SCT, tuple[int, int]], Generic[_SCT]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> coo_array[_SCT, tuple[int, int]]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _AsSCT, /) -> coo_array[_AsSCT, tuple[int, int]]: ...

    # NOTE: keep in sync with `bsr_matrix.__init__`
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT] | _ToData[_SCT],
        shape: ToShape2D | None = None,
        dtype: None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: bsr_array[np.float64],
        /,
        arg1: ToShape2D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: bsr_array[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: bsr_array[np.int_],
        /,
        arg1: _ToMatrixPy[op.JustInt],
        shape: ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: bsr_array[np.float64],
        /,
        arg1: _ToMatrixPy[op.JustFloat],
        shape: ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: bsr_array[np.complex128],
        /,
        arg1: _ToMatrixPy[op.JustComplex],
        shape: ToShape2D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (positional)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShape2D | None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (keyword)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShape2D | None = None,
        *,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        maxprint: int | None = None,
    ) -> None: ...

class bsr_matrix(_bsr_base[_SCT], spmatrix[_SCT], Generic[_SCT]):
    # NOTE: These two methods do not exist at runtime.
    # See the relevant comment in `sparse._base._spbase` for more information.
    @override
    @type_check_only
    def __assoc_stacked__(self, /) -> coo_matrix[_SCT]: ...
    @override
    @type_check_only
    def __assoc_stacked_as__(self, sctype: _AsSCT, /) -> coo_matrix[_AsSCT]: ...

    # NOTE: keep in sync with `bsr_array.__init__`
    @overload  # matrix-like (known dtype), dtype: None
    def __init__(
        self,
        /,
        arg1: _ToMatrix[_SCT] | _ToData[_SCT],
        shape: ToShape2D | None = None,
        dtype: None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # 2-d shape-like, dtype: None
    def __init__(
        self: bsr_matrix[np.float64],
        /,
        arg1: ToShape2D,
        shape: None = None,
        dtype: None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.bool, dtype: type[bool] | None
    def __init__(
        self: bsr_matrix[np.bool_],
        /,
        arg1: _ToMatrixPy[bool],
        shape: ToShape2D | None = None,
        dtype: onp.AnyBoolDType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.int, dtype: type[int] | None
    def __init__(
        self: bsr_matrix[np.int_],
        /,
        arg1: _ToMatrixPy[op.JustInt],
        shape: ToShape2D | None = None,
        dtype: onp.AnyIntDType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.float, dtype: type[float] | None
    def __init__(
        self: bsr_matrix[np.float64],
        /,
        arg1: _ToMatrixPy[op.JustFloat],
        shape: ToShape2D | None = None,
        dtype: onp.AnyFloat64DType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # matrix-like builtins.complex, dtype: type[complex] | None
    def __init__(
        self: bsr_matrix[np.complex128],
        /,
        arg1: _ToMatrixPy[op.JustComplex],
        shape: ToShape2D | None = None,
        dtype: onp.AnyComplex128DType | None = None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (positional)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShape2D | None,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # dtype: <known> (keyword)
    def __init__(
        self,
        /,
        arg1: onp.ToComplexND,
        shape: ToShape2D | None = None,
        *,
        copy: bool = False,
        blocksize: tuple[int, int] | None = None,
        maxprint: int | None = None,
    ) -> None: ...

#
def isspmatrix_bsr(x: object) -> TypeIs[bsr_matrix]: ...
