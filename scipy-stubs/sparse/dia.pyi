# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

__all__ = [
    "check_shape",
    "dia_matrix",
    "dia_matvec",
    "get_sum_dtype",
    "getdtype",
    "isshape",
    "isspmatrix_dia",
    "spmatrix",
    "upcast_char",
    "validateaxis",
]

@deprecated("will be removed in SciPy v2.0.0")
class spmatrix:
    @property
    def shape(self, /) -> tuple[int, ...]: ...
    def __mul__(self, other: object, /) -> object: ...
    def __rmul__(self, other: object, /) -> object: ...
    def __pow__(self, power: object, /) -> object: ...
    def set_shape(self, /, shape: object) -> None: ...
    def get_shape(self, /) -> tuple[int, ...]: ...
    def asfptype(self, /) -> object: ...
    def getmaxprint(self, /) -> object: ...
    def getformat(self, /) -> object: ...
    def getnnz(self, /, axis: object | None = None) -> object: ...
    def getH(self, /) -> object: ...
    def getcol(self, /, j: int) -> object: ...
    def getrow(self, /, i: int) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
class dia_matrix: ...

@deprecated("will be removed in SciPy v2.0.0")
def dia_matvec(*args: object, **kwargs: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix_dia(x: object) -> object: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def upcast_char(*args: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def getdtype(dtype: object, a: object = ..., default: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isshape(x: object, nonneg: object = ..., *, allow_1d: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_shape(args: object, current_shape: object = ..., *, allow_1d: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def get_sum_dtype(dtype: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def validateaxis(axis: object) -> None: ...
