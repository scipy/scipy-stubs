from typing_extensions import Self

from scipy._typing import Untyped

from ._matrix import spmatrix as spmatrix
from ._sputils import (
    asmatrix as asmatrix,
    check_reshape_kwargs as check_reshape_kwargs,
    check_shape as check_shape,
    get_sum_dtype as get_sum_dtype,
    getdtype as getdtype,
    isdense as isdense,
    isscalarlike as isscalarlike,
    matrix as matrix,
    validateaxis as validateaxis,
)

class SparseWarning(Warning): ...
class SparseFormatWarning(SparseWarning): ...
class SparseEfficiencyWarning(SparseWarning): ...

MAXPRINT: int

class _spbase:
    __array_priority__: float
    @property
    def ndim(self) -> int: ...
    maxprint: Untyped
    def __init__(self, arg1, *, maxprint: Untyped | None = None): ...
    @property
    def shape(self) -> Untyped: ...
    def reshape(self, *args, **kwargs) -> Untyped: ...
    def resize(self, shape): ...
    def astype(self, dtype, casting: str = "unsafe", copy: bool = True) -> Untyped: ...
    def __iter__(self) -> Untyped: ...
    def count_nonzero(self, axis: Untyped | None = None): ...
    @property
    def nnz(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def format(self) -> str: ...
    @property
    def T(self) -> Untyped: ...
    @property
    def real(self) -> Untyped: ...
    @property
    def imag(self) -> Untyped: ...
    def __bool__(self) -> bool: ...
    __nonzero__ = __bool__
    def __len__(self) -> int: ...
    def asformat(self, format, copy: bool = False) -> Untyped: ...
    def multiply(self, other) -> Untyped: ...
    def maximum(self, other) -> Untyped: ...
    def minimum(self, other) -> Untyped: ...
    def dot(self, other) -> Untyped: ...
    def power(self, n, dtype: Untyped | None = None) -> Untyped: ...
    def __eq__(self, other) -> Untyped: ...
    def __ne__(self, other) -> Untyped: ...
    def __lt__(self, other) -> Untyped: ...
    def __gt__(self, other) -> Untyped: ...
    def __le__(self, other) -> Untyped: ...
    def __ge__(self, other) -> Untyped: ...
    def __abs__(self) -> Untyped: ...
    def __round__(self, ndigits: int = 0) -> Untyped: ...
    def __add__(self, other) -> Untyped: ...
    def __radd__(self, other) -> Untyped: ...
    def __sub__(self, other) -> Untyped: ...
    def __rsub__(self, other) -> Untyped: ...
    def __mul__(self, other) -> Untyped: ...
    def __rmul__(self, other) -> Untyped: ...
    def __matmul__(self, other) -> Untyped: ...
    def __rmatmul__(self, other) -> Untyped: ...
    def __truediv__(self, other) -> Untyped: ...
    def __div__(self, other) -> Untyped: ...
    def __rtruediv__(self, other) -> Untyped: ...
    def __rdiv__(self, other) -> Untyped: ...
    def __neg__(self) -> Untyped: ...
    def __iadd__(self, other) -> Self: ...
    def __isub__(self, other) -> Self: ...
    def __imul__(self, other) -> Self: ...
    def __idiv__(self, other) -> Self: ...
    def __itruediv__(self, other) -> Self: ...
    def __pow__(self, *args, **kwargs) -> Untyped: ...
    def transpose(self, axes: Untyped | None = None, copy: bool = False) -> Untyped: ...
    def conjugate(self, copy: bool = True) -> Untyped: ...
    def conj(self, copy: bool = True) -> Untyped: ...
    def nonzero(self) -> Untyped: ...
    def todense(self, order: Untyped | None = None, out: Untyped | None = None) -> Untyped: ...
    def toarray(self, order: Untyped | None = None, out: Untyped | None = None) -> Untyped: ...
    def tocsr(self, copy: bool = False) -> Untyped: ...
    def todok(self, copy: bool = False) -> Untyped: ...
    def tocoo(self, copy: bool = False) -> Untyped: ...
    def tolil(self, copy: bool = False) -> Untyped: ...
    def todia(self, copy: bool = False) -> Untyped: ...
    def tobsr(self, blocksize: Untyped | None = None, copy: bool = False) -> Untyped: ...
    def tocsc(self, copy: bool = False) -> Untyped: ...
    def copy(self) -> Untyped: ...
    def sum(self, axis: Untyped | None = None, dtype: Untyped | None = None, out: Untyped | None = None) -> Untyped: ...
    def mean(self, axis: Untyped | None = None, dtype: Untyped | None = None, out: Untyped | None = None) -> Untyped: ...
    def diagonal(self, k: int = 0) -> Untyped: ...
    def trace(self, offset: int = 0) -> Untyped: ...
    def setdiag(self, values, k: int = 0): ...

class sparray: ...

def issparse(x) -> Untyped: ...
def isspmatrix(x) -> Untyped: ...
