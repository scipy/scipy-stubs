from scipy._typing import Untyped
from scipy.sparse.linalg import LinearOperator

__all__ = ["LaplacianNd"]

class LaplacianNd(LinearOperator):
    grid_shape: Untyped
    boundary_conditions: Untyped
    def __init__(self, /, grid_shape: Untyped, *, boundary_conditions: str = "neumann", dtype: Untyped = ...) -> None: ...
    def eigenvalues(self, /, m: Untyped | None = None) -> Untyped: ...
    def eigenvectors(self, /, m: Untyped | None = None) -> Untyped: ...
    def toarray(self, /) -> Untyped: ...
    def tosparse(self, /) -> Untyped: ...

class Sakurai(LinearOperator):
    n: Untyped
    def __init__(self, /, n: Untyped, dtype: Untyped = ...) -> None: ...
    def eigenvalues(self, /, m: Untyped | None = None) -> Untyped: ...
    def tobanded(self, /) -> Untyped: ...
    def tosparse(self, /) -> Untyped: ...
    def toarray(self, /) -> Untyped: ...

class MikotaM(LinearOperator):
    def __init__(self, /, shape: Untyped, dtype: Untyped = ...) -> None: ...
    def tobanded(self, /) -> Untyped: ...
    def tosparse(self, /) -> Untyped: ...
    def toarray(self, /) -> Untyped: ...

class MikotaK(LinearOperator):
    def __init__(self, /, shape: Untyped, dtype: Untyped = ...) -> None: ...
    def tobanded(self, /) -> Untyped: ...
    def tosparse(self, /) -> Untyped: ...
    def toarray(self, /) -> Untyped: ...

class MikotaPair:
    n: Untyped
    dtype: Untyped
    shape: Untyped
    m: Untyped
    k: Untyped
    def __init__(self, /, n: Untyped, dtype: Untyped = ...) -> None: ...
    def eigenvalues(self, /, m: Untyped | None = None) -> Untyped: ...
