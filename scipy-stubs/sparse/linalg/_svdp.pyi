from scipy._typing import Untyped

from scipy._lib._util import check_random_state as check_random_state
from scipy.linalg import LinAlgError as LinAlgError
from scipy.sparse.linalg import aslinearoperator as aslinearoperator

class _AProd:
    A: Untyped
    def __init__(self, A) -> None: ...
    def __call__(self, transa, m, n, x, y, sparm, iparm): ...
    @property
    def shape(self) -> Untyped: ...
    @property
    def dtype(self) -> Untyped: ...
