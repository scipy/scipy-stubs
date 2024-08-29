from scipy._typing import Untyped

from ._base import sparray as sparray
from ._coo import coo_array as coo_array, coo_matrix as coo_matrix

__docformat__: str

def find(A) -> Untyped: ...
def tril(A, k: int = 0, format: Untyped | None = None) -> Untyped: ...
def triu(A, k: int = 0, format: Untyped | None = None) -> Untyped: ...
