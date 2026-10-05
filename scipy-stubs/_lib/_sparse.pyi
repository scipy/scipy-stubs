from abc import ABC
from typing import Any
from typing_extensions import TypeIs

from scipy.sparse import (
    bsr_array,
    bsr_matrix,
    coo_array,
    coo_matrix,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
    dia_array,
    dia_matrix,
    dok_array,
    dok_matrix,
    lil_array,
    lil_matrix,
)

__all__ = ["SparseABC", "issparse"]

class SparseABC(ABC): ...

# NOTE: `[Any, Any]` instead of `[Any, tuple[Any, ...]]`, because otherwise mypy won't narrow e.g.
# `csr_array[float64, tuple[int, int]]` out of the negative branch.
def issparse(
    x: object,
) -> TypeIs[
    bsr_matrix[Any]
    | coo_matrix[Any]
    | csc_matrix[Any]
    | csr_matrix[Any]
    | dia_matrix[Any]
    | dok_matrix[Any]
    | lil_matrix[Any]
    | bsr_array
    | coo_array[Any, Any]
    | csc_array
    | csr_array[Any, Any]
    | dia_array
    | dok_array[Any, Any]
    | lil_array
]: ...
