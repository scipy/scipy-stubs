# https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_group_columns.py

from typing import Final
from typing_extensions import LiteralString

import numpy as np
import optype.numpy as onp

__pythran__: Final[tuple[LiteralString, LiteralString]]

# (int | bool, int | bool, int | bool[:, :]) -> int | bool[:]
def group_dense(m: onp.ToJustInt, n: onp.ToJustInt, A: onp.ToJustInt2D) -> onp.Array1D[np.int32]: ...

# (int | bool, int | bool, int | bool[:], int | bool[:]) -> int | bool[:]
def group_sparse(
    m: onp.ToJustInt,
    n: onp.ToJustInt,
    indices: onp.ToJustInt1D,
    indptr: onp.ToJustInt1D,
) -> onp.Array1D[np.int32]: ...
