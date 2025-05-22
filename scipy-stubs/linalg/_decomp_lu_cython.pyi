from typing import TypeAlias, TypeVar

import numpy as np
import optype.numpy as onp

_AnyInexactT = TypeVar("_AnyInexactT", np.float32, np.float64, np.complex64, np.complex128)
_Int1D: TypeAlias = onp.ArrayND[np.int32 | np.int64]

###

# the `lu_decompose` function in scipy's bundled stub file does not exist
def lu_dispatcher(a: onp.Array2D[_AnyInexactT], u: onp.Array2D[_AnyInexactT], piv: _Int1D, permute_l: onp.ToBool) -> None: ...
