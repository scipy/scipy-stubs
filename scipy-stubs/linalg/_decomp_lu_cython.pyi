# NOTE: keep in sync with scipy/stubs/linalg/_decomp_lu_cython.pyi

from typing import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

# this mimicks the `ctypedef fused lapack_t`
_LapackT = TypeVar("_LapackT", np.float32, np.float64, np.complex64, np.complex128)

def lu_dispatcher(a: onp.ArrayND[_LapackT], u: onp.ArrayND[_LapackT], piv: onp.ArrayND[npc.integer], permute_l: bool) -> None: ...
