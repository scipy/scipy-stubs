from typing import TypeAlias

import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["bandwidth", "ishermitian", "issymmetric"]

# simplified version of the `ctypedef fused np_numeric_t` in `scipy/linalg/_cythonized_array_utils.pxd`
_Numeric: TypeAlias = npc.integer | npc.inexact32 | npc.inexact64 | npc.floating80

def bandwidth(a: onp.ArrayND[_Numeric]) -> tuple[int, int]: ...
def issymmetric(a: onp.ArrayND[_Numeric], atol: float | None = None, rtol: float | None = None) -> bool: ...
def ishermitian(a: onp.ArrayND[_Numeric], atol: float | None = None, rtol: float | None = None) -> bool: ...
