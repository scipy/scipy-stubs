from typing import TypeVar, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy._typing import CanArrayND  # path-dependent Pyright bug workaround

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

@overload
def tukeylambda_variance(lam: onp.ToFloat) -> onp.Array0D[np.float64]: ...
@overload
def tukeylambda_variance(lam: CanArrayND[npc.floating | npc.integer | np.bool_, _ShapeT]) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def tukeylambda_variance(lam: onp.ToFloatND) -> onp.ArrayND[np.float64]: ...

#
@overload
def tukeylambda_kurtosis(lam: onp.ToFloat) -> onp.Array0D[np.float64]: ...
@overload
def tukeylambda_kurtosis(lam: CanArrayND[npc.floating | npc.integer | np.bool_, _ShapeT]) -> onp.Array[_ShapeT, np.float64]: ...
@overload
def tukeylambda_kurtosis(lam: onp.ToFloatND) -> onp.ArrayND[np.float64]: ...
