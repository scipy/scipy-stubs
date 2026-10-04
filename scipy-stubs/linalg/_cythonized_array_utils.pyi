from typing import Any, Never, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["ishermitian", "issymmetric"]

###

# see `scipy/linalg/_cythonized_array_utils.pxd`
type _Numeric = npc.integer | np.float32 | np.float64 | npc.floating80 | np.complex64 | np.complex128

# workaround for mypy & pyright's failure to conform to the overload typing specification
type _JustAnyShape = tuple[Never, Never]

###

@overload  # ?d  (workaround)
def issymmetric(a: onp.ArrayND[_Numeric, _JustAnyShape], atol: float | None = None, rtol: float | None = None) -> bool | Any: ...
@overload  # 2d
def issymmetric(a: onp.Array2D[_Numeric], atol: float | None = None, rtol: float | None = None) -> bool: ...
@overload  # >=3d
def issymmetric[ShapeT: onp.AtLeast3D](
    a: onp.ArrayND[_Numeric, ShapeT], atol: float | None = None, rtol: float | None = None
) -> onp.ArrayND[np.bool]: ...
@overload  # fallback
def issymmetric(a: onp.CanArrayND[_Numeric], atol: float | None = None, rtol: float | None = None) -> bool | Any: ...

#
@overload  # 2d
def ishermitian(a: onp.Array2D[_Numeric], atol: float | None = None, rtol: float | None = None) -> bool: ...
@overload  # >=3d
def ishermitian[ShapeT: onp.AtLeast3D](
    a: onp.ArrayND[_Numeric, ShapeT], atol: float | None = None, rtol: float | None = None
) -> onp.ArrayND[np.bool]: ...
@overload  # fallback
def ishermitian(a: onp.CanArrayND[_Numeric], atol: float | None = None, rtol: float | None = None) -> bool | Any: ...
