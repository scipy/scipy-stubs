from collections.abc import Iterable, Sequence
from typing import Literal as L, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy._typing import SequenceNotStr, _FortranFunction

__all__ = ["find_best_blas_type", "get_blas_funcs"]

# see `scipy.linalg.blas._type_conv`
def find_best_blas_type(
    arrays: Sequence[onp.ArrayND] = (), dtype: npt.DTypeLike | None = None
) -> (
    tuple[L["s"], np.dtype[np.float32], bool]
    | tuple[L["f"], np.dtype[np.float64], bool]
    | tuple[L["c"], np.dtype[np.complex64], bool]
    | tuple[L["z"], np.dtype[np.complex128], bool]
): ...

#
@overload
def get_blas_funcs(
    names: str, arrays: Sequence[onp.ArrayND] = (), dtype: npt.DTypeLike | None = None, ilp64: L["preferred"] | bool = False
) -> _FortranFunction: ...
@overload
def get_blas_funcs(
    names: SequenceNotStr[str],
    arrays: Sequence[onp.ArrayND] = (),
    dtype: npt.DTypeLike | None = None,
    ilp64: L["preferred"] | bool = False,
) -> list[_FortranFunction]: ...
@overload
def get_blas_funcs(
    names: Iterable[str],
    arrays: Sequence[onp.ArrayND] = (),
    dtype: npt.DTypeLike | None = None,
    ilp64: L["preferred"] | bool = False,
) -> list[_FortranFunction] | _FortranFunction: ...
