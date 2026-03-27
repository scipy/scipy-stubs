from typing import Literal, SupportsIndex, TypeVar, overload

import numpy as np
import optype.numpy as onp

_SCT = TypeVar("_SCT", bound=np.generic)

###

def axis_slice(
    a: onp.ArrayND[_SCT],
    start: SupportsIndex | None = None,
    stop: SupportsIndex | None = None,
    step: SupportsIndex | None = None,
    axis: SupportsIndex = -1,
) -> onp.ArrayND[_SCT]: ...
def axis_reverse(a: onp.ArrayND[_SCT], axis: SupportsIndex = -1) -> onp.ArrayND[_SCT]: ...

#
def odd_ext(x: onp.ArrayND[_SCT], n: onp.ToInt, axis: SupportsIndex = -1) -> onp.ArrayND[_SCT]: ...
def even_ext(x: onp.ArrayND[_SCT], n: onp.ToInt, axis: SupportsIndex = -1) -> onp.ArrayND[_SCT]: ...
def const_ext(x: onp.ArrayND[_SCT], n: onp.ToInt, axis: SupportsIndex = -1) -> onp.ArrayND[_SCT]: ...
def zero_ext(x: onp.ArrayND[_SCT], n: onp.ToInt, axis: SupportsIndex = -1) -> onp.ArrayND[_SCT]: ...

#
@overload
def _validate_fs(fs: None, allow_none: Literal[True] = True) -> None: ...
@overload
def _validate_fs(fs: onp.ToFloat, allow_none: bool = True) -> float: ...
