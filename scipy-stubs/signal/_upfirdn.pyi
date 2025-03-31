from typing import Any, Literal, TypeAlias

import numpy as np
import optype.numpy as onp

__all__ = ["_output_len", "upfirdn"]

_FIRMode: TypeAlias = Literal["constant", "symmetric", "reflect", "wrap"]
_int64_t: TypeAlias = int | bool | np.int64  # noqa: PYI042

class _UpFIRDn:
    def __init__(
        self, /, h: onp.ArrayND[np.floating[Any]], x_dtype: np.dtype[np.floating[Any]], up: int | bool, down: int | bool
    ) -> None: ...
    def apply_filter(
        self,
        /,
        x: onp.ArrayND[np.number[Any]],
        axis: int | bool = -1,
        mode: _FIRMode = "constant",
        cval: int | bool = 0,
    ) -> onp.ArrayND[np.floating[Any]]: ...

def upfirdn(
    h: onp.AnyFloatingArray,
    x: onp.AnyIntegerArray | onp.AnyFloatingArray,
    up: int | bool = 1,
    down: int | bool = 1,
    axis: int | bool = -1,
    mode: _FIRMode = "constant",
    cval: float | int | bool = 0,
) -> onp.ArrayND[np.floating[Any]]: ...

# originally defined in `scipy/signal/_upfirdn_apply.pyx` (as `(((in_len - 1) * up + len_h) - 1) // down + 1`)
def _output_len(len_h: _int64_t, in_len: _int64_t, up: _int64_t, down: _int64_t) -> _int64_t: ...
