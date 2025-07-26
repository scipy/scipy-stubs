from typing import TypeAlias, assert_type, type_check_only

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.integrate import odeint

# based on the example from the `odeint` docstring
@type_check_only
def pend_0(y: onp.ArrayND[npc.floating], t: float) -> list[float]: ...
@type_check_only
def pend_ty_0(t: float, y: onp.ArrayND[npc.floating]) -> list[float]: ...
@type_check_only
def pend(y: onp.ArrayND[npc.floating], t: float, b: float, c: float) -> list[float]: ...
@type_check_only
def pend_ty(t: float, y: onp.ArrayND[npc.floating], b: float, c: float) -> list[float]: ...

y0: list[float] = ...
b: float = ...
c: float = ...
t: onp.ArrayND[np.float64] = ...

_VecF64: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_MatF64: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]

###

assert_type(odeint(pend_0, y0, t), _MatF64)
assert_type(odeint(pend_0, y0, t, full_output=True)[0], _MatF64)
assert_type(odeint(pend_0, y0, t, full_output=True)[1]["hu"], _VecF64)
assert_type(odeint(pend_0, y0, t, full_output=True)[1]["message"], str)

assert_type(odeint(pend_ty_0, y0, t, tfirst=True), _MatF64)
assert_type(odeint(pend_ty_0, y0, t, full_output=True, tfirst=True)[0], _MatF64)
assert_type(odeint(pend_ty_0, y0, t, full_output=True, tfirst=True)[1]["hu"], _VecF64)
assert_type(odeint(pend_ty_0, y0, t, full_output=True, tfirst=True)[1]["message"], str)

assert_type(odeint(pend, y0, t, args=(b, c)), _MatF64)
assert_type(odeint(pend, y0, t, full_output=True, args=(b, c))[0], _MatF64)
assert_type(odeint(pend, y0, t, full_output=True, args=(b, c))[1]["hu"], _VecF64)
assert_type(odeint(pend, y0, t, full_output=True, args=(b, c))[1]["message"], str)

assert_type(odeint(pend_ty, y0, t, args=(b, c), tfirst=True), _MatF64)
assert_type(odeint(pend_ty, y0, t, full_output=True, args=(b, c), tfirst=True)[0], _MatF64)
assert_type(odeint(pend_ty, y0, t, full_output=True, args=(b, c), tfirst=True)[1]["hu"], _VecF64)
assert_type(odeint(pend_ty, y0, t, full_output=True, args=(b, c), tfirst=True)[1]["message"], str)
