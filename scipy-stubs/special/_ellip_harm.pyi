from typing import Literal, overload

import numpy as np
import optype.numpy as onp

#
@overload  # 0d
def ellip_harm(
    h2: onp.ToFloat,
    k2: onp.ToFloat,
    n: onp.ToInt,
    p: onp.ToFloat,
    s: onp.ToFloat,
    signm: Literal[-1, 1] = 1,
    signn: Literal[-1, 1] = 1,
) -> np.float64: ...
@overload  # Nd h2
def ellip_harm(
    h2: onp.ToFloatND,
    k2: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt | onp.ToIntND,
    p: onp.ToFloat | onp.ToFloatND,
    s: onp.ToFloat | onp.ToFloatND,
    signm: Literal[-1, 1] = 1,
    signn: Literal[-1, 1] = 1,
) -> onp.ArrayND[np.float64]: ...
@overload  # Nd k2
def ellip_harm(
    h2: onp.ToFloat | onp.ToFloatND,
    k2: onp.ToFloatND,
    n: onp.ToInt | onp.ToIntND,
    p: onp.ToFloat | onp.ToFloatND,
    s: onp.ToFloat | onp.ToFloatND,
    signm: Literal[-1, 1] = 1,
    signn: Literal[-1, 1] = 1,
) -> onp.ArrayND[np.float64]: ...
@overload  # Nd n
def ellip_harm(
    h2: onp.ToFloat | onp.ToFloatND,
    k2: onp.ToFloat | onp.ToFloatND,
    n: onp.ToIntND,
    p: onp.ToFloat | onp.ToFloatND,
    s: onp.ToFloat | onp.ToFloatND,
    signm: Literal[-1, 1] = 1,
    signn: Literal[-1, 1] = 1,
) -> onp.ArrayND[np.float64]: ...
@overload  # Nd p
def ellip_harm(
    h2: onp.ToFloat | onp.ToFloatND,
    k2: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt | onp.ToIntND,
    p: onp.ToFloatND,
    s: onp.ToFloat | onp.ToFloatND,
    signm: Literal[-1, 1] = 1,
    signn: Literal[-1, 1] = 1,
) -> onp.ArrayND[np.float64]: ...
@overload  # Nd s
def ellip_harm(
    h2: onp.ToFloat | onp.ToFloatND,
    k2: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt | onp.ToIntND,
    p: onp.ToFloat | onp.ToFloatND,
    s: onp.ToFloatND,
    signm: Literal[-1, 1] = 1,
    signn: Literal[-1, 1] = 1,
) -> onp.ArrayND[np.float64]: ...

#
@overload  # 0d
def ellip_harm_2(h2: onp.ToFloat, k2: onp.ToFloat, n: onp.ToInt, p: onp.ToInt, s: onp.ToFloat) -> onp.Array0D[np.float64]: ...
@overload  # Nd
def ellip_harm_2(
    h2: onp.ToFloat | onp.ToFloatND,
    k2: onp.ToFloat | onp.ToFloatND,
    n: onp.ToInt | onp.ToIntND,
    p: onp.ToInt | onp.ToIntND,
    s: onp.ToFloat | onp.ToFloatND,
) -> onp.ArrayND[np.float64]: ...

#
@overload  # 0d
def ellip_normal(h2: onp.ToFloat, k2: onp.ToFloat, n: onp.ToFloat, p: onp.ToFloat) -> onp.Array0D[np.float64]: ...
@overload  # Nd
def ellip_normal(
    h2: onp.ToFloat | onp.ToFloatND,
    k2: onp.ToFloat | onp.ToFloatND,
    n: onp.ToFloat | onp.ToFloatND,
    p: onp.ToFloat | onp.ToFloatND,
) -> onp.ArrayND[np.float64]: ...
