from collections.abc import Callable
from typing import Concatenate, TypeAlias, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import AnyBool

__all__ = ["convolve", "convolve_z", "destroy_convolve_cache", "init_convolution_kernel"]

_Float1D: TypeAlias = onp.Array1D[np.float64]

# NOTE: this doesn't do anything; nothing is cached
def destroy_convolve_cache() -> None: ...

#
def convolve(
    inout: onp.ToFloat1D,
    omega: onp.ToFloat1D,
    swap_real_imag: AnyBool = False,
    overwrite_x: AnyBool = False,
) -> _Float1D: ...

#
def convolve_z(
    inout: onp.ToFloat1D,
    omega_real: onp.ToFloat1D,
    omega_imag: onp.ToFloat1D,
    overwrite_x: AnyBool = False,
) -> _Float1D: ...

#
@overload
def init_convolution_kernel(
    n: onp.ToInt,
    kernel_func: Callable[[int | bool], float | int | bool],
    d: onp.ToInt = 0,
    zero_nyquist: onp.ToInt | None = None,
    kernel_func_extra_args: tuple[()] = (),
) -> _Float1D: ...
@overload
def init_convolution_kernel(
    n: onp.ToInt,
    kernel_func: Callable[Concatenate[int | bool, ...], float | int | bool],
    d: onp.ToInt,
    zero_nyquist: onp.ToInt | None,
    kernel_func_extra_args: tuple[object, ...],
) -> _Float1D: ...
@overload
def init_convolution_kernel(
    n: onp.ToInt,
    kernel_func: Callable[Concatenate[int | bool, ...], float | int | bool],
    d: onp.ToInt = 0,
    zero_nyquist: onp.ToInt | None = None,
    *,
    kernel_func_extra_args: tuple[object, ...],
) -> _Float1D: ...
