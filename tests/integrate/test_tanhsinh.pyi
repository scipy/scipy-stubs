from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.integrate import nsum, tanhsinh

def integrand_f(x: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]: ...
def integrand_c(x: onp.ArrayND[np.float64 | np.complex128]) -> onp.ArrayND[np.complex128]: ...

###
# tanhsinh

assert_type(tanhsinh(integrand_f, 0.0, 1.0).integral, np.float64)
assert_type(tanhsinh(integrand_c, 0.0, 1.0).integral, np.complex128)

assert_type(tanhsinh(integrand_f, 0.0, 1.0, preserve_shape=True).integral, onp.ArrayND[np.float64] | Any)
assert_type(tanhsinh(integrand_c, 0.0, 1.0, preserve_shape=True).integral, onp.ArrayND[np.complex128] | Any)

assert_type(tanhsinh(integrand_f, [0.0, 0.5], 1.0).integral, onp.ArrayND[np.float64])
assert_type(tanhsinh(integrand_c, [0.0, 0.5], 1.0).error, onp.ArrayND[np.complex128])
assert_type(tanhsinh(integrand_f, [0.0, 0.5], 1.0, preserve_shape=True).integral, onp.ArrayND[np.float64])
assert_type(tanhsinh(integrand_c, [0.0, 0.5], 1.0, preserve_shape=True).error, onp.ArrayND[np.complex128])

assert_type(tanhsinh(integrand_f, 0.0, [1.0, 2.0]).integral, onp.ArrayND[np.float64])
assert_type(tanhsinh(integrand_c, 0.0, [1.0, 2.0]).error, onp.ArrayND[np.complex128])
assert_type(tanhsinh(integrand_f, 0.0, [1.0, 2.0], preserve_shape=True).integral, onp.ArrayND[np.float64])
assert_type(tanhsinh(integrand_c, 0.0, [1.0, 2.0], preserve_shape=True).error, onp.ArrayND[np.complex128])

assert_type(tanhsinh(integrand_f, [0.0, 1.0], [1.0, 2.0]).integral, onp.ArrayND[np.float64])
assert_type(tanhsinh(integrand_c, [0.0, 1.0], [1.0, 2.0]).error, onp.ArrayND[np.complex128])
assert_type(tanhsinh(integrand_f, [0.0, 1.0], [1.0, 2.0], preserve_shape=True).integral, onp.ArrayND[np.float64])
assert_type(tanhsinh(integrand_c, [0.0, 1.0], [1.0, 2.0], preserve_shape=True).error, onp.ArrayND[np.complex128])

###
# nsum (only reals)

assert_type(nsum(integrand_f, 0.0, 1.0).sum, np.float64)
assert_type(nsum(integrand_f, [0.0], 1.0).sum, onp.ArrayND[np.float64])
assert_type(nsum(integrand_f, 0.0, [1.0]).sum, onp.ArrayND[np.float64])
assert_type(nsum(integrand_f, [0.0], [1.0]).sum, onp.ArrayND[np.float64])
assert_type(nsum(integrand_f, 0.0, 1.0, step=[1, 2]).sum, onp.ArrayND[np.float64])
assert_type(nsum(integrand_f, [0.0], [1.0], step=[1, 2]).sum, onp.ArrayND[np.float64])
