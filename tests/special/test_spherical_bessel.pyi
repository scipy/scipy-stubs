from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.special import spherical_in, spherical_jn, spherical_kn, spherical_yn

_Inexact: TypeAlias = np.float64 | np.complex128
_InexactND: TypeAlias = onp.ArrayND[np.float64 | np.complex128]
_FloatND: TypeAlias = onp.ArrayND[np.float64]

n_scalar: int
n_arr: onp.ArrayND[np.intp]
z_float: float
z_float_arr: onp.ArrayND[np.float64]
z_complex: complex
z_complex_arr: onp.ArrayND[np.complex128]

# spherical_jn
assert_type(spherical_jn(n_scalar, z_float), np.float64)
assert_type(spherical_jn(n_scalar, z_float_arr), _FloatND)
assert_type(spherical_jn(n_arr, z_float), _FloatND)
assert_type(spherical_jn(n_scalar, z_complex), _Inexact)
assert_type(spherical_jn(n_scalar, z_complex_arr), _InexactND)
assert_type(spherical_jn(n_arr, z_complex), _InexactND)

# spherical_yn
assert_type(spherical_yn(n_scalar, z_float), np.float64)
assert_type(spherical_yn(n_scalar, z_float_arr), _FloatND)
assert_type(spherical_yn(n_arr, z_float), _FloatND)
assert_type(spherical_yn(n_scalar, z_complex), _Inexact)
assert_type(spherical_yn(n_scalar, z_complex_arr), _InexactND)
assert_type(spherical_yn(n_arr, z_complex), _InexactND)

# spherical_in
assert_type(spherical_in(n_scalar, z_float), np.float64)
assert_type(spherical_in(n_scalar, z_float_arr), _FloatND)
assert_type(spherical_in(n_arr, z_float), _FloatND)
assert_type(spherical_in(n_scalar, z_complex), _Inexact)
assert_type(spherical_in(n_scalar, z_complex_arr), _InexactND)
assert_type(spherical_in(n_arr, z_complex), _InexactND)

# spherical_kn
assert_type(spherical_kn(n_scalar, z_float), np.float64)
assert_type(spherical_kn(n_scalar, z_float_arr), _FloatND)
assert_type(spherical_kn(n_arr, z_float), _FloatND)
assert_type(spherical_kn(n_scalar, z_complex), _Inexact)
assert_type(spherical_kn(n_scalar, z_complex_arr), _InexactND)
assert_type(spherical_kn(n_arr, z_complex), _InexactND)
