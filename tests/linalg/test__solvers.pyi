# type-tests for `linalg/_solvers.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import (
    solve_continuous_are,
    solve_continuous_lyapunov,
    solve_discrete_are,
    solve_discrete_lyapunov,
    solve_lyapunov,
    solve_sylvester,
)

###

type _FloatND = onp.ArrayND[np.float32 | np.float64]
type _ComplexND = onp.ArrayND[np.complex64 | np.complex128]

###

_bool_nd: onp.ArrayND[np.bool]
_i8_nd: onp.ArrayND[np.int8]
_i32_nd: onp.ArrayND[np.int32]
_f16_nd: onp.ArrayND[np.float16]
_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]
_f80_nd: onp.ArrayND[np.float128]
_c64_nd: onp.ArrayND[np.complex64]
_c128_nd: onp.ArrayND[np.complex128]
_c160_nd: onp.ArrayND[np.complex256]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

###
# solve_sylvester

assert_type(solve_sylvester(_f32_nd, _f32_nd, _f32_nd), onp.ArrayND[np.float32])
assert_type(solve_sylvester(_f32_nd, _f32_nd, _i8_nd), onp.ArrayND[np.float32])
assert_type(solve_sylvester(_f32_nd, _f32_nd, _i32_nd), onp.ArrayND[np.float64])
assert_type(solve_sylvester(_i8_nd, _f32_nd, _f32_nd), onp.ArrayND[np.float64])
assert_type(solve_sylvester(_f32_nd, _f64_nd, _f32_nd), onp.ArrayND[np.float64])
assert_type(solve_sylvester(_f32_nd, _f32_nd, _py_f_2d), onp.ArrayND[np.float64])
assert_type(solve_sylvester(_i8_nd, _i8_nd, _i8_nd), onp.ArrayND[np.float64])
assert_type(solve_sylvester(_f64_nd, _f64_nd, _f64_nd), onp.ArrayND[np.float64])
assert_type(solve_sylvester(_py_f_2d, _py_f_2d, _py_f_2d), onp.ArrayND[np.float64])

assert_type(solve_sylvester(_c64_nd, _c64_nd, _c64_nd), onp.ArrayND[np.complex64])
assert_type(solve_sylvester(_c64_nd, _f32_nd, _i8_nd), onp.ArrayND[np.complex64])
assert_type(solve_sylvester(_f32_nd, _c64_nd, _i8_nd), onp.ArrayND[np.complex64])
assert_type(solve_sylvester(_f32_nd, _f32_nd, _c64_nd), onp.ArrayND[np.complex64])
assert_type(solve_sylvester(_c64_nd, _c64_nd, _i32_nd), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_c64_nd, _c64_nd, _f64_nd), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_c64_nd, _f64_nd, _f32_nd), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_f64_nd, _c64_nd, _f32_nd), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_f64_nd, _f32_nd, _c64_nd), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_c64_nd, _f32_nd, _py_f_2d), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_c128_nd, _c64_nd, _c64_nd), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_c128_nd, _f64_nd, _f64_nd), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_f64_nd, _c128_nd, _f64_nd), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_f64_nd, _f64_nd, _c128_nd), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_f32_nd, _f32_nd, _py_c_2d), onp.ArrayND[np.complex128])
assert_type(solve_sylvester(_py_c_2d, _py_c_2d, _py_c_2d), onp.ArrayND[np.complex128])

assert_type(solve_sylvester(_bool_nd, _f64_nd, _f64_nd), onp.ArrayND[Any])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(solve_sylvester(_f64_nd, _f16_nd, _f64_nd), onp.ArrayND[Any])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(solve_sylvester(_f64_nd, _f64_nd, _f80_nd), onp.ArrayND[Any])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(solve_sylvester(_c160_nd, _c160_nd, _c160_nd), onp.ArrayND[Any])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# solve_continuous_lyapunov / solve_discrete_lyapunov

assert_type(solve_continuous_lyapunov(_f64_nd, _f64_nd), _FloatND)
assert_type(solve_continuous_lyapunov(_c128_nd, _f64_nd), _ComplexND)
assert_type(solve_continuous_lyapunov(_f64_nd, _c128_nd), _ComplexND)

assert_type(solve_discrete_lyapunov(_f64_nd, _f64_nd), _FloatND)
assert_type(solve_discrete_lyapunov(_c128_nd, _f64_nd), _ComplexND)
assert_type(solve_discrete_lyapunov(_f64_nd, _c128_nd), _ComplexND)

###
# solve_continuous_are

assert_type(solve_continuous_are(_f64_nd, _f64_nd, _f64_nd, _f64_nd), _FloatND)
assert_type(solve_continuous_are(_c128_nd, _f64_nd, _f64_nd, _f64_nd), _ComplexND)
assert_type(solve_continuous_are(_f64_nd, _c128_nd, _f64_nd, _f64_nd), _ComplexND)
assert_type(solve_continuous_are(_f64_nd, _f64_nd, _c128_nd, _f64_nd), _ComplexND)
assert_type(solve_continuous_are(_f64_nd, _f64_nd, _f64_nd, _c128_nd), _ComplexND)
assert_type(solve_continuous_are(_f64_nd, _f64_nd, _f64_nd, _f64_nd, _c128_nd), _ComplexND)
assert_type(solve_continuous_are(_f64_nd, _f64_nd, _f64_nd, _f64_nd, s=_c128_nd), _ComplexND)

###
# solve_discrete_are

assert_type(solve_discrete_are(_f64_nd, _f64_nd, _f64_nd, _f64_nd), _FloatND)
assert_type(solve_discrete_are(_c128_nd, _f64_nd, _f64_nd, _f64_nd), _ComplexND)
assert_type(solve_discrete_are(_f64_nd, _c128_nd, _f64_nd, _f64_nd), _ComplexND)
assert_type(solve_discrete_are(_f64_nd, _f64_nd, _c128_nd, _f64_nd), _ComplexND)
assert_type(solve_discrete_are(_f64_nd, _f64_nd, _f64_nd, _c128_nd), _ComplexND)
assert_type(solve_discrete_are(_f64_nd, _f64_nd, _f64_nd, _f64_nd, _c128_nd), _ComplexND)
assert_type(solve_discrete_are(_f64_nd, _f64_nd, _f64_nd, _f64_nd, s=_c128_nd), _ComplexND)

###
# solve_lyapunov  (alias for solve_continuous_lyapunov)

assert_type(solve_lyapunov(_f64_nd, _f64_nd), _FloatND)
assert_type(solve_lyapunov(_c128_nd, _f64_nd), _ComplexND)
assert_type(solve_lyapunov(_f64_nd, _c128_nd), _ComplexND)
