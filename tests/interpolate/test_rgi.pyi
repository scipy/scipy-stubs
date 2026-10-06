from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import RegularGridInterpolator, interpn

# regression test for https://github.com/scipy/scipy-stubs/issues/497
RegularGridInterpolator(np.array([], dtype=np.float64), np.array([], dtype=np.float64))

###
# interpn

_f64_1d: onp.Array1D[np.float64]
_f64_c128_1d: onp.Array1D[np.float64 | np.complex128]
_py_c_2d: list[list[complex]]
_solver_args: dict[str, float]

assert_type(interpn((_f64_1d,), _f64_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(interpn((_f64_1d, _f64_1d), _py_c_2d, _f64_1d), onp.ArrayND[np.complex128])

###
# RegularGridInterpolator

assert_type(RegularGridInterpolator((_f64_1d,), _f64_1d, solver_args=_solver_args), RegularGridInterpolator[np.float64])
assert_type(
    RegularGridInterpolator((_f64_1d, _f64_1d), _py_c_2d, solver_args=_solver_args), RegularGridInterpolator[np.complex128]
)
assert_type(RegularGridInterpolator((_f64_1d,), _f64_c128_1d, solver_args=_solver_args), RegularGridInterpolator[Any])
