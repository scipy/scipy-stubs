from typing import TypeAlias, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.linalg import svd

ArrayF32: TypeAlias = onp.ArrayND[np.float32]
ArrayF64: TypeAlias = onp.ArrayND[np.float64]
ArrayC64: TypeAlias = onp.ArrayND[np.complex64]
ArrayC128: TypeAlias = onp.ArrayND[np.complex128]

###

py_i_2d: list[list[int]]
py_f_2d: list[list[float]]
py_c_2d: list[list[complex]]

f16_nd: onp.ArrayND[np.float16]
f32_nd: onp.ArrayND[np.float32]
f64_nd: onp.ArrayND[np.float64]
f80_nd: onp.ArrayND[np.longdouble]

c64_nd: onp.ArrayND[np.complex64]
c128_nd: onp.ArrayND[np.complex128]
c160_nd: onp.ArrayND[np.clongdouble]

###
# svd

assert_type(svd(py_i_2d), tuple[ArrayF64, ArrayF64, ArrayF64])
assert_type(svd(py_f_2d), tuple[ArrayF64, ArrayF64, ArrayF64])
assert_type(svd(py_c_2d), tuple[ArrayC128, ArrayF64, ArrayC128])
assert_type(svd(f16_nd), tuple[ArrayF32, ArrayF32, ArrayF32])
assert_type(svd(f32_nd), tuple[ArrayF32, ArrayF32, ArrayF32])
assert_type(svd(f64_nd), tuple[ArrayF64, ArrayF64, ArrayF64])
assert_type(svd(f80_nd), tuple[ArrayF64, ArrayF64, ArrayF64])
assert_type(svd(c64_nd), tuple[ArrayC64, ArrayF32, ArrayC64])
assert_type(svd(c128_nd), tuple[ArrayC128, ArrayF64, ArrayC128])
assert_type(svd(c160_nd), tuple[ArrayC128, ArrayF64, ArrayC128])

assert_type(svd(py_i_2d, compute_uv=False), ArrayF64)
assert_type(svd(py_f_2d, compute_uv=False), ArrayF64)
assert_type(svd(py_c_2d, compute_uv=False), ArrayF64)
assert_type(svd(f16_nd, compute_uv=False), ArrayF32)
assert_type(svd(f32_nd, compute_uv=False), ArrayF32)
assert_type(svd(f64_nd, compute_uv=False), ArrayF64)
assert_type(svd(f80_nd, compute_uv=False), ArrayF64)
assert_type(svd(c64_nd, compute_uv=False), ArrayF32)
assert_type(svd(c128_nd, compute_uv=False), ArrayF64)
assert_type(svd(c160_nd, compute_uv=False), ArrayF64)

####
# TODO: test the remaining functions in `_decomp_svd.pyi`
