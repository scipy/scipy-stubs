from typing import TypeAlias, assert_type

import numpy as np
import numpy.typing as npt

from scipy.linalg import svd

ArrayF32: TypeAlias = npt.NDArray[np.float32]
ArrayF64: TypeAlias = npt.NDArray[np.float64]
ArrayC64: TypeAlias = npt.NDArray[np.complex64]
ArrayC128: TypeAlias = npt.NDArray[np.complex128]

###

py_i_2d: list[list[int]]
py_f_2d: list[list[float]]
py_c_2d: list[list[complex]]

f16_nd: npt.NDArray[np.float16]
f32_nd: npt.NDArray[np.float32]
f64_nd: npt.NDArray[np.float64]
f80_nd: npt.NDArray[np.longdouble]

c64_nd: npt.NDArray[np.complex64]
c128_nd: npt.NDArray[np.complex128]
c160_nd: npt.NDArray[np.clongdouble]

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
