from typing import TypeAlias, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.linalg import diagsvd, null_space, orth, subspace_angles, svd, svdvals

ArrayF32: TypeAlias = onp.ArrayND[np.float32]
ArrayF64: TypeAlias = onp.ArrayND[np.float64]
ArrayC64: TypeAlias = onp.ArrayND[np.complex64]
ArrayC128: TypeAlias = onp.ArrayND[np.complex128]

###

py_i_2d: list[list[int]]
py_f_2d: list[list[float]]
py_c_2d: list[list[complex]]

f16_nd: npt.NDArray[np.float16]
f32_nd: npt.NDArray[np.float32]
f64_nd: npt.NDArray[np.float64]
f80_nd: npt.NDArray[np.float128]

c64_nd: npt.NDArray[np.complex64]
c128_nd: npt.NDArray[np.complex128]
c160_nd: npt.NDArray[np.complex256]

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

###
# svdvals

assert_type(svdvals(f64_nd), onp.ArrayND[np.float64 | np.float32])
assert_type(svdvals(f32_nd), onp.ArrayND[np.float64 | np.float32])
assert_type(svdvals(c128_nd), onp.ArrayND[np.float64 | np.float32])
assert_type(svdvals(c64_nd), onp.ArrayND[np.float64 | np.float32])
assert_type(svdvals(py_f_2d), onp.ArrayND[np.float64 | np.float32])
assert_type(svdvals(py_c_2d), onp.ArrayND[np.float64 | np.float32])

###
# diagsvd

assert_type(diagsvd(f64_nd, 3, 4), onp.ArrayND[np.float64])
assert_type(diagsvd(f32_nd, 3, 4), onp.ArrayND[np.float32])
assert_type(diagsvd([True, False], 2, 3), onp.ArrayND[np.bool_])  # type: ignore[assert-type]  # mypy bug
assert_type(diagsvd([1, 2], 2, 3), onp.ArrayND[np.intp])  # type: ignore[assert-type]  # mypy bug
assert_type(diagsvd([1.0, 2.0], 2, 3), onp.ArrayND[np.float64])  # type: ignore[assert-type]  # mypy bug

###
# orth

assert_type(orth(f64_nd), onp.ArrayND[np.float64])
assert_type(orth(py_f_2d), onp.ArrayND[np.float64])
assert_type(orth(py_c_2d), onp.ArrayND[np.complex128])
assert_type(orth(f32_nd), onp.ArrayND[np.float32])
assert_type(orth(c64_nd), onp.ArrayND[np.complex64])

###
# null_space

assert_type(null_space(f64_nd), onp.ArrayND[np.float64])
assert_type(null_space(py_f_2d), onp.ArrayND[np.float64])
assert_type(null_space(py_c_2d), onp.ArrayND[np.complex128])
assert_type(null_space(f32_nd), onp.ArrayND[np.float32])
assert_type(null_space(c64_nd), onp.ArrayND[np.complex64])

###
# subspace_angles

assert_type(subspace_angles(f64_nd, f64_nd), onp.ArrayND[np.float64 | np.float32])
assert_type(subspace_angles(c128_nd, c128_nd), onp.ArrayND[np.float64 | np.float32])
assert_type(subspace_angles(py_f_2d, py_f_2d), onp.ArrayND[np.float64 | np.float32])
