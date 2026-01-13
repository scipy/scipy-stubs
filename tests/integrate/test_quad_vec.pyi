from typing import assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.integrate import quad_vec

###

def _f_float(x: float) -> float: ...

assert_type(quad_vec(_f_float, 0.0, 1.0), tuple[float, float])
assert_type(quad_vec(_f_float, 0.0, 1.0, full_output=True)[0], float)
assert_type(quad_vec(_f_float, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_float, 0.0, 1.0, full_output=True)[2].integrals, onp.Array1D[np.float64])

def _f_complex(x: float) -> complex: ...

assert_type(quad_vec(_f_complex, 0.0, 1.0), tuple[complex, float])
assert_type(quad_vec(_f_complex, 0.0, 1.0, full_output=True)[0], complex)
assert_type(quad_vec(_f_complex, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_complex, 0.0, 1.0, full_output=True)[2].integrals, onp.Array1D[np.complex128])

def _f_i64(x: float) -> np.int64: ...

assert_type(quad_vec(_f_i64, 0.0, 1.0), tuple[np.float64, float])
assert_type(quad_vec(_f_i64, 0.0, 1.0, full_output=True)[0], np.float64)
assert_type(quad_vec(_f_i64, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_i64, 0.0, 1.0, full_output=True)[2].integrals, onp.Array1D[np.float64])

def _f_i64_nd(x: float) -> npt.NDArray[np.int64]: ...

assert_type(quad_vec(_f_i64_nd, 0.0, 1.0), tuple[onp.Array1D[np.float64], float])
assert_type(quad_vec(_f_i64_nd, 0.0, 1.0, full_output=True)[0], onp.Array1D[np.float64])
assert_type(quad_vec(_f_i64_nd, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_i64_nd, 0.0, 1.0, full_output=True)[2].integrals, onp.Array2D[np.float64])

def _f_f32(x: float) -> np.float32: ...

assert_type(quad_vec(_f_f32, 0.0, 1.0), tuple[np.float32, float])
assert_type(quad_vec(_f_f32, 0.0, 1.0, full_output=True)[0], np.float32)
assert_type(quad_vec(_f_f32, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_f32, 0.0, 1.0, full_output=True)[2].integrals, onp.Array1D[np.float32])

def _f_f32_nd(x: float) -> npt.NDArray[np.float32]: ...

assert_type(quad_vec(_f_f32_nd, 0.0, 1.0), tuple[onp.Array1D[np.float32], float])
assert_type(quad_vec(_f_f32_nd, 0.0, 1.0, full_output=True)[0], onp.Array1D[np.float32])
assert_type(quad_vec(_f_f32_nd, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_f32_nd, 0.0, 1.0, full_output=True)[2].integrals, onp.Array2D[np.float32])

def _f_f64(x: float) -> np.float64: ...

assert_type(quad_vec(_f_f64, 0.0, 1.0), tuple[np.float64, float])
assert_type(quad_vec(_f_f64, 0.0, 1.0, full_output=True)[0], np.float64)
assert_type(quad_vec(_f_f64, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_f64, 0.0, 1.0, full_output=True)[2].integrals, onp.Array1D[np.float64])

def _f_f64_nd(x: float) -> npt.NDArray[np.float64]: ...

assert_type(quad_vec(_f_f64_nd, 0.0, 1.0), tuple[onp.Array1D[np.float64], float])
assert_type(quad_vec(_f_f64_nd, 0.0, 1.0, full_output=True)[0], onp.Array1D[np.float64])
assert_type(quad_vec(_f_f64_nd, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_f64_nd, 0.0, 1.0, full_output=True)[2].integrals, onp.Array2D[np.float64])

def _f_c64(x: float) -> np.complex64: ...

assert_type(quad_vec(_f_c64, 0.0, 1.0), tuple[np.complex64, float])
assert_type(quad_vec(_f_c64, 0.0, 1.0, full_output=True)[0], np.complex64)
assert_type(quad_vec(_f_c64, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_c64, 0.0, 1.0, full_output=True)[2].integrals, onp.Array1D[np.complex64])

def _f_c64_nd(x: float) -> npt.NDArray[np.complex64]: ...

assert_type(quad_vec(_f_c64_nd, 0.0, 1.0), tuple[onp.Array1D[np.complex64], float])
assert_type(quad_vec(_f_c64_nd, 0.0, 1.0, full_output=True)[0], onp.Array1D[np.complex64])
assert_type(quad_vec(_f_c64_nd, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_c64_nd, 0.0, 1.0, full_output=True)[2].integrals, onp.Array2D[np.complex64])

def _f_c128(x: float) -> np.complex128: ...

assert_type(quad_vec(_f_c128, 0.0, 1.0), tuple[np.complex128, float])
assert_type(quad_vec(_f_c128, 0.0, 1.0, full_output=True)[0], np.complex128)
assert_type(quad_vec(_f_c128, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_c128, 0.0, 1.0, full_output=True)[2].integrals, onp.Array1D[np.complex128])

def _f_c128_nd(x: float) -> npt.NDArray[np.complex128]: ...

assert_type(quad_vec(_f_c128_nd, 0.0, 1.0), tuple[onp.Array1D[np.complex128], float])
assert_type(quad_vec(_f_c128_nd, 0.0, 1.0, full_output=True)[0], onp.Array1D[np.complex128])
assert_type(quad_vec(_f_c128_nd, 0.0, 1.0, full_output=True)[1], float)
assert_type(quad_vec(_f_c128_nd, 0.0, 1.0, full_output=True)[2].integrals, onp.Array2D[np.complex128])
