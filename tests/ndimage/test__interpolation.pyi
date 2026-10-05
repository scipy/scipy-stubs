# type-tests for `ndimage/_interpolation.pyi`

import ctypes as ct
from typing import assert_type
from typing_extensions import CapsuleType

import numpy as np
import optype.numpy as onp

from scipy import LowLevelCallable
from scipy.ndimage import (
    affine_transform,
    geometric_transform,
    map_coordinates,
    rotate,
    shift,
    spline_filter,
    spline_filter1d,
    zoom,
)

###

_i: int

_i16_2d: onp.Array2D[np.int16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

def _mapping(output_coords: tuple[int, ...]) -> tuple[float, ...]: ...

_mapping_llc: LowLevelCallable[CapsuleType, ct.c_void_p]

###

# spline_filter1d

assert_type(spline_filter1d(_f64_2d), onp.ArrayND[np.float64])
assert_type(spline_filter1d(_py_f_2d), onp.ArrayND[np.float64])

assert_type(spline_filter1d(_f64_2d, output=complex), onp.ArrayND[np.complex128])

assert_type(spline_filter1d(_f64_2d, 3, -1, np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(spline_filter1d(_f64_2d, output=np.dtype(np.complex64)), onp.ArrayND[np.complex64])
assert_type(spline_filter1d(_f64_2d, _i), onp.ArrayND[np.float64])

# spline_filter

assert_type(spline_filter(_f64_2d), onp.ArrayND[np.float64])
assert_type(spline_filter(_py_f_2d), onp.ArrayND[np.float64])

assert_type(spline_filter(_f64_2d, output=complex), onp.ArrayND[np.complex128])

assert_type(spline_filter(_f64_2d, 3, np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(spline_filter(_f64_2d, output=np.dtype(np.complex64)), onp.ArrayND[np.complex64])
assert_type(spline_filter(_f64_2d, _i), onp.ArrayND[np.float64])

# geometric_transform

assert_type(geometric_transform(_i16_2d, _mapping), onp.Array2D[np.int16])
assert_type(geometric_transform(_f32_2d, _mapping), onp.Array2D[np.float32])
assert_type(geometric_transform(_f64_2d, _mapping), onp.Array2D[np.float64])
assert_type(geometric_transform(_c64_2d, _mapping), onp.Array2D[np.complex64])
assert_type(geometric_transform(_c128_2d, _mapping), onp.Array2D[np.complex128])
assert_type(geometric_transform(_py_i_2d, _mapping), onp.ArrayND[np.int_])
assert_type(geometric_transform(_py_f_2d, _mapping), onp.ArrayND[np.float64])
assert_type(geometric_transform(_py_c_2d, _mapping), onp.ArrayND[np.complex128])

assert_type(geometric_transform(_f64_2d, _mapping, output=int), onp.ArrayND[np.int_])
assert_type(geometric_transform(_f64_2d, _mapping, output=float), onp.ArrayND[np.float64])
assert_type(geometric_transform(_f64_2d, _mapping, output=complex), onp.ArrayND[np.complex128])
assert_type(geometric_transform(_f64_2d, _mapping, None, np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(geometric_transform(_f64_2d, _mapping, output=np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(geometric_transform(_f64_2d, _mapping_llc), onp.Array2D[np.float64])
assert_type(geometric_transform(_f64_2d, _mapping, order=_i), onp.Array2D[np.float64])

# map_coordinates

assert_type(map_coordinates(_i16_2d, _f64_2d), onp.ArrayND[np.int16])
assert_type(map_coordinates(_f32_2d, _f64_2d), onp.ArrayND[np.float32])
assert_type(map_coordinates(_f64_2d, _f64_2d), onp.ArrayND[np.float64])
assert_type(map_coordinates(_c64_2d, _f64_2d), onp.ArrayND[np.complex64])
assert_type(map_coordinates(_c128_2d, _f64_2d), onp.ArrayND[np.complex128])
assert_type(map_coordinates(_py_i_2d, _f64_2d), onp.ArrayND[np.int_])
assert_type(map_coordinates(_py_f_2d, _f64_2d), onp.ArrayND[np.float64])
assert_type(map_coordinates(_py_c_2d, _f64_2d), onp.ArrayND[np.complex128])

assert_type(map_coordinates(_f64_2d, _f64_2d, output=bool), onp.ArrayND[np.bool])
assert_type(map_coordinates(_f64_2d, _f64_2d, output=int), onp.ArrayND[np.int_])
assert_type(map_coordinates(_f64_2d, _f64_2d, output=float), onp.ArrayND[np.float64])
assert_type(map_coordinates(_f64_2d, _f64_2d, output=complex), onp.ArrayND[np.complex128])
assert_type(map_coordinates(_f64_2d, _f64_2d, np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(map_coordinates(_f64_2d, _f64_2d, order=_i), onp.ArrayND[np.float64])

# affine_transform

assert_type(affine_transform(_i16_2d, _f64_2d), onp.ArrayND[np.int16])
assert_type(affine_transform(_f32_2d, _f64_2d), onp.ArrayND[np.float32])
assert_type(affine_transform(_f64_2d, _f64_2d), onp.ArrayND[np.float64])
assert_type(affine_transform(_c64_2d, _f64_2d), onp.ArrayND[np.complex64])
assert_type(affine_transform(_c128_2d, _f64_2d), onp.ArrayND[np.complex128])
assert_type(affine_transform(_py_i_2d, _f64_2d), onp.ArrayND[np.int_])
assert_type(affine_transform(_py_f_2d, _f64_2d), onp.ArrayND[np.float64])
assert_type(affine_transform(_py_c_2d, _f64_2d), onp.ArrayND[np.complex128])

assert_type(affine_transform(_f64_2d, _f64_2d, output=int), onp.ArrayND[np.int_])
assert_type(affine_transform(_f64_2d, _f64_2d, output=float), onp.ArrayND[np.float64])
assert_type(affine_transform(_f64_2d, _f64_2d, output=complex), onp.ArrayND[np.complex128])
assert_type(affine_transform(_f64_2d, _f64_2d, output=np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(affine_transform(_f64_2d, _f64_2d, order=_i), onp.ArrayND[np.float64])

# shift

assert_type(shift(_i16_2d, 1.0), onp.Array2D[np.int16])
assert_type(shift(_f32_2d, 1.0), onp.Array2D[np.float32])
assert_type(shift(_f64_2d, 1.0), onp.Array2D[np.float64])
assert_type(shift(_c64_2d, 1.0), onp.Array2D[np.complex64])
assert_type(shift(_c128_2d, 1.0), onp.Array2D[np.complex128])
assert_type(shift(_py_i_2d, 1.0), onp.ArrayND[np.int_])
assert_type(shift(_py_f_2d, 1.0), onp.ArrayND[np.float64])
assert_type(shift(_py_c_2d, 1.0), onp.ArrayND[np.complex128])

assert_type(shift(_f64_2d, 1.0, output=int), onp.ArrayND[np.int_])
assert_type(shift(_f64_2d, 1.0, output=float), onp.ArrayND[np.float64])
assert_type(shift(_f64_2d, 1.0, output=complex), onp.ArrayND[np.complex128])
assert_type(shift(_f64_2d, 1.0, np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(shift(_f64_2d, 1.0, order=_i), onp.Array2D[np.float64])

# zoom

assert_type(zoom(_i16_2d, 2.0), onp.Array2D[np.int16])
assert_type(zoom(_f32_2d, 2.0), onp.Array2D[np.float32])
assert_type(zoom(_f64_2d, 2.0), onp.Array2D[np.float64])
assert_type(zoom(_c64_2d, 2.0), onp.Array2D[np.complex64])
assert_type(zoom(_c128_2d, 2.0), onp.Array2D[np.complex128])
assert_type(zoom(_py_i_2d, 2.0), onp.ArrayND[np.int_])
assert_type(zoom(_py_f_2d, 2.0), onp.ArrayND[np.float64])
assert_type(zoom(_py_c_2d, 2.0), onp.ArrayND[np.complex128])

assert_type(zoom(_f64_2d, 2.0, output=int), onp.ArrayND[np.int_])
assert_type(zoom(_f64_2d, 2.0, output=float), onp.ArrayND[np.float64])
assert_type(zoom(_f64_2d, 2.0, output=complex), onp.ArrayND[np.complex128])
assert_type(zoom(_f64_2d, 2.0, np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(zoom(_f64_2d, 2.0, order=_i), onp.Array2D[np.float64])

# rotate

assert_type(rotate(_i16_2d, 2.0), onp.Array2D[np.int16])
assert_type(rotate(_f32_2d, 2.0), onp.Array2D[np.float32])
assert_type(rotate(_f64_2d, 2.0), onp.Array2D[np.float64])
assert_type(rotate(_c64_2d, 2.0), onp.Array2D[np.complex64])
assert_type(rotate(_c128_2d, 2.0), onp.Array2D[np.complex128])
assert_type(rotate(_py_i_2d, 2.0), onp.ArrayND[np.int_])
assert_type(rotate(_py_f_2d, 2.0), onp.ArrayND[np.float64])
assert_type(rotate(_py_c_2d, 2.0), onp.ArrayND[np.complex128])

assert_type(rotate(_f64_2d, 2.0, output=int), onp.ArrayND[np.int_])
assert_type(rotate(_f64_2d, 2.0, output=float), onp.ArrayND[np.float64])
assert_type(rotate(_f64_2d, 2.0, output=complex), onp.ArrayND[np.complex128])
assert_type(rotate(_f64_2d, 2.0, output=np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(rotate(_f64_2d, 2.0, order=_i), onp.Array2D[np.float64])
