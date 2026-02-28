# type-tests for `ndimage/_interpolation.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

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

_FloatOut: TypeAlias = onp.ArrayND[np.float64 | np.float32]
_ComplexOut: TypeAlias = onp.ArrayND[np.complex128 | np.float64 | np.complex64 | np.float32]

# typed input arrays
f64_2d: onp.Array2D[np.float64]
c128_2d: onp.Array2D[np.complex128]

# plain-Python inputs (also valid: float -> ToFloat, complex -> ToComplex)
float_2d: list[list[float]]
complex_2d: list[list[complex]]

# coordinates / matrix for map_coordinates / affine_transform
coords_2d: onp.Array2D[np.float64]
matrix_2d: onp.Array2D[np.float64]

# mapping function for geometric_transform
def _mapping(output_coords: tuple[int, ...]) -> tuple[float, ...]: ...

###
# spline_filter1d

assert_type(spline_filter1d(f64_2d), onp.ArrayND[np.float64])
assert_type(spline_filter1d(float_2d), onp.ArrayND[np.float64])

assert_type(spline_filter1d(f64_2d, output=complex), onp.ArrayND[np.complex128 | np.float64])

assert_type(spline_filter1d(f64_2d, 3, -1, np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(spline_filter1d(f64_2d, output=np.dtype(np.complex64)), onp.ArrayND[np.complex64])

###
# spline_filter

assert_type(spline_filter(f64_2d), onp.ArrayND[np.float64])
assert_type(spline_filter(float_2d), onp.ArrayND[np.float64])

assert_type(spline_filter(f64_2d, output=complex), onp.ArrayND[np.complex128 | np.float64])

assert_type(spline_filter(f64_2d, 3, np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(spline_filter(f64_2d, output=np.dtype(np.complex64)), onp.ArrayND[np.complex64])

###
# geometric_transform

assert_type(geometric_transform(f64_2d, _mapping), _FloatOut)
assert_type(geometric_transform(float_2d, _mapping), _FloatOut)

assert_type(geometric_transform(c128_2d, _mapping), _ComplexOut)
assert_type(geometric_transform(complex_2d, _mapping), _ComplexOut)

assert_type(geometric_transform(f64_2d, _mapping, output=int), onp.ArrayND[np.int_])
assert_type(geometric_transform(f64_2d, _mapping, output=float), onp.ArrayND[np.float64 | np.int_])
assert_type(geometric_transform(f64_2d, _mapping, output=complex), onp.ArrayND[np.complex128 | np.float64 | np.int_])

assert_type(geometric_transform(f64_2d, _mapping, None, np.dtype(np.float32)), onp.ArrayND[np.float32])
assert_type(geometric_transform(f64_2d, _mapping, output=np.dtype(np.float32)), onp.ArrayND[np.float32])

###
# map_coordinates

assert_type(map_coordinates(f64_2d, coords_2d), _FloatOut)
assert_type(map_coordinates(float_2d, coords_2d), _FloatOut)

assert_type(map_coordinates(c128_2d, coords_2d), _ComplexOut)
assert_type(map_coordinates(complex_2d, coords_2d), _ComplexOut)

assert_type(map_coordinates(f64_2d, coords_2d, output=bool), onp.ArrayND[np.bool_])
assert_type(map_coordinates(f64_2d, coords_2d, output=int), onp.ArrayND[np.int_ | np.bool_])
assert_type(map_coordinates(f64_2d, coords_2d, output=float), onp.ArrayND[np.float64 | np.int_ | np.bool_])
assert_type(map_coordinates(f64_2d, coords_2d, output=complex), onp.ArrayND[np.complex128 | np.float64 | np.int_ | np.bool_])

assert_type(map_coordinates(f64_2d, coords_2d, np.dtype(np.float32)), onp.ArrayND[np.float32])

###
# affine_transform

assert_type(affine_transform(f64_2d, matrix_2d), _FloatOut)
assert_type(affine_transform(float_2d, matrix_2d), _FloatOut)

assert_type(affine_transform(c128_2d, matrix_2d), _ComplexOut)
assert_type(affine_transform(complex_2d, matrix_2d), _ComplexOut)

assert_type(affine_transform(f64_2d, matrix_2d, output=int), onp.ArrayND[np.int_])
assert_type(affine_transform(f64_2d, matrix_2d, output=float), onp.ArrayND[np.float64 | np.int_])
assert_type(affine_transform(f64_2d, matrix_2d, output=complex), onp.ArrayND[np.complex128 | np.float64 | np.int_])
assert_type(affine_transform(f64_2d, matrix_2d, output=np.dtype(np.float32)), onp.ArrayND[np.float32])

###
# shift

assert_type(shift(f64_2d, 1.0), _FloatOut)
assert_type(shift(float_2d, 1.0), _FloatOut)

assert_type(shift(c128_2d, 1.0), _ComplexOut)
assert_type(shift(complex_2d, 1.0), _ComplexOut)

assert_type(shift(f64_2d, 1.0, output=int), onp.ArrayND[np.int_])
assert_type(shift(f64_2d, 1.0, output=float), onp.ArrayND[np.float64 | np.int_])
assert_type(shift(f64_2d, 1.0, output=complex), onp.ArrayND[np.complex128 | np.float64 | np.int_])
assert_type(shift(f64_2d, 1.0, np.dtype(np.float32)), onp.ArrayND[np.float32])

###
# zoom

assert_type(zoom(f64_2d, 2.0), _FloatOut)
assert_type(zoom(float_2d, 2.0), _FloatOut)

assert_type(zoom(c128_2d, 2.0), _ComplexOut)
assert_type(zoom(complex_2d, 2.0), _ComplexOut)

assert_type(zoom(f64_2d, 2.0, output=int), onp.ArrayND[np.int_])
assert_type(zoom(f64_2d, 2.0, output=float), onp.ArrayND[np.float64 | np.int_])
assert_type(zoom(f64_2d, 2.0, output=complex), onp.ArrayND[np.complex128 | np.float64 | np.int_])
assert_type(zoom(f64_2d, 2.0, np.dtype(np.float32)), onp.ArrayND[np.float32])

###
# rotate

assert_type(rotate(f64_2d, 45.0), _FloatOut)
assert_type(rotate(float_2d, 45.0), _FloatOut)

assert_type(rotate(c128_2d, 45.0), _ComplexOut)
assert_type(rotate(complex_2d, 45.0), _ComplexOut)

assert_type(rotate(f64_2d, 45.0, output=int), onp.ArrayND[np.int_])
assert_type(rotate(f64_2d, 45.0, output=float), onp.ArrayND[np.float64 | np.int_])
assert_type(rotate(f64_2d, 45.0, output=complex), onp.ArrayND[np.complex128 | np.float64 | np.int_])
assert_type(rotate(f64_2d, 45.0, output=np.dtype(np.float32)), onp.ArrayND[np.float32])
