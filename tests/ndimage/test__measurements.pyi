# type-tests for `ndimage/_measurements.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.ndimage import (
    center_of_mass,
    extrema,
    find_objects,
    histogram,
    label,
    labeled_comprehension,
    maximum,
    maximum_position,
    mean,
    median,
    minimum,
    minimum_position,
    standard_deviation,
    sum,
    sum_labels,
    value_indices,
    variance,
    watershed_ift,
)

# typed test arrays
f64_2d: onp.Array2D[np.float64]
f32_2d: onp.Array2D[np.float32]
c128_2d: onp.Array2D[np.complex128]
int_2d: onp.Array2D[np.int32]
uint8_2d: onp.Array2D[np.uint8]

# label / index arrays for supervised statistics
label_1d: onp.Array1D[np.int32]
index_1d: onp.Array1D[np.intp]

# output array accepted by label(output=...)
out_labeled: onp.ArrayND[np.int32 | np.intp]

# helper function for labeled_comprehension
def _stat_func(x: onp.ToComplex | onp.ToComplexND) -> onp.ToComplex: ...

###
# label

# output array provided -> returns feature count
assert_type(label(f64_2d, output=out_labeled), int)
assert_type(label(int_2d, output=out_labeled), int)

# no output -> (labeled_array, num_features)
assert_type(label(f64_2d), tuple[onp.ArrayND[np.int32 | np.intp], int])
assert_type(label(int_2d), tuple[onp.ArrayND[np.int32 | np.intp], int])

###
# find_objects

assert_type(find_objects(int_2d), list[tuple[slice, ...]])

###
# value_indices

assert_type(value_indices(int_2d), dict[np.intp, tuple[onp.ArrayND[np.intp], ...]])

###
# labeled_comprehension

# out_dtype: DTypeLike[_SCT] -> ArrayND[_SCT]
assert_type(labeled_comprehension(f64_2d, None, None, _stat_func, np.dtype(np.float32), 0.0), onp.ArrayND[np.float32])
# out_dtype: AnyIntPDType -> ArrayND[np.intp]
assert_type(labeled_comprehension(f64_2d, None, None, _stat_func, np.intp, 0), onp.ArrayND[np.intp])
# out_dtype: AnyFloat64DType | None -> ArrayND[np.float64]
assert_type(labeled_comprehension(f64_2d, None, None, _stat_func, None, 0.0), onp.ArrayND[np.float64])
assert_type(labeled_comprehension(f64_2d, None, None, _stat_func, np.float64, 0.0), onp.ArrayND[np.float64])
# out_dtype: AnyComplex128DType -> ArrayND[np.complex128]
assert_type(labeled_comprehension(f64_2d, None, None, _stat_func, np.complex128, 0.0), onp.ArrayND[np.complex128])

###
# sum / sum_labels / mean / variance / standard_deviation / median

# scalar return (no index)
assert_type(mean(f64_2d), np.float64)
assert_type(mean(f32_2d), np.float32)
assert_type(mean(c128_2d), np.complex128)
assert_type(sum(f64_2d), np.float64)
assert_type(sum_labels(f64_2d), np.float64)
assert_type(variance(f64_2d), np.float64)
assert_type(standard_deviation(f64_2d), np.float64)
assert_type(median(f64_2d), np.float64)

# array return (array index)
assert_type(mean(f64_2d, label_1d, index_1d), onp.ArrayND[np.float64])
assert_type(mean(f32_2d, label_1d, index_1d), onp.ArrayND[np.float32])
assert_type(sum(f64_2d, label_1d, index_1d), onp.ArrayND[np.float64])

###
# minimum / maximum

assert_type(minimum(f64_2d), np.float64)
assert_type(minimum(f32_2d), np.float32)
assert_type(maximum(f64_2d), np.float64)

assert_type(minimum(f64_2d, label_1d, index_1d), onp.ArrayND[np.float64])
assert_type(maximum(f64_2d, label_1d, index_1d), onp.ArrayND[np.float64])

###
# extrema

# ArrayLike[_SCT], no index -> _Extrema0D[_SCT]
assert_type(extrema(f64_2d), tuple[np.float64, np.float64, tuple[np.intp, ...], tuple[np.intp, ...]])
assert_type(extrema(f32_2d), tuple[np.float32, np.float32, tuple[np.intp, ...], tuple[np.intp, ...]])

# ArrayLike[_SCT], array index -> _ExtremaND[_SCT]
assert_type(
    extrema(f64_2d, label_1d, index_1d),
    tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], list[tuple[np.intp, ...]], list[tuple[np.intp, ...]]],
)
assert_type(
    extrema(f32_2d, index=index_1d),
    tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], list[tuple[np.intp, ...]], list[tuple[np.intp, ...]]],
)

###
# minimum_position / maximum_position

assert_type(minimum_position(f64_2d), tuple[np.intp, ...])
assert_type(maximum_position(f64_2d), tuple[np.intp, ...])

assert_type(minimum_position(f64_2d, label_1d, index_1d), list[tuple[np.intp, ...]])
assert_type(maximum_position(f64_2d, label_1d, index_1d), list[tuple[np.intp, ...]])

###
# center_of_mass

# no index / scalar index -> _Coord0D
assert_type(center_of_mass(f64_2d), tuple[np.float64, ...])
assert_type(center_of_mass(f64_2d, label_1d, 1), tuple[np.float64, ...])

# list of int indices -> _Coord1D
assert_type(center_of_mass(f64_2d, label_1d, [1, 2, 3]), list[tuple[np.float64, ...]])
assert_type(center_of_mass(f64_2d, index=[1, 2, 3]), list[tuple[np.float64, ...]])

###
# histogram

# no index / scalar index -> ArrayND[np.intp]
assert_type(histogram(f64_2d, 0, 100, 10), onp.ArrayND[np.intp])
assert_type(histogram(f64_2d, 0, 100, 10, label_1d, 1), onp.ArrayND[np.intp])

# array index -> ArrayND[np.object_]
assert_type(histogram(f64_2d, 0, 100, 10, label_1d, index_1d), onp.ArrayND[np.object_])
assert_type(histogram(f64_2d, 0, 100, 10, label_1d, index=index_1d), onp.ArrayND[np.object_])

###
# watershed_ift

assert_type(watershed_ift(uint8_2d, int_2d), onp.ArrayND[npc.signedinteger])
