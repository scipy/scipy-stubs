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
f16_2d: onp.Array2D[np.float16]
f80_2d: onp.Array2D[np.longdouble]
c64_2d: onp.Array2D[np.complex64]
c128_2d: onp.Array2D[np.complex128]
c160_2d: onp.Array2D[np.clongdouble]
int_2d: onp.Array2D[np.int32]
uint8_2d: onp.Array2D[np.uint8]
bool_2d: onp.Array2D[np.bool_]

# python scalar sequences
py_float_1d: list[float]
py_int_2d: list[list[int]]
py_complex_1d: list[complex]

# label / index arrays for supervised statistics
label_1d: onp.Array1D[np.int32]
index_1d: onp.Array1D[np.intp]

# output array accepted by label(output=...)
out_labeled: onp.ArrayND[np.int32 | np.intp]

# helper function for labeled_comprehension
def _stat_func(x: onp.ToComplex | onp.ToComplexND) -> np.float64: ...
def _stat_func_with_positions(x: onp.ToComplex | onp.ToComplexND, positions: onp.ToComplex | onp.ToComplexND) -> np.float64: ...

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

assert_type(find_objects(1), list[tuple[()] | None])
assert_type(find_objects(int_2d), list[tuple[slice[int, int, None], ...] | None])

###
# value_indices

assert_type(value_indices(int_2d), dict[np.intp, tuple[onp.ArrayND[np.intp], ...]])

###
# labeled_comprehension

# index=None -> the return type of `func`, as-is; `out_dtype` and `default` are ignored at runtime
assert_type(labeled_comprehension(f64_2d, None, None, _stat_func, np.dtype(np.float32), 0.0), np.float64)
assert_type(labeled_comprehension(f64_2d, None, None, _stat_func, np.intp, 0), np.float64)
assert_type(labeled_comprehension(f64_2d, None, None, _stat_func, None, 0.0), np.float64)
assert_type(labeled_comprehension(f64_2d, None, None, _stat_func, np.float64, 0.0), np.float64)
assert_type(labeled_comprehension(f64_2d, None, None, _stat_func, np.complex128, 0.0), np.float64)
assert_type(
    labeled_comprehension(f64_2d, None, None, _stat_func_with_positions, np.float64, 0.0, pass_positions=True), np.float64
)
# labels may also be given with index=None
assert_type(labeled_comprehension(f64_2d, label_1d, None, _stat_func, np.complex128, 0.0), np.float64)

# out_dtype: DTypeLike[_SCT] -> _SCT
assert_type(labeled_comprehension(f64_2d, label_1d, 1, _stat_func, np.dtype(np.float32), 0.0), np.float32)
# out_dtype: AnyIntPDType -> np.intp
assert_type(labeled_comprehension(f64_2d, label_1d, 1, _stat_func, np.intp, 0), np.intp)
# out_dtype: AnyFloat64DType | None -> np.float64
assert_type(labeled_comprehension(f64_2d, label_1d, 1, _stat_func, None, 0.0), np.float64)
assert_type(labeled_comprehension(f64_2d, label_1d, 1, _stat_func, np.float64, 0.0), np.float64)
# out_dtype: AnyComplex128DType -> np.complex128
assert_type(labeled_comprehension(f64_2d, label_1d, 1, _stat_func, np.complex128, 0.0), np.complex128)

# index=<int array> -> ArrayND output

# out_dtype: DTypeLike[_SCT] -> ArrayND[_SCT]
assert_type(labeled_comprehension(f64_2d, label_1d, index_1d, _stat_func, np.dtype(np.float32), 0.0), onp.ArrayND[np.float32])
# out_dtype: AnyIntPDType -> ArrayND[np.intp]
assert_type(labeled_comprehension(f64_2d, label_1d, index_1d, _stat_func, np.intp, 0), onp.ArrayND[np.intp])
# out_dtype: AnyFloat64DType | None -> ArrayND[np.float64]
assert_type(labeled_comprehension(f64_2d, label_1d, index_1d, _stat_func, None, 0.0), onp.ArrayND[np.float64])
assert_type(labeled_comprehension(f64_2d, label_1d, index_1d, _stat_func, np.float64, 0.0), onp.ArrayND[np.float64])
# out_dtype: AnyComplex128DType -> ArrayND[np.complex128]
assert_type(labeled_comprehension(f64_2d, label_1d, index_1d, _stat_func, np.complex128, 0.0), onp.ArrayND[np.complex128])

# labels=None with a non-None index raises ValueError at runtime
labeled_comprehension(f64_2d, None, 1, _stat_func, np.float64, 0.0)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]  # pyrefly: ignore[no-matching-overload]
labeled_comprehension(f64_2d, None, index_1d, _stat_func, np.float64, 0.0)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]  # pyrefly: ignore[no-matching-overload]

###
# sum / sum_labels / mean / variance / standard_deviation / median

# scalar return (no index); mean / variance / standard_deviation upcast to at least float64
assert_type(mean(f64_2d), np.float64)
assert_type(mean(f32_2d), np.float64)
assert_type(mean(f16_2d), np.float64)
assert_type(mean(f80_2d), np.longdouble)
assert_type(mean(int_2d), np.float64)
assert_type(mean(bool_2d), np.float64)
assert_type(mean(c64_2d), np.complex128)
assert_type(mean(c128_2d), np.complex128)
assert_type(mean(c160_2d), np.clongdouble)
assert_type(mean(py_float_1d), np.float64)
assert_type(mean(py_int_2d), np.float64)
assert_type(mean(py_complex_1d), np.complex128)
assert_type(variance(f64_2d), np.float64)
assert_type(variance(f32_2d), np.float64)
assert_type(variance(f80_2d), np.longdouble)
assert_type(variance(int_2d), np.float64)
assert_type(variance(c64_2d), np.complex128)
assert_type(variance(c128_2d), np.complex128)
assert_type(variance(c160_2d), np.clongdouble)
assert_type(standard_deviation(f64_2d), np.float64)
assert_type(standard_deviation(f32_2d), np.float64)
assert_type(standard_deviation(f80_2d), np.longdouble)
assert_type(standard_deviation(int_2d), np.float64)
assert_type(standard_deviation(c64_2d), np.complex128)
assert_type(standard_deviation(c128_2d), np.complex128)
assert_type(standard_deviation(c160_2d), np.clongdouble)

# scalar return (no index); sum / sum_labels / median preserve inexact dtypes
assert_type(sum(f64_2d), np.float64)
assert_type(sum(f32_2d), np.float32)
assert_type(sum(f80_2d), np.longdouble)
assert_type(sum(c64_2d), np.complex64)
assert_type(sum(py_float_1d), np.float64)
assert_type(sum(py_complex_1d), np.complex128)
assert_type(sum_labels(f64_2d), np.float64)
assert_type(sum_labels(f32_2d), np.float32)
assert_type(median(f64_2d), np.float64)
assert_type(median(f32_2d), np.float32)
assert_type(median(py_float_1d), np.float64)

# scalar return (no index); sum / sum_labels promote integers to int64 or uint64, median to float64
assert_type(sum(int_2d), np.int64)
assert_type(sum(uint8_2d), np.uint64)
assert_type(sum(bool_2d), np.int64)
assert_type(sum(py_int_2d), np.int64)
assert_type(sum_labels(int_2d), np.int64)
assert_type(sum_labels(uint8_2d), np.uint64)
assert_type(median(int_2d), np.float64)
assert_type(median(uint8_2d), np.float64)

# labels without index -> still a scalar
assert_type(mean(f32_2d, label_1d), np.float64)
assert_type(variance(c64_2d, label_1d), np.complex128)
assert_type(standard_deviation(int_2d, label_1d), np.float64)
assert_type(sum(int_2d, label_1d), np.int64)
assert_type(sum(uint8_2d, label_1d), np.uint64)

# array return (array index)
assert_type(mean(f64_2d, label_1d, index_1d), onp.ArrayND[np.float64])
assert_type(mean(f32_2d, label_1d, index_1d), onp.ArrayND[np.float32])
assert_type(mean(int_2d, label_1d, index_1d), onp.ArrayND[np.float64])
assert_type(mean(f64_2d, index=index_1d), onp.ArrayND[np.float64])
assert_type(variance(f32_2d, label_1d, index_1d), onp.ArrayND[np.float32])
assert_type(variance(int_2d, label_1d, index_1d), onp.ArrayND[np.float64])
assert_type(variance(int_2d, index=index_1d), onp.ArrayND[np.float64])
assert_type(standard_deviation(f32_2d, label_1d, index_1d), onp.ArrayND[np.float32])
assert_type(standard_deviation(int_2d, label_1d, index_1d), onp.ArrayND[np.float64])
assert_type(sum(f64_2d, label_1d, index_1d), onp.ArrayND[np.float64])
assert_type(sum(int_2d, label_1d, index_1d), onp.ArrayND[np.float64])
assert_type(sum(uint8_2d, label_1d, index_1d), onp.ArrayND[np.float64])
assert_type(sum_labels(int_2d, label_1d, index_1d), onp.ArrayND[np.float64])
assert_type(median(f32_2d, label_1d, index_1d), onp.ArrayND[np.float32])
assert_type(median(int_2d, label_1d, index_1d), onp.ArrayND[np.float64])

# array return (scalar or sequence index)
assert_type(mean(f64_2d, label_1d, [1, 2]), onp.ArrayND[np.float64])
assert_type(mean(int_2d, label_1d, 1), onp.ArrayND[np.float64])
assert_type(mean(int_2d, label_1d, [1, 2]), onp.ArrayND[np.float64])
assert_type(mean(py_complex_1d, [1, 2], [1, 2]), onp.ArrayND[np.complex128])
assert_type(variance(int_2d, label_1d, 1), onp.ArrayND[np.float64])
assert_type(variance(f64_2d, label_1d, [1, 2]), onp.ArrayND[np.float64])
assert_type(standard_deviation(int_2d, label_1d, [1, 2]), onp.ArrayND[np.float64])
assert_type(standard_deviation(py_float_1d, [1, 2], 1), onp.ArrayND[np.float64])

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
