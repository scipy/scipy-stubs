# type-tests for `ndimage/_measurements.pyi`

from typing import Any, assert_type

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

_bool_2d: onp.Array2D[np.bool]
_u8_2d: onp.Array2D[np.uint8]
_i32_1d: onp.Array1D[np.int32]
_i32_2d: onp.Array2D[np.int32]
_intp_1d: onp.Array1D[np.intp]
_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_f80_2d: onp.Array2D[np.float128]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]
_c160_2d: onp.Array2D[np.complex256]

_py_i_2d: list[list[int]]
_py_f_1d: list[float]
_py_c_1d: list[complex]

_i32_i64_nd: onp.ArrayND[np.int32 | np.int64]

# helper function for labeled_comprehension
def _stat_func(x: onp.ToComplex | onp.ToComplexND) -> np.float64: ...
def _stat_func_with_positions(x: onp.ToComplex | onp.ToComplexND, positions: onp.ToComplex | onp.ToComplexND) -> np.float64: ...

###
# label

# output array provided -> returns feature count
assert_type(label(_f64_2d, output=_i32_i64_nd), int)
assert_type(label(_i32_2d, output=_i32_i64_nd), int)

# no output -> (labeled_array, num_features)
assert_type(label(_f64_2d), tuple[onp.ArrayND[np.int32 | np.intp], int])
assert_type(label(_i32_2d), tuple[onp.ArrayND[np.int32 | np.intp], int])

###
# find_objects

assert_type(find_objects(1), list[tuple[()] | None])
assert_type(find_objects(_i32_2d), list[tuple[slice[int, int, None], ...] | None])

###
# value_indices

assert_type(value_indices(_i32_2d), dict[np.intp, tuple[onp.ArrayND[np.intp], ...]])

###
# labeled_comprehension

# index=None -> the return type of `func`, as-is; `out_dtype` and `default` are ignored at runtime
assert_type(labeled_comprehension(_f64_2d, None, None, _stat_func, np.dtype(np.float32), 0.0), np.float64)
assert_type(labeled_comprehension(_f64_2d, None, None, _stat_func, np.intp, 0), np.float64)
assert_type(labeled_comprehension(_f64_2d, None, None, _stat_func, None, 0.0), np.float64)
assert_type(labeled_comprehension(_f64_2d, None, None, _stat_func, np.float64, 0.0), np.float64)
assert_type(labeled_comprehension(_f64_2d, None, None, _stat_func, np.complex128, 0.0), np.float64)
assert_type(
    labeled_comprehension(_f64_2d, None, None, _stat_func_with_positions, np.float64, 0.0, pass_positions=True), np.float64
)
# labels may also be given with index=None
assert_type(labeled_comprehension(_f64_2d, _i32_1d, None, _stat_func, np.complex128, 0.0), np.float64)

# out_dtype: DTypeLike[_SCT] -> _SCT
assert_type(labeled_comprehension(_f64_2d, _i32_1d, 1, _stat_func, np.dtype(np.float32), 0.0), np.float32)
# out_dtype: AnyIntPDType -> np.intp
assert_type(labeled_comprehension(_f64_2d, _i32_1d, 1, _stat_func, np.intp, 0), np.intp)
# out_dtype: AnyFloat64DType | None -> np.float64
assert_type(labeled_comprehension(_f64_2d, _i32_1d, 1, _stat_func, None, 0.0), np.float64)
assert_type(labeled_comprehension(_f64_2d, _i32_1d, 1, _stat_func, np.float64, 0.0), np.float64)
# out_dtype: AnyComplex128DType -> np.complex128
assert_type(labeled_comprehension(_f64_2d, _i32_1d, 1, _stat_func, np.complex128, 0.0), np.complex128)

# index=<int array> -> ArrayND output

# out_dtype: DTypeLike[_SCT] -> ArrayND[_SCT]
assert_type(labeled_comprehension(_f64_2d, _i32_1d, _intp_1d, _stat_func, np.dtype(np.float32), 0.0), onp.ArrayND[np.float32])
# out_dtype: AnyIntPDType -> ArrayND[np.intp]
assert_type(labeled_comprehension(_f64_2d, _i32_1d, _intp_1d, _stat_func, np.intp, 0), onp.ArrayND[np.intp])
# out_dtype: AnyFloat64DType | None -> ArrayND[np.float64]
assert_type(labeled_comprehension(_f64_2d, _i32_1d, _intp_1d, _stat_func, None, 0.0), onp.ArrayND[np.float64])
assert_type(labeled_comprehension(_f64_2d, _i32_1d, _intp_1d, _stat_func, np.float64, 0.0), onp.ArrayND[np.float64])
# out_dtype: AnyComplex128DType -> ArrayND[np.complex128]
assert_type(labeled_comprehension(_f64_2d, _i32_1d, _intp_1d, _stat_func, np.complex128, 0.0), onp.ArrayND[np.complex128])

# labels=None with a non-None index raises ValueError at runtime
labeled_comprehension(_f64_2d, None, 1, _stat_func, np.float64, 0.0)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]  # pyrefly: ignore[no-matching-overload]
labeled_comprehension(_f64_2d, None, _intp_1d, _stat_func, np.float64, 0.0)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]  # pyrefly: ignore[no-matching-overload]

###
# sum / sum_labels / mean / variance / standard_deviation / median

# scalar return (no index); mean / variance / standard_deviation upcast to at least float64
assert_type(mean(_f64_2d), np.float64)
assert_type(mean(_f32_2d), np.float64)
assert_type(mean(_f16_2d), np.float64)
assert_type(mean(_f80_2d), np.float128)
assert_type(mean(_i32_2d), np.float64)
assert_type(mean(_bool_2d), np.float64)
assert_type(mean(_c64_2d), np.complex128)
assert_type(mean(_c128_2d), np.complex128)
assert_type(mean(_c160_2d), np.complex256)
assert_type(mean(_py_f_1d), np.float64)
assert_type(mean(_py_i_2d), np.float64)
assert_type(mean(_py_c_1d), np.complex128)
assert_type(variance(_f64_2d), np.float64)
assert_type(variance(_f32_2d), np.float64)
assert_type(variance(_f80_2d), np.float128)
assert_type(variance(_i32_2d), np.float64)
assert_type(variance(_c64_2d), np.complex128)
assert_type(variance(_c128_2d), np.complex128)
assert_type(variance(_c160_2d), np.complex256)
assert_type(standard_deviation(_f64_2d), np.float64)
assert_type(standard_deviation(_f32_2d), np.float64)
assert_type(standard_deviation(_f80_2d), np.float128)
assert_type(standard_deviation(_i32_2d), np.float64)
assert_type(standard_deviation(_c64_2d), np.complex128)
assert_type(standard_deviation(_c128_2d), np.complex128)
assert_type(standard_deviation(_c160_2d), np.complex256)

# scalar return (no index); sum / sum_labels / median preserve inexact dtypes
assert_type(sum(_f64_2d), np.float64)
assert_type(sum(_f32_2d), np.float32)
assert_type(sum(_f80_2d), np.float128)
assert_type(sum(_c64_2d), np.complex64)
assert_type(sum(_py_f_1d), np.float64)
assert_type(sum(_py_c_1d), np.complex128)
assert_type(sum_labels(_f64_2d), np.float64)
assert_type(sum_labels(_f32_2d), np.float32)
assert_type(median(_f64_2d), np.float64)
assert_type(median(_f32_2d), np.float32)
assert_type(median(_py_f_1d), np.float64)

# scalar return (no index); sum / sum_labels promote integers to int64 or uint64, median to float64
assert_type(sum(_i32_2d), np.int64)
assert_type(sum(_u8_2d), np.uint64)
assert_type(sum(_bool_2d), np.int64)
assert_type(sum(_py_i_2d), np.int64)
assert_type(sum_labels(_i32_2d), np.int64)
assert_type(sum_labels(_u8_2d), np.uint64)
assert_type(median(_i32_2d), np.float64)
assert_type(median(_u8_2d), np.float64)

# labels without index -> still a scalar
assert_type(mean(_f32_2d, _i32_1d), np.float64)
assert_type(variance(_c64_2d, _i32_1d), np.complex128)
assert_type(standard_deviation(_i32_2d, _i32_1d), np.float64)
assert_type(sum(_i32_2d, _i32_1d), np.int64)
assert_type(sum(_u8_2d, _i32_1d), np.uint64)

# array return (labels with array index); everything except median goes through bincount and upcasts to float64
assert_type(mean(_f64_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(mean(_f32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(mean(_i32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(mean(_f64_2d, _i32_1d, [1, 2]), onp.ArrayND[np.float64])
assert_type(mean(_i32_2d, _i32_1d, [1, 2]), onp.ArrayND[np.float64])
assert_type(variance(_f32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(variance(_i32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(variance(_f64_2d, _i32_1d, [1, 2]), onp.ArrayND[np.float64])
assert_type(standard_deviation(_f32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(standard_deviation(_i32_2d, _i32_1d, [1, 2]), onp.ArrayND[np.float64])
assert_type(sum(_f64_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(sum(_f32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(sum(_i32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(sum(_u8_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(sum_labels(_i32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(sum_labels(_f32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])

# array return (labels with array index); median preserves inexact dtypes
assert_type(median(_f32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float32])
assert_type(median(_c64_2d, _i32_1d, _intp_1d), onp.ArrayND[np.complex64])
assert_type(median(_i32_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])

# scalar return (labels with scalar index)
assert_type(sum(_f32_2d, _i32_1d, 1), np.float32)
assert_type(sum(_i32_2d, _i32_1d, 1), np.int64)
assert_type(sum(_u8_2d, _i32_1d, 1), np.uint64)
assert_type(mean(_f32_2d, _i32_1d, 1), np.float64)
assert_type(mean(_i32_2d, _i32_1d, 1), np.float64)
assert_type(mean(_c64_2d, _i32_1d, 1), np.complex128)
assert_type(mean(_f80_2d, _i32_1d, 1), np.float128)
assert_type(variance(_i32_2d, _i32_1d, 1), np.float64)
assert_type(standard_deviation(_py_f_1d, [1, 2], 1), np.float64)
assert_type(median(_f32_2d, _i32_1d, 1), np.float32)

# scalar return (no labels); index is ignored when labels is None
assert_type(mean(_f64_2d, index=_intp_1d), np.float64)
assert_type(variance(_i32_2d, index=_intp_1d), np.float64)
assert_type(sum(_f32_2d, index=_intp_1d), np.float32)
assert_type(sum(_f32_2d, None, _intp_1d), np.float32)
assert_type(median(_f32_2d, None, _intp_1d), np.float32)

# complex or longdouble input with labels and an array index raises a TypeError at runtime
assert_type(mean(_c128_2d, _i32_1d, _intp_1d), onp.ArrayND[Any])
assert_type(mean(_py_c_1d, [1, 2], [1, 2]), onp.ArrayND[Any])
assert_type(sum(_f80_2d, _i32_1d, _intp_1d), onp.ArrayND[Any])

###
# minimum / maximum

assert_type(minimum(_f64_2d), np.float64)
assert_type(minimum(_f32_2d), np.float32)
assert_type(maximum(_f64_2d), np.float64)

assert_type(minimum(_f64_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])
assert_type(maximum(_f64_2d, _i32_1d, _intp_1d), onp.ArrayND[np.float64])

###
# extrema

# ArrayLike[_SCT], no index -> _Extrema0D[_SCT]
assert_type(extrema(_f64_2d), tuple[np.float64, np.float64, tuple[np.intp, ...], tuple[np.intp, ...]])
assert_type(extrema(_f32_2d), tuple[np.float32, np.float32, tuple[np.intp, ...], tuple[np.intp, ...]])

# ArrayLike[_SCT], array index -> _ExtremaND[_SCT]
assert_type(
    extrema(_f64_2d, _i32_1d, _intp_1d),
    tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], list[tuple[np.intp, ...]], list[tuple[np.intp, ...]]],
)
assert_type(
    extrema(_f32_2d, index=_intp_1d),
    tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32], list[tuple[np.intp, ...]], list[tuple[np.intp, ...]]],
)

###
# minimum_position / maximum_position

assert_type(minimum_position(_f64_2d), tuple[np.intp, ...])
assert_type(maximum_position(_f64_2d), tuple[np.intp, ...])

assert_type(minimum_position(_f64_2d, _i32_1d, _intp_1d), list[tuple[np.intp, ...]])
assert_type(maximum_position(_f64_2d, _i32_1d, _intp_1d), list[tuple[np.intp, ...]])

###
# center_of_mass

# no index / scalar index -> _Coord0D
assert_type(center_of_mass(_f64_2d), tuple[np.float64, ...])
assert_type(center_of_mass(_f64_2d, _i32_1d, 1), tuple[np.float64, ...])

# list of int indices -> _Coord1D
assert_type(center_of_mass(_f64_2d, _i32_1d, [1, 2, 3]), list[tuple[np.float64, ...]])
assert_type(center_of_mass(_f64_2d, index=[1, 2, 3]), list[tuple[np.float64, ...]])

###
# histogram

# no index / scalar index -> ArrayND[np.intp]
assert_type(histogram(_f64_2d, 0, 100, 10), onp.ArrayND[np.intp])
assert_type(histogram(_f64_2d, 0, 100, 10, _i32_1d, 1), onp.ArrayND[np.intp])

# array index -> ArrayND[np.object_]
assert_type(histogram(_f64_2d, 0, 100, 10, _i32_1d, _intp_1d), onp.ArrayND[np.object_])
assert_type(histogram(_f64_2d, 0, 100, 10, _i32_1d, index=_intp_1d), onp.ArrayND[np.object_])

###
# watershed_ift

assert_type(watershed_ift(_u8_2d, _i32_2d), onp.ArrayND[npc.signedinteger])
