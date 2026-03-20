# type-tests for `ndimage/_morphology.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_hit_or_miss,
    binary_opening,
    binary_propagation,
    black_tophat,
    distance_transform_bf,
    distance_transform_cdt,
    distance_transform_edt,
    generate_binary_structure,
    grey_closing,
    grey_dilation,
    grey_erosion,
    grey_opening,
    iterate_structure,
    morphological_gradient,
    morphological_laplace,
    white_tophat,
)

###

_b_1d: onp.Array1D[np.bool_]
_b_2d: onp.Array2D[np.bool_]
_b_nd: onp.ArrayND[np.bool_]

_u32_2d: onp.Array2D[np.uint32]
_i32_2d: onp.Array2D[np.int32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_py_b_2d: list[list[bool]]
_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]

_i32: np.int32

###

# iterate_structure
assert_type(iterate_structure(_b_1d, 1), onp.Array1D[np.bool_])
assert_type(iterate_structure(_b_2d, 1), onp.Array2D[np.bool_])
assert_type(iterate_structure(_b_nd, 1), onp.ArrayND[np.bool_])
assert_type(iterate_structure(_b_1d, 1, origin=0), tuple[onp.Array1D[np.bool_], list[int]])
assert_type(iterate_structure(_b_2d, 1, origin=0), tuple[onp.Array2D[np.bool_], list[int]])
assert_type(iterate_structure(_b_nd, 1, origin=0), tuple[onp.ArrayND[np.bool_], list[int]])
assert_type(iterate_structure(_b_nd, 1, origin=_i32), tuple[onp.ArrayND[np.bool_], list[np.int32]])

# generate_binary_structure
assert_type(generate_binary_structure(-3, 1), onp.Array0D[np.bool_])
assert_type(generate_binary_structure(-2, 1), onp.Array0D[np.bool_])
assert_type(generate_binary_structure(-1, 1), onp.Array0D[np.bool_])
assert_type(generate_binary_structure(0, 1), onp.Array0D[np.bool_])
assert_type(generate_binary_structure(1, 1), onp.Array1D[np.bool_])
assert_type(generate_binary_structure(2, 1), onp.Array2D[np.bool_])
assert_type(generate_binary_structure(3, 2), onp.Array3D[np.bool_])
assert_type(generate_binary_structure(42, 2), onp.ArrayND[np.bool_])

# binary_erosion
assert_type(binary_erosion(_py_i_2d), onp.ArrayND[np.bool_])
assert_type(binary_erosion(_py_f_2d), onp.ArrayND[np.bool_])
assert_type(binary_erosion(_f64_2d), onp.ArrayND[np.bool_])
assert_type(binary_erosion(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_erosion(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_erosion(_f64_2d, output=_f64_2d), onp.Array2D[np.float64])

# binary_dilation (same as above)
assert_type(binary_dilation(_py_i_2d), onp.ArrayND[np.bool_])
assert_type(binary_dilation(_py_f_2d), onp.ArrayND[np.bool_])
assert_type(binary_dilation(_f64_2d), onp.ArrayND[np.bool_])
assert_type(binary_dilation(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_dilation(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_dilation(_f64_2d, output=_f64_2d), onp.Array2D[np.float64])

# binary_opening (same as above)
assert_type(binary_opening(_py_i_2d), onp.ArrayND[np.bool_])
assert_type(binary_opening(_py_f_2d), onp.ArrayND[np.bool_])
assert_type(binary_opening(_f64_2d), onp.ArrayND[np.bool_])
assert_type(binary_opening(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_opening(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_opening(_f64_2d, output=_f64_2d), onp.Array2D[np.float64])

# binary_closing (same as above)
assert_type(binary_closing(_py_i_2d), onp.ArrayND[np.bool_])
assert_type(binary_closing(_py_f_2d), onp.ArrayND[np.bool_])
assert_type(binary_closing(_f64_2d), onp.ArrayND[np.bool_])
assert_type(binary_closing(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_closing(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_closing(_f64_2d, output=_f64_2d), onp.Array2D[np.float64])

# binary_hit_or_miss (same as above)
assert_type(binary_hit_or_miss(_py_i_2d), onp.ArrayND[np.bool_])
assert_type(binary_hit_or_miss(_py_f_2d), onp.ArrayND[np.bool_])
assert_type(binary_hit_or_miss(_f64_2d), onp.ArrayND[np.bool_])
assert_type(binary_hit_or_miss(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_hit_or_miss(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_hit_or_miss(_f64_2d, output=_f64_2d), onp.Array2D[np.float64])

# binary_propagation (same as above)
assert_type(binary_propagation(_py_i_2d), onp.ArrayND[np.bool_])
assert_type(binary_propagation(_py_f_2d), onp.ArrayND[np.bool_])
assert_type(binary_propagation(_f64_2d), onp.ArrayND[np.bool_])
assert_type(binary_propagation(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_propagation(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_propagation(_f64_2d, output=_f64_2d), onp.Array2D[np.float64])

# binary_fill_holes (same as above)
assert_type(binary_fill_holes(_py_i_2d), onp.ArrayND[np.bool_])
assert_type(binary_fill_holes(_py_f_2d), onp.ArrayND[np.bool_])
assert_type(binary_fill_holes(_f64_2d), onp.ArrayND[np.bool_])
assert_type(binary_fill_holes(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_fill_holes(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(binary_fill_holes(_f64_2d, output=_f64_2d), onp.Array2D[np.float64])

# grey_erosion
assert_type(grey_erosion(_b_1d), onp.Array1D[np.bool_])
assert_type(grey_erosion(_b_2d), onp.Array2D[np.bool_])
assert_type(grey_erosion(_b_nd), onp.ArrayND[np.bool_])
assert_type(grey_erosion(_f64_1d), onp.Array1D[np.float64])
assert_type(grey_erosion(_f64_2d), onp.Array2D[np.float64])
assert_type(grey_erosion(_f64_nd), onp.ArrayND[np.float64])
assert_type(grey_erosion(_py_b_2d), onp.ArrayND[np.bool_])
assert_type(grey_erosion(_py_i_2d), onp.ArrayND[np.int_])
assert_type(grey_erosion(_py_f_2d), onp.ArrayND[np.float64])
assert_type(grey_erosion(_py_b_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(grey_erosion(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(grey_erosion(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])

# grey_dilation (same as above)
assert_type(grey_dilation(_b_1d), onp.Array1D[np.bool_])
assert_type(grey_dilation(_b_2d), onp.Array2D[np.bool_])
assert_type(grey_dilation(_b_nd), onp.ArrayND[np.bool_])
assert_type(grey_dilation(_f64_1d), onp.Array1D[np.float64])
assert_type(grey_dilation(_f64_2d), onp.Array2D[np.float64])
assert_type(grey_dilation(_f64_nd), onp.ArrayND[np.float64])
assert_type(grey_dilation(_py_b_2d), onp.ArrayND[np.bool_])
assert_type(grey_dilation(_py_i_2d), onp.ArrayND[np.int_])
assert_type(grey_dilation(_py_f_2d), onp.ArrayND[np.float64])
assert_type(grey_dilation(_py_b_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(grey_dilation(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(grey_dilation(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])

# grey_opening (same as above)
assert_type(grey_opening(_b_1d), onp.Array1D[np.bool_])
assert_type(grey_opening(_b_2d), onp.Array2D[np.bool_])
assert_type(grey_opening(_b_nd), onp.ArrayND[np.bool_])
assert_type(grey_opening(_f64_1d), onp.Array1D[np.float64])
assert_type(grey_opening(_f64_2d), onp.Array2D[np.float64])
assert_type(grey_opening(_f64_nd), onp.ArrayND[np.float64])
assert_type(grey_opening(_py_b_2d), onp.ArrayND[np.bool_])
assert_type(grey_opening(_py_i_2d), onp.ArrayND[np.int_])
assert_type(grey_opening(_py_f_2d), onp.ArrayND[np.float64])
assert_type(grey_opening(_py_b_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(grey_opening(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(grey_opening(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])

# grey_closing (same as above)
assert_type(grey_closing(_b_1d), onp.Array1D[np.bool_])
assert_type(grey_closing(_b_2d), onp.Array2D[np.bool_])
assert_type(grey_closing(_b_nd), onp.ArrayND[np.bool_])
assert_type(grey_closing(_f64_1d), onp.Array1D[np.float64])
assert_type(grey_closing(_f64_2d), onp.Array2D[np.float64])
assert_type(grey_closing(_f64_nd), onp.ArrayND[np.float64])
assert_type(grey_closing(_py_b_2d), onp.ArrayND[np.bool_])
assert_type(grey_closing(_py_i_2d), onp.ArrayND[np.int_])
assert_type(grey_closing(_py_f_2d), onp.ArrayND[np.float64])
assert_type(grey_closing(_py_b_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(grey_closing(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(grey_closing(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])

# morphological_gradient (same as above)
assert_type(morphological_gradient(_b_1d), onp.Array1D[np.bool_])
assert_type(morphological_gradient(_b_2d), onp.Array2D[np.bool_])
assert_type(morphological_gradient(_b_nd), onp.ArrayND[np.bool_])
assert_type(morphological_gradient(_f64_1d), onp.Array1D[np.float64])
assert_type(morphological_gradient(_f64_2d), onp.Array2D[np.float64])
assert_type(morphological_gradient(_f64_nd), onp.ArrayND[np.float64])
assert_type(morphological_gradient(_py_b_2d), onp.ArrayND[np.bool_])
assert_type(morphological_gradient(_py_i_2d), onp.ArrayND[np.int_])
assert_type(morphological_gradient(_py_f_2d), onp.ArrayND[np.float64])
assert_type(morphological_gradient(_py_b_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(morphological_gradient(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(morphological_gradient(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])

# morphological_laplace (same as above)
assert_type(morphological_laplace(_b_1d), onp.Array1D[np.bool_])
assert_type(morphological_laplace(_b_2d), onp.Array2D[np.bool_])
assert_type(morphological_laplace(_b_nd), onp.ArrayND[np.bool_])
assert_type(morphological_laplace(_f64_1d), onp.Array1D[np.float64])
assert_type(morphological_laplace(_f64_2d), onp.Array2D[np.float64])
assert_type(morphological_laplace(_f64_nd), onp.ArrayND[np.float64])
assert_type(morphological_laplace(_py_b_2d), onp.ArrayND[np.bool_])
assert_type(morphological_laplace(_py_i_2d), onp.ArrayND[np.int_])
assert_type(morphological_laplace(_py_f_2d), onp.ArrayND[np.float64])
assert_type(morphological_laplace(_py_b_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(morphological_laplace(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(morphological_laplace(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])

# white_tophat (same as above)
assert_type(white_tophat(_b_1d), onp.Array1D[np.bool_])
assert_type(white_tophat(_b_2d), onp.Array2D[np.bool_])
assert_type(white_tophat(_b_nd), onp.ArrayND[np.bool_])
assert_type(white_tophat(_f64_1d), onp.Array1D[np.float64])
assert_type(white_tophat(_f64_2d), onp.Array2D[np.float64])
assert_type(white_tophat(_f64_nd), onp.ArrayND[np.float64])
assert_type(white_tophat(_py_b_2d), onp.ArrayND[np.bool_])
assert_type(white_tophat(_py_i_2d), onp.ArrayND[np.int_])
assert_type(white_tophat(_py_f_2d), onp.ArrayND[np.float64])
assert_type(white_tophat(_py_b_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(white_tophat(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(white_tophat(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])

# black_tophat (same as above)
assert_type(black_tophat(_b_1d), onp.Array1D[np.bool_])
assert_type(black_tophat(_b_2d), onp.Array2D[np.bool_])
assert_type(black_tophat(_b_nd), onp.ArrayND[np.bool_])
assert_type(black_tophat(_f64_1d), onp.Array1D[np.float64])
assert_type(black_tophat(_f64_2d), onp.Array2D[np.float64])
assert_type(black_tophat(_f64_nd), onp.ArrayND[np.float64])
assert_type(black_tophat(_py_b_2d), onp.ArrayND[np.bool_])
assert_type(black_tophat(_py_i_2d), onp.ArrayND[np.int_])
assert_type(black_tophat(_py_f_2d), onp.ArrayND[np.float64])
assert_type(black_tophat(_py_b_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(black_tophat(_py_i_2d, output=_f64_2d), onp.Array2D[np.float64])
assert_type(black_tophat(_py_f_2d, output=_f64_2d), onp.Array2D[np.float64])

# distance_transform_bf
assert_type(distance_transform_bf(_f64_2d), onp.ArrayND[np.float64])
assert_type(distance_transform_bf(_f64_2d, return_distances=False), None)
assert_type(distance_transform_bf(_f64_2d, distances=_f64_2d), None)
assert_type(distance_transform_bf(_f64_2d, return_distances=False, return_indices=True), onp.ArrayND[np.int32])
assert_type(distance_transform_bf(_f64_2d, distances=_f64_2d, return_indices=True), onp.Array2D[np.int32])
assert_type(distance_transform_bf(_f64_2d, return_indices=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]])
assert_type(distance_transform_bf(_f64_2d, "taxicab"), onp.ArrayND[np.uint32])
assert_type(distance_transform_bf(_f64_2d, "taxicab", return_distances=False), None)
assert_type(distance_transform_bf(_f64_2d, "taxicab", distances=_u32_2d), None)
assert_type(distance_transform_bf(_f64_2d, "taxicab", return_distances=False, return_indices=True), onp.ArrayND[np.int32])
assert_type(distance_transform_bf(_f64_2d, "taxicab", distances=_u32_2d, return_indices=True), onp.Array2D[np.int32])
assert_type(distance_transform_bf(_f64_2d, "taxicab", return_indices=True), tuple[onp.ArrayND[np.uint32], onp.ArrayND[np.int32]])

# distance_transform_cdt
assert_type(distance_transform_cdt(_f64_2d), onp.ArrayND[np.int32])
assert_type(distance_transform_cdt(_f64_2d, return_distances=False), None)
assert_type(distance_transform_cdt(_f64_2d, distances=_i32_2d), None)
assert_type(distance_transform_cdt(_f64_2d, return_distances=False, return_indices=True), onp.ArrayND[np.int32])
assert_type(distance_transform_cdt(_f64_2d, distances=_i32_2d, return_indices=True), onp.Array2D[np.int32])
assert_type(distance_transform_cdt(_f64_2d, return_indices=True), tuple[onp.ArrayND[np.int32], onp.ArrayND[np.int32]])

# distance_transform_edt
assert_type(distance_transform_edt(_f64_2d), onp.ArrayND[np.float64])
assert_type(distance_transform_edt(_f64_2d, return_distances=False), None)
assert_type(distance_transform_edt(_f64_2d, distances=_f64_2d), None)
assert_type(distance_transform_edt(_f64_2d, return_distances=False, return_indices=True), onp.ArrayND[np.int32])
assert_type(distance_transform_edt(_f64_2d, distances=_f64_2d, return_indices=True), onp.Array2D[np.int32])
assert_type(distance_transform_edt(_f64_2d, return_indices=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]])
