# type-tests for `ndimage/_morphology.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

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
# Test variables

f64_nd: onp.ArrayND[np.float64]
c128_nd: onp.ArrayND[np.complex128]
b_nd: onp.ArrayND[np.bool_]

int_nd: list[list[int]]
float_nd: list[list[float]]

###
# generate_binary_structure / iterate_structure

assert_type(generate_binary_structure(2, 1), onp.ArrayND[np.bool_])
assert_type(generate_binary_structure(3, 2), onp.ArrayND[np.bool_])

assert_type(iterate_structure(b_nd, 2), onp.ArrayND[np.bool_])

###
# binary_erosion / binary_dilation / binary_opening / binary_closing

assert_type(binary_erosion(f64_nd), onp.ArrayND[np.bool_])
assert_type(binary_erosion(c128_nd), onp.ArrayND[np.bool_])
assert_type(binary_erosion(int_nd), onp.ArrayND[np.bool_])
assert_type(binary_erosion(float_nd), onp.ArrayND[np.bool_])
assert_type(binary_erosion(f64_nd, structure=b_nd, iterations=2), onp.ArrayND[np.bool_])

assert_type(binary_dilation(f64_nd), onp.ArrayND[np.bool_])
assert_type(binary_dilation(c128_nd), onp.ArrayND[np.bool_])
assert_type(binary_dilation(int_nd), onp.ArrayND[np.bool_])
assert_type(binary_dilation(f64_nd, structure=b_nd, iterations=2), onp.ArrayND[np.bool_])

assert_type(binary_opening(f64_nd), onp.ArrayND[np.bool_])
assert_type(binary_opening(c128_nd), onp.ArrayND[np.bool_])
assert_type(binary_opening(int_nd), onp.ArrayND[np.bool_])

assert_type(binary_closing(f64_nd), onp.ArrayND[np.bool_])
assert_type(binary_closing(c128_nd), onp.ArrayND[np.bool_])
assert_type(binary_closing(int_nd), onp.ArrayND[np.bool_])

###
# binary_hit_or_miss / binary_propagation / binary_fill_holes

assert_type(binary_hit_or_miss(f64_nd), onp.ArrayND[np.bool_])
assert_type(binary_hit_or_miss(int_nd), onp.ArrayND[np.bool_])

assert_type(binary_propagation(f64_nd), onp.ArrayND[np.bool_])
assert_type(binary_propagation(int_nd), onp.ArrayND[np.bool_])

assert_type(binary_fill_holes(f64_nd), onp.ArrayND[np.bool_])
assert_type(binary_fill_holes(int_nd), onp.ArrayND[np.bool_])

###
# grey_erosion / grey_dilation / grey_opening / grey_closing

assert_type(grey_erosion(f64_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])
assert_type(grey_erosion(c128_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])
assert_type(grey_erosion(int_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])

assert_type(grey_dilation(f64_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])
assert_type(grey_dilation(c128_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])

assert_type(grey_opening(f64_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])
assert_type(grey_opening(c128_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])

assert_type(grey_closing(f64_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])
assert_type(grey_closing(c128_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])

###
# morphological_gradient / morphological_laplace

assert_type(morphological_gradient(f64_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])
assert_type(morphological_gradient(c128_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])

assert_type(morphological_laplace(f64_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])
assert_type(morphological_laplace(c128_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])

###
# white_tophat / black_tophat

assert_type(white_tophat(f64_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])
assert_type(white_tophat(c128_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])

assert_type(black_tophat(f64_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])
assert_type(black_tophat(c128_nd, size=(3, 3)), onp.ArrayND[npc.number | np.bool_])

###
# distance_transform_edt

assert_type(
    distance_transform_edt(f64_nd),
    onp.ArrayND[np.float64] | onp.ArrayND[np.int32] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]],
)
assert_type(
    distance_transform_edt(int_nd),
    onp.ArrayND[np.float64] | onp.ArrayND[np.int32] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]],
)
assert_type(
    distance_transform_edt(f64_nd, return_indices=True),
    onp.ArrayND[np.float64] | onp.ArrayND[np.int32] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]],
)

###
# distance_transform_bf

assert_type(
    distance_transform_bf(f64_nd),
    onp.ArrayND[npc.number | np.bool_] | onp.ArrayND[np.int32] | tuple[onp.ArrayND[npc.number | np.bool_], onp.ArrayND[np.int32]],
)

###
# distance_transform_cdt

assert_type(distance_transform_cdt(f64_nd), onp.ArrayND[np.int32] | tuple[onp.ArrayND[np.int32], onp.ArrayND[np.int32]])
