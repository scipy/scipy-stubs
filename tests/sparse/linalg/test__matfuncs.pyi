from typing import assert_type

import numpy as np

from scipy.sparse import (
    bsr_array,
    bsr_matrix,
    coo_array,
    coo_matrix,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
    dia_array,
    dia_matrix,
    dok_array,
    dok_matrix,
    lil_array,
    lil_matrix,
)
from scipy.sparse.linalg import expm, inv, matrix_power

###

_a_bsr_i16: bsr_array[np.int16]
_a_csc_i16: csc_array[np.int16]
_a_csr_i16: csr_array[np.int16]
_a_coo_i16: coo_array[np.int16]
_a_dia_i16: dia_array[np.int16]
_a_dok_i16: dok_array[np.int16]
_a_lil_i16: lil_array[np.int16]

_a_bsr_f32: bsr_array[np.float32]
_a_csc_f32: csc_array[np.float32]
_a_csr_f32: csr_array[np.float32]
_a_coo_f32: coo_array[np.float32]
_a_dia_f32: dia_array[np.float32]
_a_dok_f32: dok_array[np.float32]
_a_lil_f32: lil_array[np.float32]

_m_bsr_i16: bsr_matrix[np.int16]
_m_csc_i16: csc_matrix[np.int16]
_m_csr_i16: csr_matrix[np.int16]
_m_coo_i16: coo_matrix[np.int16]
_m_dia_i16: dia_matrix[np.int16]
_m_dok_i16: dok_matrix[np.int16]
_m_lil_i16: lil_matrix[np.int16]

_m_bsr_f32: bsr_matrix[np.float32]
_m_csc_f32: csc_matrix[np.float32]
_m_csr_f32: csr_matrix[np.float32]
_m_coo_f32: coo_matrix[np.float32]
_m_dia_f32: dia_matrix[np.float32]
_m_dok_f32: dok_matrix[np.float32]
_m_lil_f32: lil_matrix[np.float32]

###
# inv

assert_type(inv(_a_bsr_i16), csc_array[np.float32])
assert_type(inv(_a_csc_i16), csc_array[np.float32])
assert_type(inv(_a_csr_i16), csr_array[np.float32])
assert_type(inv(_a_coo_i16), csc_array[np.float32])
assert_type(inv(_a_dia_i16), csc_array[np.float32])
assert_type(inv(_a_dok_i16), csc_array[np.float32])
assert_type(inv(_a_lil_i16), csc_array[np.float32])

assert_type(inv(_a_bsr_f32), csc_array[np.float32])
assert_type(inv(_a_csc_f32), csc_array[np.float32])
assert_type(inv(_a_csr_f32), csr_array[np.float32])
assert_type(inv(_a_coo_f32), csc_array[np.float32])
assert_type(inv(_a_dia_f32), csc_array[np.float32])
assert_type(inv(_a_dok_f32), csc_array[np.float32])
assert_type(inv(_a_lil_f32), csc_array[np.float32])

assert_type(inv(_m_bsr_i16), csc_array[np.float32])
assert_type(inv(_m_csc_i16), csc_matrix[np.float32])
assert_type(inv(_m_csr_i16), csr_matrix[np.float32])
assert_type(inv(_m_coo_i16), csc_array[np.float32])
assert_type(inv(_m_dia_i16), csc_array[np.float32])
assert_type(inv(_m_dok_i16), csc_array[np.float32])
assert_type(inv(_m_lil_i16), csc_array[np.float32])

assert_type(inv(_m_bsr_f32), csc_array[np.float32])
assert_type(inv(_m_csc_f32), csc_matrix[np.float32])
assert_type(inv(_m_csr_f32), csr_matrix[np.float32])
assert_type(inv(_m_coo_f32), csc_array[np.float32])
assert_type(inv(_m_dia_f32), csc_array[np.float32])
assert_type(inv(_m_dok_f32), csc_array[np.float32])
assert_type(inv(_m_lil_f32), csc_array[np.float32])

###
# expm

assert_type(expm(_a_bsr_i16), csc_array[np.float64])
assert_type(expm(_a_csc_i16), csc_array[np.float64])
assert_type(expm(_a_csr_i16), csr_array[np.float64])
assert_type(expm(_a_coo_i16), csr_array[np.float64])
assert_type(expm(_a_dia_i16), csc_array[np.float64])
assert_type(expm(_a_dok_i16), csr_array[np.float64])
assert_type(expm(_a_lil_i16), csr_array[np.float64])

assert_type(expm(_a_bsr_f32), csc_array[np.float32])
assert_type(expm(_a_csc_f32), csc_array[np.float32])
assert_type(expm(_a_csr_f32), csr_array[np.float32])
assert_type(expm(_a_coo_f32), csr_array[np.float32])
assert_type(expm(_a_dia_f32), csc_array[np.float32])
assert_type(expm(_a_dok_f32), csr_array[np.float32])
assert_type(expm(_a_lil_f32), csr_array[np.float32])

assert_type(expm(_m_bsr_i16), csc_array[np.float64])
assert_type(expm(_m_csc_i16), csc_matrix[np.float64])
assert_type(expm(_m_csr_i16), csr_matrix[np.float64])
assert_type(expm(_m_coo_i16), csr_matrix[np.float64])
assert_type(expm(_m_dia_i16), csc_array[np.float64])
assert_type(expm(_m_dok_i16), csr_matrix[np.float64])
assert_type(expm(_m_lil_i16), csr_matrix[np.float64])

assert_type(expm(_m_bsr_f32), csc_array[np.float32])
assert_type(expm(_m_csc_f32), csc_matrix[np.float32])
assert_type(expm(_m_csr_f32), csr_matrix[np.float32])
assert_type(expm(_m_coo_f32), csr_matrix[np.float32])
assert_type(expm(_m_dia_f32), csc_array[np.float32])
assert_type(expm(_m_dok_f32), csr_matrix[np.float32])
assert_type(expm(_m_lil_f32), csr_matrix[np.float32])

###
# matrix_power

assert_type(matrix_power(_a_bsr_i16, 3), bsr_array[np.int16])
assert_type(matrix_power(_a_csc_i16, 3), csc_array[np.int16])
assert_type(matrix_power(_a_csr_i16, 3), csr_array[np.int16])
assert_type(matrix_power(_a_coo_i16, 3), csr_array[np.int16])
assert_type(matrix_power(_a_dia_i16, 3), dia_array[np.int16])
assert_type(matrix_power(_a_dok_i16, 3), csr_array[np.int16])
assert_type(matrix_power(_a_lil_i16, 3), csr_array[np.int16])

assert_type(matrix_power(_a_bsr_f32, 3), bsr_array[np.float32])
assert_type(matrix_power(_a_csc_f32, 3), csc_array[np.float32])
assert_type(matrix_power(_a_csr_f32, 3), csr_array[np.float32])
assert_type(matrix_power(_a_coo_f32, 3), csr_array[np.float32])
assert_type(matrix_power(_a_dia_f32, 3), dia_array[np.float32])
assert_type(matrix_power(_a_dok_f32, 3), csr_array[np.float32])
assert_type(matrix_power(_a_lil_f32, 3), csr_array[np.float32])

assert_type(matrix_power(_m_bsr_i16, 3), bsr_matrix[np.int16])
assert_type(matrix_power(_m_csc_i16, 3), csc_matrix[np.int16])
assert_type(matrix_power(_m_csr_i16, 3), csr_matrix[np.int16])
assert_type(matrix_power(_m_coo_i16, 3), csr_matrix[np.int16])
assert_type(matrix_power(_m_dia_i16, 3), dia_matrix[np.int16])
assert_type(matrix_power(_m_dok_i16, 3), csr_matrix[np.int16])
assert_type(matrix_power(_m_lil_i16, 3), csr_matrix[np.int16])

assert_type(matrix_power(_m_bsr_f32, 3), bsr_matrix[np.float32])
assert_type(matrix_power(_m_csc_f32, 3), csc_matrix[np.float32])
assert_type(matrix_power(_m_csr_f32, 3), csr_matrix[np.float32])
assert_type(matrix_power(_m_coo_f32, 3), csr_matrix[np.float32])
assert_type(matrix_power(_m_dia_f32, 3), dia_matrix[np.float32])
assert_type(matrix_power(_m_dok_f32, 3), csr_matrix[np.float32])
assert_type(matrix_power(_m_lil_f32, 3), csr_matrix[np.float32])
