from typing import TypeAlias

import numpy as np

from scipy import sparse

ScalarType: TypeAlias = np.float32

bsr_mat: sparse.bsr_matrix[ScalarType]
coo_mat: sparse.coo_matrix[ScalarType]
csc_mat: sparse.csc_matrix[ScalarType]
csr_mat: sparse.csr_matrix[ScalarType]
dia_mat: sparse.dia_matrix[ScalarType]
dok_mat: sparse.dok_matrix[ScalarType]
lil_mat: sparse.lil_matrix[ScalarType]
any_mat: (
    sparse.bsr_matrix[ScalarType]
    | sparse.coo_matrix[ScalarType]
    | sparse.csc_matrix[ScalarType]
    | sparse.csr_matrix[ScalarType]
    | sparse.dia_matrix[ScalarType]
    | sparse.dok_matrix[ScalarType]
    | sparse.lil_matrix[ScalarType]
)

bsr_arr: sparse.bsr_array[ScalarType]
coo_arr: sparse.coo_array[ScalarType, tuple[int, int]]
csc_arr: sparse.csc_array[ScalarType]
csr_arr: sparse.csr_array[ScalarType, tuple[int, int]]
dia_arr: sparse.dia_array[ScalarType]
dok_arr: sparse.dok_array[ScalarType, tuple[int, int]]
lil_arr: sparse.lil_array[ScalarType]
any_arr: (
    sparse.bsr_array[ScalarType]
    | sparse.coo_array[ScalarType, tuple[int, int]]
    | sparse.csc_array[ScalarType]
    | sparse.csr_array[ScalarType, tuple[int, int]]
    | sparse.dia_array[ScalarType]
    | sparse.dok_array[ScalarType, tuple[int, int]]
    | sparse.lil_array[ScalarType]
)
