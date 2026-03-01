from typing import assert_type

from scipy.sparse import csc_array, csr_array
from scipy.sparse.linalg import expm, inv, matrix_power

a_csr: csr_array
a_csc: csc_array

# inv
assert_type(inv(a_csr), csr_array)
assert_type(inv(a_csc), csc_array)

# expm
assert_type(expm(a_csr), csr_array)
assert_type(expm(a_csc), csc_array)

# matrix_power
assert_type(matrix_power(a_csr, 3), csr_array)
assert_type(matrix_power(a_csc, 3), csc_array)
