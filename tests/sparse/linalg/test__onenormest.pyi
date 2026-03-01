from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import onenormest

a: csr_array[np.float64]

# onenormest
assert_type(onenormest(a), np.float64)
assert_type(onenormest(a, 2, 5, False, True), tuple[np.float64, onp.Array1D[np.float64]])
assert_type(onenormest(a, compute_w=True), tuple[np.float64, onp.Array1D[np.float64]])
assert_type(onenormest(a, 2, 5, True, False), tuple[np.float64, onp.Array1D[np.float64]])
assert_type(onenormest(a, compute_v=True), tuple[np.float64, onp.Array1D[np.float64]])
assert_type(onenormest(a, 2, 5, True, True), tuple[np.float64, onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(onenormest(a, compute_v=True, compute_w=True), tuple[np.float64, onp.Array1D[np.float64], onp.Array1D[np.float64]])
