from typing import TypeAlias, TypeVar, assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import coo_matrix
from scipy.stats.contingency import crosstab

###

_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]

_list_bool: list[bool]
_list_int: list[int]
_list_float: list[float]
_list_complex: list[complex]
_list_bytes: list[bytes]
_list_str: list[str]

_T = TypeVar("_T")
_Tuple1N: TypeAlias = tuple[_T, *tuple[_T, ...]]

###

assert_type(crosstab(_f32_2d, _f32_2d).elements, _Tuple1N[onp.Array1D[np.float32]])
assert_type(crosstab(_f64_2d, _f64_2d).elements, _Tuple1N[onp.Array1D[np.float64]])
assert_type(crosstab(_list_bool, _list_bool).elements, _Tuple1N[onp.Array1D[np.bool_]])
assert_type(crosstab(_list_int, _list_int).elements, _Tuple1N[onp.Array1D[np.int_]])
assert_type(crosstab(_list_float, _list_float).elements, _Tuple1N[onp.Array1D[np.float64]])
assert_type(crosstab(_list_complex, _list_complex).elements, _Tuple1N[onp.Array1D[np.complex128]])
assert_type(crosstab(_list_bytes, _list_bytes).elements, _Tuple1N[onp.Array1D[np.bytes_]])
assert_type(crosstab(_list_str, _list_str).elements, _Tuple1N[onp.Array1D[np.str_]])

assert_type(crosstab(_f64_2d, _f64_2d).count, onp.ArrayND[np.intp])
assert_type(crosstab(_f64_2d, _f64_2d, sparse=True).count, coo_matrix[np.intp])
