from collections.abc import Sequence
from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import coo_array
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

###

assert_type(crosstab(_f32_2d, _f32_2d).elements, Sequence[onp.Array1D[np.float32] | Any])
assert_type(crosstab(_f64_2d, _f64_2d, levels=(None, _list_float)).elements, Sequence[onp.Array1D[np.float64] | Any])
assert_type(crosstab(_list_bool, _list_bool, levels=(None, _list_bool)).elements, Sequence[onp.Array1D[np.bool] | Any])
assert_type(crosstab(_list_int, _list_int, levels=(None, _list_int)).elements, Sequence[onp.Array1D[np.int_] | Any])
assert_type(crosstab(_list_float, _list_float, levels=(None, _list_float)).elements, Sequence[onp.Array1D[np.float64] | Any])
assert_type(
    crosstab(_list_complex, _list_complex, levels=(None, _list_complex)).elements, Sequence[onp.Array1D[np.complex128] | Any]
)
assert_type(crosstab(_list_bytes, _list_bytes, levels=(None, _list_bytes)).elements, Sequence[onp.Array1D[np.bytes_] | Any])
assert_type(crosstab(_list_str, _list_str, levels=(None, _list_str)).elements, Sequence[onp.Array1D[np.str_] | Any])
assert_type(crosstab(_list_str, _list_int).elements, Sequence[onp.Array1D[Any] | Any])

assert_type(crosstab(_f64_2d, _f64_2d).count, onp.ArrayND[np.intp])
assert_type(crosstab(_f64_2d, _f64_2d, levels=_f64_2d, sparse=True).count, coo_array[np.intp])
assert_type(crosstab(_list_bool, _list_bool, levels=(None, _list_bool), sparse=True).count, coo_array[np.intp])
assert_type(crosstab(_list_int, _list_int, levels=(None, _list_int), sparse=True).count, coo_array[np.intp])
assert_type(crosstab(_list_float, _list_float, levels=(None, _list_float), sparse=True).count, coo_array[np.intp])
assert_type(crosstab(_list_complex, _list_complex, levels=(None, _list_complex), sparse=True).count, coo_array[np.intp])
assert_type(crosstab(_list_bytes, _list_bytes, levels=(None, _list_bytes), sparse=True).count, coo_array[np.intp])
assert_type(crosstab(_list_str, _list_str, levels=(None, _list_str), sparse=True).count, coo_array[np.intp])
assert_type(crosstab(_list_str, _list_int, sparse=True).count, coo_array[np.intp])
