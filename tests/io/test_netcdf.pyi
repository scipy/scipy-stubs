from typing import Any, assert_type

import numpy as np

from scipy.io import netcdf_file, netcdf_variable

###

_file: netcdf_file

_var_0d_i64: netcdf_variable[tuple[()], np.int64]
_var_1d_i64: netcdf_variable[tuple[()], np.int64]
_var_0d_f32: netcdf_variable[tuple[()], np.float32]
_var_1d_f32: netcdf_variable[tuple[()], np.float32]

###

assert_type(_file.createDimension("time", 10), None)
assert_type(_file.createVariable("temperature", np.float32, ("time",)), netcdf_variable[tuple[Any, ...], np.float32])

assert_type(_var_0d_i64.getValue(), int)
assert_type(_var_1d_i64.getValue(), int)
assert_type(_var_0d_f32.getValue(), float)
assert_type(_var_1d_f32.getValue(), float)
