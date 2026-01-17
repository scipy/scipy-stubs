from typing import Any, assert_type

import numpy as np

from scipy.io import netcdf_file, netcdf_variable

###

_file: netcdf_file

###

assert_type(_file.createDimension("time", 10), None)
assert_type(_file.createVariable("temperature", np.float32, ("time",)), netcdf_variable[tuple[Any, ...], np.float32])
