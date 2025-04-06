import numpy as np
from scipy.interpolate import RegularGridInterpolator

# regression test for https://github.com/scipy/scipy-stubs/issues/497
RegularGridInterpolator(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
