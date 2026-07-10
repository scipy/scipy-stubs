# type-tests for `stats/_sensitivity_analysis.pyi`

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import BootstrapSobolResult, SobolResult

_f_A: onp.Array2D[np.float64]
_f_AB: onp.Array3D[np.float64]

###
# sobol_indices

def _func_1d(x: onp.Array2D[np.float64]) -> onp.Array1D[np.float64]: ...
def _func_2d(x: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]: ...
def _func_nd(x: onp.Array2D[np.float64]) -> npt.NDArray[np.float64]: ...

_si_1d = sobol_indices(func=_func_1d, n=1024, dists=[])
assert_type(_si_1d, SobolResult[tuple[int]])
assert_type(_si_1d.first_order, onp.Array1D[np.float64])
assert_type(_si_1d.total_order, onp.Array1D[np.float64])

_si_2d = sobol_indices(func=_func_2d, n=1024, dists=[])
assert_type(_si_2d, SobolResult[tuple[int, int]])
assert_type(_si_2d.first_order, onp.Array2D[np.float64])
assert_type(_si_2d.total_order, onp.Array2D[np.float64])

_si_nd = sobol_indices(func=_func_nd, n=1024, dists=[])
assert_type(_si_nd, SobolResult[tuple[Any, ...]])
assert_type(_si_nd.first_order, onp.ArrayND[np.float64])
assert_type(_si_nd.total_order, onp.ArrayND[np.float64])

_si_dict = sobol_indices(func={"f_A": _f_A, "f_B": _f_A, "f_AB": _f_AB}, n=1024)
assert_type(_si_dict, SobolResult[tuple[Any, ...]])
assert_type(_si_dict.first_order, onp.ArrayND[np.float64])
assert_type(_si_dict.total_order, onp.ArrayND[np.float64])

_bs = _si_2d.bootstrap(confidence_level=0.95, n_resamples=100)
assert_type(_bs, BootstrapSobolResult)
assert_type(_bs.first_order, BootstrapResult)
assert_type(_bs.total_order, BootstrapResult)
