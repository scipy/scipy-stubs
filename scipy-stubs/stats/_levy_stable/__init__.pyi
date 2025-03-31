from collections.abc import Callable
from typing import Final, Literal

import numpy as np
import optype.numpy as onp
from scipy.stats.distributions import rv_continuous

__all__ = ["levy_stable", "levy_stable_gen", "pdf_from_cf_with_fft"]

Cotes: Final[onp.Array2D[np.float64]] = ...
Cotes_table: Final[onp.Array1D[np.object_]] = ...
levy_stable: Final[levy_stable_gen] = ...

class levy_stable_gen(rv_continuous):
    parameterization: Literal["S0", "S1"]
    pdf_default_method: Literal["piecewise", "best", "zolotarev", "dni", "quadrature", "fft-simpson"]
    cdf_default_method: Literal["piecewise", "fft-simpson"]
    quad_eps: float | int | bool
    piecewise_x_tol_near_zeta: float | int | bool
    piecewise_alpha_tol_near_one: float | int | bool
    pdf_fft_min_points_threshold: float | int | bool | None
    pdf_fft_grid_spacing: float | int | bool
    pdf_fft_n_points_two_power: float | int | bool | None
    pdf_fft_interpolation_level: int | bool
    pdf_fft_interpolation_degree: int | bool

def pdf_from_cf_with_fft(
    cf: Callable[[float | int | bool], complex | float | int | bool],
    h: float | int | bool = 0.01,
    q: int | bool = 9,
    level: int | bool = 3,
) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]]: ...
