from typing import Literal, Never, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["expm_cond", "expm_frechet"]

###

type _Method = Literal["SPS", "blockEnlarge"]

###

@overload  # +f64, +f64
def expm_frechet(
    A: onp.ToFloatND, E: onp.ToFloatND, method: _Method | None = None, compute_expm: onp.ToTrue = True, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...
@overload  # +f64, ~complex
def expm_frechet(
    A: onp.ToFloatND,
    E: onp.ToJustComplexND,
    method: _Method | None = None,
    compute_expm: onp.ToTrue = True,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]]: ...
@overload  # ~complex, +complex
def expm_frechet(
    A: onp.ToJustComplexND,
    E: onp.ToComplexND,
    method: _Method | None = None,
    compute_expm: onp.ToTrue = True,
    check_finite: bool = True,
) -> tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]]: ...
@overload  # +f64, +f64, compute_expm=False
def expm_frechet(
    A: onp.ToFloatND, E: onp.ToFloatND, method: _Method | None, compute_expm: onp.ToFalse, check_finite: bool = True
) -> onp.ArrayND[np.float64]: ...
@overload  # +f64, +f64, compute_expm=False (keyword)
def expm_frechet(
    A: onp.ToFloatND, E: onp.ToFloatND, method: _Method | None = None, *, compute_expm: onp.ToFalse, check_finite: bool = True
) -> onp.ArrayND[np.float64]: ...
@overload  # +f64, ~complex, compute_expm=False
def expm_frechet(
    A: onp.ToFloatND, E: onp.ToJustComplexND, method: _Method | None, compute_expm: onp.ToFalse, check_finite: bool = True
) -> onp.ArrayND[np.complex128]: ...
@overload  # +f64, ~complex, compute_expm=False (keyword)
def expm_frechet(
    A: onp.ToFloatND,
    E: onp.ToJustComplexND,
    method: _Method | None = None,
    *,
    compute_expm: onp.ToFalse,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...
@overload  # ~complex, +complex, compute_expm=False
def expm_frechet(
    A: onp.ToJustComplexND, E: onp.ToComplexND, method: _Method | None, compute_expm: onp.ToFalse, check_finite: bool = True
) -> onp.ArrayND[np.complex128]: ...
@overload  # ~complex, +complex, compute_expm=False (keyword)
def expm_frechet(
    A: onp.ToJustComplexND,
    E: onp.ToComplexND,
    method: _Method | None = None,
    *,
    compute_expm: onp.ToFalse,
    check_finite: bool = True,
) -> onp.ArrayND[np.complex128]: ...

#
@overload
def expm_cond(
    A: onp.ArrayND[npc.number, tuple[Never, Never, Never, Never]], check_finite: bool = True
) -> np.float64 | onp.ArrayND[np.float64]: ...
@overload
def expm_cond(A: onp.ToComplexStrict2D, check_finite: bool = True) -> np.float64: ...
@overload
def expm_cond(A: onp.ToComplexStrict3D, check_finite: bool = True) -> onp.Array1D[np.float64]: ...
@overload
def expm_cond(A: onp.ToComplexND, check_finite: bool = True) -> np.float64 | onp.ArrayND[np.float64]: ...
