from typing import overload

import numpy as np
import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["orthogonal_procrustes"]

###

type _ToF32ND = onp.ToFloat32_ND | onp.ToBoolND
type _AsF64ND = onp.ToArrayND[op.JustFloat | op.JustInt, npc.integer32 | npc.integer64 | npc.floating64 | npc.floating80]
type _AsC128ND = onp.ToJustComplex128_ND | onp.ToJustCLongDoubleND

###

@overload  # ~f64, +f64
def orthogonal_procrustes(  # type: ignore[overload-overlap]
    A: _AsF64ND, B: onp.ToFloatND, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float64], np.float64]: ...
@overload  # +f64, ~f64
def orthogonal_procrustes(  # type: ignore[overload-overlap]
    A: onp.ToFloatND, B: _AsF64ND, check_finite: bool = True
) -> tuple[onp.ArrayND[np.float64], np.float64]: ...
@overload  # +f32, +f32
def orthogonal_procrustes(A: _ToF32ND, B: _ToF32ND, check_finite: bool = True) -> tuple[onp.ArrayND[np.float32], np.float32]: ...
@overload  # ~c128, +c128
def orthogonal_procrustes(  # type: ignore[overload-overlap]
    A: _AsC128ND, B: onp.ToComplexND, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex128], np.float64]: ...
@overload  # +c128, ~c128
def orthogonal_procrustes(  # type: ignore[overload-overlap]
    A: onp.ToComplexND, B: _AsC128ND, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex128], np.float64]: ...
@overload  # ~f64, ~c64
def orthogonal_procrustes(  # type: ignore[overload-overlap]
    A: _AsF64ND, B: onp.ToJustComplex64_ND, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex128], np.float64]: ...
@overload  # ~c64, +f64
def orthogonal_procrustes(  # type: ignore[overload-overlap]
    A: onp.ToJustComplex64_ND, B: _AsF64ND, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex128], np.float64]: ...
@overload  # ~c64, +c64
def orthogonal_procrustes(
    A: onp.ToJustComplex64_ND, B: onp.ToComplex64_ND | onp.ToBoolND, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex64], np.float32]: ...
@overload  # +f32, ~c64
def orthogonal_procrustes(
    A: _ToF32ND, B: onp.ToJustComplex64_ND, check_finite: bool = True
) -> tuple[onp.ArrayND[np.complex64], np.float32]: ...
