# NOTE: mypy incorrectly sees disjoint dtypes as overlapping
# mypy: disable-error-code=overload-overlap
from typing import Literal, overload

import numpy as np
import optype.numpy as onp

__all__ = ["qr_delete", "qr_insert", "qr_update"]

###

type _Inexact = np.float32 | np.float64 | np.complex64 | np.complex128
type _QR[SCT: _Inexact] = tuple[onp.ArrayND[SCT], onp.ArrayND[SCT]]
type _Which = Literal["row", "col"]

###

@overload  # float32 -> float32
def qr_delete(
    Q: onp.ToArrayND[np.float32, np.float32],
    R: onp.ToArrayND[np.float32, np.float32],
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QR[np.float32]: ...
@overload  # float64 -> float64
def qr_delete(
    Q: onp.ToJustFloat64_ND,
    R: onp.ToJustFloat64_ND,
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QR[np.float64]: ...
@overload  # complex64 -> complex64
def qr_delete(
    Q: onp.ToArrayND[np.complex64, np.complex64],
    R: onp.ToArrayND[np.complex64, np.complex64],
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QR[np.complex64]: ...
@overload  # complex128 -> complex128
def qr_delete(
    Q: onp.ToJustComplex128_ND,
    R: onp.ToJustComplex128_ND,
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QR[np.complex128]: ...

#
@overload  # float32 -> float32
def qr_insert(
    Q: onp.ToArrayND[np.float32, np.float32],
    R: onp.ToArrayND[np.float32, np.float32],
    u: onp.ToArrayND[np.float32, np.float32],
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> _QR[np.float32]: ...
@overload  # float64 -> float64
def qr_insert(
    Q: onp.ToJustFloat64_ND,
    R: onp.ToJustFloat64_ND,
    u: onp.ToJustFloat64_ND,
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> _QR[np.float64]: ...
@overload  # complex64 -> complex64
def qr_insert(
    Q: onp.ToArrayND[np.complex64, np.complex64],
    R: onp.ToArrayND[np.complex64, np.complex64],
    u: onp.ToArrayND[np.complex64, np.complex64],
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> _QR[np.complex64]: ...
@overload  # complex128 -> complex128
def qr_insert(
    Q: onp.ToJustComplex128_ND,
    R: onp.ToJustComplex128_ND,
    u: onp.ToJustComplex128_ND,
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> _QR[np.complex128]: ...

#
@overload  # float32 -> float32
def qr_update(
    Q: onp.ToArrayND[np.float32, np.float32],
    R: onp.ToArrayND[np.float32, np.float32],
    u: onp.ToArrayND[np.float32, np.float32],
    v: onp.ToArrayND[np.float32, np.float32],
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QR[np.float32]: ...
@overload  # float64 -> float64
def qr_update(
    Q: onp.ToJustFloat64_ND,
    R: onp.ToJustFloat64_ND,
    u: onp.ToJustFloat64_ND,
    v: onp.ToJustFloat64_ND,
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QR[np.float64]: ...
@overload  # complex64 -> complex64
def qr_update(
    Q: onp.ToArrayND[np.complex64, np.complex64],
    R: onp.ToArrayND[np.complex64, np.complex64],
    u: onp.ToArrayND[np.complex64, np.complex64],
    v: onp.ToArrayND[np.complex64, np.complex64],
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QR[np.complex64]: ...
@overload  # complex128 -> complex128
def qr_update(
    Q: onp.ToJustComplex128_ND,
    R: onp.ToJustComplex128_ND,
    u: onp.ToJustComplex128_ND,
    v: onp.ToJustComplex128_ND,
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QR[np.complex128]: ...
