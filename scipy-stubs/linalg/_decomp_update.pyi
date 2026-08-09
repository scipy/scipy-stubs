# NOTE: mypy incorrectly sees disjoint dtypes as overlapping
# mypy: disable-error-code=overload-overlap
from typing import Literal, overload

import numpy as np
import optype.numpy as onp

__all__ = ["qr_delete", "qr_insert", "qr_update"]

###

type _QR32 = tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]]
type _QR64 = tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
type _QRc64 = tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64]]
type _QRc128 = tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]]

type _Which = Literal["row", "col"]

###
# qr_delete

@overload
def qr_delete(
    Q: onp.ToJustFloat32_ND,
    R: onp.ToJustFloat32_ND,
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QR32: ...
@overload
def qr_delete(
    Q: onp.ToJustFloat64_ND,
    R: onp.ToJustFloat64_ND,
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QR64: ...
@overload
def qr_delete(
    Q: onp.ToJustComplex64_ND,
    R: onp.ToJustComplex64_ND,
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QRc64: ...
@overload
def qr_delete(
    Q: onp.ToJustComplex128_ND,
    R: onp.ToJustComplex128_ND,
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QRc128: ...

###
# qr_insert

@overload
def qr_insert(
    Q: onp.ToJustFloat32_ND,
    R: onp.ToJustFloat32_ND,
    u: onp.ToJustFloat32_ND,
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> _QR32: ...
@overload
def qr_insert(
    Q: onp.ToJustFloat64_ND,
    R: onp.ToJustFloat64_ND,
    u: onp.ToJustFloat64_ND,
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> _QR64: ...
@overload
def qr_insert(
    Q: onp.ToJustComplex64_ND,
    R: onp.ToJustComplex64_ND,
    u: onp.ToJustComplex64_ND,
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> _QRc64: ...
@overload
def qr_insert(
    Q: onp.ToJustComplex128_ND,
    R: onp.ToJustComplex128_ND,
    u: onp.ToJustComplex128_ND,
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> _QRc128: ...

###
# qr_update

@overload
def qr_update(
    Q: onp.ToJustFloat32_ND,
    R: onp.ToJustFloat32_ND,
    u: onp.ToJustFloat32_ND,
    v: onp.ToJustFloat32_ND,
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QR32: ...
@overload
def qr_update(
    Q: onp.ToJustFloat64_ND,
    R: onp.ToJustFloat64_ND,
    u: onp.ToJustFloat64_ND,
    v: onp.ToJustFloat64_ND,
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QR64: ...
@overload
def qr_update(
    Q: onp.ToJustComplex64_ND,
    R: onp.ToJustComplex64_ND,
    u: onp.ToJustComplex64_ND,
    v: onp.ToJustComplex64_ND,
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QRc64: ...
@overload
def qr_update(
    Q: onp.ToJustComplex128_ND,
    R: onp.ToJustComplex128_ND,
    u: onp.ToJustComplex128_ND,
    v: onp.ToJustComplex128_ND,
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QRc128: ...
