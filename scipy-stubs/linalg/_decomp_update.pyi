# NOTE: mypy incorrectly sees disjoint dtypes as overlapping
# mypy: disable-error-code=overload-overlap
from typing import Literal, overload

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["qr_delete", "qr_insert", "qr_update"]

###

type _QR[ScalarT: npc.inexact] = tuple[onp.ArrayND[ScalarT], onp.ArrayND[ScalarT]]
type _Which = Literal["row", "col"]

###

@overload  # ~f32 | ~c64
def qr_delete[ScalarT: npc.inexact32](
    Q: onp.ToArrayND[ScalarT, ScalarT],
    R: onp.ToArrayND[ScalarT, ScalarT],
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QR[ScalarT]: ...
@overload  # ~f64
def qr_delete(
    Q: onp.ToJustFloat64_ND,
    R: onp.ToJustFloat64_ND,
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QR[np.float64]: ...
@overload  # ~c128
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
@overload  # ~f32 | ~c64
def qr_insert[ScalarT: npc.inexact32](
    Q: onp.ToArrayND[ScalarT, ScalarT],
    R: onp.ToArrayND[ScalarT, ScalarT],
    u: onp.ToArrayND[ScalarT, ScalarT],
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> _QR[ScalarT]: ...
@overload  # ~f64
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
@overload  # ~c128
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
@overload  # ~f32 | ~c64
def qr_update[ScalarT: npc.inexact32](
    Q: onp.ToArrayND[ScalarT, ScalarT],
    R: onp.ToArrayND[ScalarT, ScalarT],
    u: onp.ToArrayND[ScalarT, ScalarT],
    v: onp.ToArrayND[ScalarT, ScalarT],
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QR[ScalarT]: ...
@overload  # ~f64
def qr_update(
    Q: onp.ToJustFloat64_ND,
    R: onp.ToJustFloat64_ND,
    u: onp.ToJustFloat64_ND,
    v: onp.ToJustFloat64_ND,
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QR[np.float64]: ...
@overload  # ~c128
def qr_update(
    Q: onp.ToJustComplex128_ND,
    R: onp.ToJustComplex128_ND,
    u: onp.ToJustComplex128_ND,
    v: onp.ToJustComplex128_ND,
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QR[np.complex128]: ...
