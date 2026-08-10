from typing import Literal

import numpy as np
import optype.numpy as onp

__all__ = ["qr_delete", "qr_insert", "qr_update"]

###

type _Inexact = np.float32 | np.float64 | np.complex64 | np.complex128
type _QR[SCT: _Inexact] = tuple[onp.ArrayND[SCT], onp.ArrayND[SCT]]
type _Which = Literal["row", "col"]

###

def qr_delete[SCT: _Inexact](
    Q: onp.ArrayND[SCT],
    R: onp.ArrayND[SCT],
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> _QR[SCT]: ...
def qr_insert[SCT: _Inexact](
    Q: onp.ArrayND[SCT],
    R: onp.ArrayND[SCT],
    u: onp.ArrayND[SCT],
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> _QR[SCT]: ...
def qr_update[SCT: _Inexact](
    Q: onp.ArrayND[SCT],
    R: onp.ArrayND[SCT],
    u: onp.ArrayND[SCT],
    v: onp.ArrayND[SCT],
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> _QR[SCT]: ...
