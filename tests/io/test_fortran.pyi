from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.io import FortranFile

_i32_1d: onp.Array1D[np.int32]

###

_f = FortranFile("test.unf", "w")
assert_type(_f, FortranFile)
assert_type(_f.write_record(_i32_1d), None)

_f.read_record()  # type:ignore[call-overload] # pyright:ignore[reportCallIssue] # pyrefly:ignore[no-matching-overload]

assert_type(_f.read_record(np.float32), onp.Array1D[np.float32])
assert_type(_f.read_record(dtype=np.float64), onp.Array1D[np.float64])
assert_type(_f.read_record("<f4"), onp.Array1D)
assert_type(_f.read_record(dtype="<f4"), onp.Array1D)
assert_type(_f.read_record(np.float32, np.int32), tuple[onp.Array1D[np.float32], onp.Array1D[np.int32]])
assert_type(_f.read_record(np.float32, dtype=np.int32), tuple[onp.Array1D[np.float32], onp.Array1D[np.int32]])
assert_type(_f.read_record("<f4", "<i4"), tuple[onp.Array1D, ...])
assert_type(_f.read_record("<f4", "<i4", "<i2"), tuple[onp.Array1D, ...])
assert_type(_f.read_record("<f4", dtype="<i4"), tuple[onp.Array1D, ...])
assert_type(_f.read_ints(), onp.Array1D[np.int32])
assert_type(_f.read_ints(np.int16), onp.Array1D[np.int16])
assert_type(_f.read_reals(), onp.Array1D[np.float64])
assert_type(_f.read_reals(np.float32), onp.Array1D[np.float32])
assert_type(_f.close(), None)
