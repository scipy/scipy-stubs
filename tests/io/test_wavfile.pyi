import io
from pathlib import Path
from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.io.wavfile import read, write

###

file_str: str
file_path: Path
file_io: io.BytesIO

_u8_1d: onp.Array1D[np.uint8]
_i16_1d: onp.Array1D[np.int16]
_i32_1d: onp.Array1D[np.int32]
_i64_1d: onp.Array1D[np.int64]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]

###

# read
assert_type(read(file_str), tuple[int, onp.Array])
assert_type(read(file_path), tuple[int, onp.Array])
assert_type(read(file_io), tuple[int, onp.Array])

# write
assert_type(write(file_str, 44100, _u8_1d), None)
assert_type(write(file_str, 44100, _i16_1d), None)
assert_type(write(file_str, 44100, _i32_1d), None)
assert_type(write(file_str, 44100, _i64_1d), None)
assert_type(write(file_str, 44100, _f32_1d), None)
assert_type(write(file_str, 44100, _f64_1d), None)
assert_type(write(file_path, 44100, _f64_1d), None)
assert_type(write(file_io, 44100, _f64_1d), None)
