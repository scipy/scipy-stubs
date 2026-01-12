from typing import assert_type

import numpy as np
import optype.numpy as onp

from .test_basic import (
    _assert_1d_c64,
    _assert_1d_c128,
    _assert_1d_c160,
    _assert_1d_f32,
    _assert_1d_f64,
    _assert_1d_f80,
    _assert_2d_c64,
    _assert_2d_c128,
    _assert_2d_c160,
    _assert_2d_f32,
    _assert_2d_f64,
    _assert_2d_f80,
    c64_1d,
    c64_2d,
    c128_1d,
    c128_2d,
    c160_1d,
    c160_2d,
    complex_1d,
    complex_2d,
    f32_1d,
    f32_2d,
    f64_1d,
    f64_2d,
    f80_1d,
    f80_2d,
    float_1d,
    float_2d,
    i16_1d,
    i16_2d,
    int_1d,
    int_2d,
)
from scipy.fft import dctn, idctn

###

# dctn (same as idctn)
_assert_1d_f64(dctn(int_1d))
_assert_1d_f64(dctn(float_1d))
_assert_1d_f64(dctn(float_1d))
_assert_1d_c128(dctn(complex_1d))
_assert_1d_f64(dctn(i16_1d))
_assert_1d_f32(dctn(f32_1d))
_assert_1d_f64(dctn(f64_1d))
_assert_1d_f80(dctn(f80_1d))
_assert_1d_c64(dctn(c64_1d))
_assert_1d_c128(dctn(c128_1d))
_assert_1d_c160(dctn(c160_1d))
_assert_2d_f64(dctn(int_2d))
_assert_2d_f64(dctn(float_2d))
_assert_2d_f64(dctn(complex_2d))
_assert_2d_f64(dctn(i16_2d))
_assert_2d_f32(dctn(f32_2d))
_assert_2d_f64(dctn(f64_2d))
_assert_2d_f80(dctn(f80_2d))
_assert_2d_c64(dctn(c64_2d))
_assert_2d_c128(dctn(c128_2d))
_assert_2d_c160(dctn(c160_2d))
# idctn (same as dctn)
_assert_1d_f64(idctn(int_1d))
_assert_1d_f64(idctn(float_1d))
_assert_1d_f64(idctn(float_1d))
_assert_1d_c128(idctn(complex_1d))
_assert_1d_f64(idctn(i16_1d))
_assert_1d_f32(idctn(f32_1d))
_assert_1d_f64(idctn(f64_1d))
_assert_1d_f80(idctn(f80_1d))
_assert_1d_c64(idctn(c64_1d))
_assert_1d_c128(idctn(c128_1d))
_assert_1d_c160(idctn(c160_1d))
_assert_2d_f64(idctn(int_2d))
_assert_2d_f64(idctn(float_2d))
_assert_2d_f64(idctn(complex_2d))
_assert_2d_f64(idctn(i16_2d))
_assert_2d_f32(idctn(f32_2d))
_assert_2d_f64(idctn(f64_2d))
_assert_2d_f80(idctn(f80_2d))
_assert_2d_c64(idctn(c64_2d))
_assert_2d_c128(idctn(c128_2d))
_assert_2d_c160(idctn(c160_2d))
