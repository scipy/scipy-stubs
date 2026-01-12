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
from scipy.fft import dctn, dstn, idctn, idstn

###
# NOTE: the signatures are practically equivalent

# dctn
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

# idctn
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

# dstn
_assert_1d_f64(dstn(int_1d))
_assert_1d_f64(dstn(float_1d))
_assert_1d_f64(dstn(float_1d))
_assert_1d_c128(dstn(complex_1d))
_assert_1d_f64(dstn(i16_1d))
_assert_1d_f32(dstn(f32_1d))
_assert_1d_f64(dstn(f64_1d))
_assert_1d_f80(dstn(f80_1d))
_assert_1d_c64(dstn(c64_1d))
_assert_1d_c128(dstn(c128_1d))
_assert_1d_c160(dstn(c160_1d))
_assert_2d_f64(dstn(int_2d))
_assert_2d_f64(dstn(float_2d))
_assert_2d_f64(dstn(complex_2d))
_assert_2d_f64(dstn(i16_2d))
_assert_2d_f32(dstn(f32_2d))
_assert_2d_f64(dstn(f64_2d))
_assert_2d_f80(dstn(f80_2d))
_assert_2d_c64(dstn(c64_2d))
_assert_2d_c128(dstn(c128_2d))
_assert_2d_c160(dstn(c160_2d))

# idstn
_assert_1d_f64(idstn(int_1d))
_assert_1d_f64(idstn(float_1d))
_assert_1d_f64(idstn(float_1d))
_assert_1d_c128(idstn(complex_1d))
_assert_1d_f64(idstn(i16_1d))
_assert_1d_f32(idstn(f32_1d))
_assert_1d_f64(idstn(f64_1d))
_assert_1d_f80(idstn(f80_1d))
_assert_1d_c64(idstn(c64_1d))
_assert_1d_c128(idstn(c128_1d))
_assert_1d_c160(idstn(c160_1d))
_assert_2d_f64(idstn(int_2d))
_assert_2d_f64(idstn(float_2d))
_assert_2d_f64(idstn(complex_2d))
_assert_2d_f64(idstn(i16_2d))
_assert_2d_f32(idstn(f32_2d))
_assert_2d_f64(idstn(f64_2d))
_assert_2d_f80(idstn(f80_2d))
_assert_2d_c64(idstn(c64_2d))
_assert_2d_c128(idstn(c128_2d))
_assert_2d_c160(idstn(c160_2d))
