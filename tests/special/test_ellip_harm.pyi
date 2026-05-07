from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.special import ellip_harm, ellip_harm_2, ellip_normal

assert_type(ellip_harm(2.0, 3.0, 3, 4.0, 6), np.float64)
assert_type(ellip_harm(np.float32(2.0), np.float32(3.0), np.uint8(3), np.float32(4.0), np.float32(6.0)), np.float64)
assert_type(ellip_harm(2.0, 3.0, 3, 4.0, 6, signm=1), np.float64)
assert_type(ellip_harm(2.0, 3.0, 3, 4.0, 6, signm=1, signn=-1), np.float64)
assert_type(ellip_harm_2(2.0, 3.0, 3, 4, 6.0), onp.Array0D[np.float64])
assert_type(ellip_normal(2.0, 3.0, 3, 4), onp.Array0D[np.float64])
