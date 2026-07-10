from typing import assert_type

import numpy as np

from scipy.interpolate import pade

coeffs: list[float]
coeffs_c: list[complex]

assert_type(pade(coeffs, 3), tuple[np.poly1d, np.poly1d])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(pade(coeffs, 3, n=2), tuple[np.poly1d, np.poly1d])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(pade(coeffs_c, 3), tuple[np.poly1d, np.poly1d])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
