from typing import assert_type

import numpy as np

from scipy.datasets import ascent, electrocardiogram, face

assert_type(ascent(), np.ndarray[tuple[int, int], np.dtype[np.uint8]])

assert_type(electrocardiogram(), np.ndarray[tuple[int], np.dtype[np.float64]])

assert_type(face(), np.ndarray[tuple[int, int, int], np.dtype[np.uint8]])
assert_type(face(False), np.ndarray[tuple[int, int, int], np.dtype[np.uint8]])
assert_type(face(True), np.ndarray[tuple[int, int], np.dtype[np.uint8]])
