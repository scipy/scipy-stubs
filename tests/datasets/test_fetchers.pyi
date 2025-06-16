from typing import assert_type

import numpy as np

from scipy import datasets

assert_type(datasets.ascent(), np.ndarray[tuple[int, int], np.dtype[np.uint8]])

assert_type(datasets.electrocardiogram(), np.ndarray[tuple[int], np.dtype[np.float64]])

assert_type(datasets.face(), np.ndarray[tuple[int, int, int], np.dtype[np.uint8]])
assert_type(datasets.face(False), np.ndarray[tuple[int, int, int], np.dtype[np.uint8]])
assert_type(datasets.face(True), np.ndarray[tuple[int, int], np.dtype[np.uint8]])
