from typing import assert_type

from scipy.fft import next_fast_len, prev_fast_len

###

# next_fast_len (same as prev_fast_len)
assert_type(next_fast_len(42), int)
assert_type(next_fast_len(42, True), int)
assert_type(next_fast_len(target=42, real=True), int)

# prev_fast_len (same as next_fast_len)
assert_type(prev_fast_len(42), int)
assert_type(prev_fast_len(42, True), int)
assert_type(prev_fast_len(target=42, real=True), int)
