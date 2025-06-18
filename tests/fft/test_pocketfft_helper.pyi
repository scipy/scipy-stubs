from typing import assert_type

from scipy import fft

###

with fft.set_workers(4) as ctx:
    assert_type(ctx, None)

assert_type(fft.get_workers(), int)
