from typing import assert_type

from scipy.fft import get_workers, set_workers

###

with set_workers(4) as ctx:
    assert_type(ctx, None)

assert_type(get_workers(), int)
