from scipy._typing import Untyped

from scipy._lib._array_api import array_namespace as array_namespace

def next_fast_len(target, real: bool = False): ...

next_fast_len: Untyped

def prev_fast_len(target, real: bool = False): ...

prev_fast_len: Untyped

def fftfreq(n, d: float = 1.0, *, xp: Untyped | None = None, device: Untyped | None = None) -> Untyped: ...
def rfftfreq(n, d: float = 1.0, *, xp: Untyped | None = None, device: Untyped | None = None) -> Untyped: ...
def fftshift(x, axes: Untyped | None = None) -> Untyped: ...
def ifftshift(x, axes: Untyped | None = None) -> Untyped: ...
