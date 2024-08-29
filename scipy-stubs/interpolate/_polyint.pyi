from scipy._typing import Untyped

from scipy._lib._util import check_random_state as check_random_state, float_factorial as float_factorial
from scipy.special import factorial as factorial

class _Interpolator1D:
    dtype: Untyped
    def __init__(self, xi: Untyped | None = None, yi: Untyped | None = None, axis: Untyped | None = None): ...
    def __call__(self, x) -> Untyped: ...

class _Interpolator1DWithDerivatives(_Interpolator1D):
    def derivatives(self, x, der: Untyped | None = None) -> Untyped: ...
    def derivative(self, x, der: int = 1) -> Untyped: ...

class KroghInterpolator(_Interpolator1DWithDerivatives):
    xi: Untyped
    yi: Untyped
    c: Untyped
    def __init__(self, xi, yi, axis: int = 0): ...

def krogh_interpolate(xi, yi, x, der: int = 0, axis: int = 0) -> Untyped: ...
def approximate_taylor_polynomial(f, x, degree, scale, order: Untyped | None = None) -> Untyped: ...

class BarycentricInterpolator(_Interpolator1DWithDerivatives):
    xi: Untyped
    n: Untyped
    wi: Untyped
    def __init__(
        self, xi, yi: Untyped | None = None, axis: int = 0, *, wi: Untyped | None = None, random_state: Untyped | None = None
    ): ...
    yi: Untyped
    def set_yi(self, yi, axis: Untyped | None = None): ...
    def add_xi(self, xi, yi: Untyped | None = None): ...
    def __call__(self, x) -> Untyped: ...
    def derivative(self, x, der: int = 1) -> Untyped: ...

def barycentric_interpolate(xi, yi, x, axis: int = 0, *, der: int = 0) -> Untyped: ...
