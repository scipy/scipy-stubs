from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

from scipy._typing import Untyped

from scipy.special import gammaln as gammaln, logsumexp as logsumexp, roots_legendre as roots_legendre

def trapezoid(y, x: Untyped | None = None, dx: float = 1.0, axis: int = -1) -> Untyped: ...

class CacheAttributes(Protocol):
    cache: dict[int, tuple[Any, Any]]

CacheAttributes = Callable

def cache_decorator(func: Callable) -> CacheAttributes: ...
def fixed_quad(func, a, b, args=(), n: int = 5) -> Untyped: ...
def tupleset(t, i, value) -> Untyped: ...
def cumulative_trapezoid(
    y, x: Untyped | None = None, dx: float = 1.0, axis: int = -1, initial: Untyped | None = None
) -> Untyped: ...
def simpson(y, *, x: Untyped | None = None, dx: float = 1.0, axis: int = -1) -> Untyped: ...
def cumulative_simpson(
    y, *, x: Untyped | None = None, dx: float = 1.0, axis: int = -1, initial: Untyped | None = None
) -> Untyped: ...
def romb(y, dx: float = 1.0, axis: int = -1, show: bool = False) -> Untyped: ...
def newton_cotes(rn, equal: int = 0) -> Untyped: ...

class QMCQuadResult(NamedTuple):
    integral: Untyped
    standard_error: Untyped

def qmc_quad(
    func, a, b, *, n_estimates: int = 8, n_points: int = 1024, qrng: Untyped | None = None, log: bool = False
) -> Untyped: ...
