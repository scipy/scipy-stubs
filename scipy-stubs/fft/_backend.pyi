from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, ClassVar, Final, Literal, Protocol, final, override, type_check_only

import optype as op

from scipy._lib._uarray._backend import _Backend

@type_check_only
class _CanUAFunction(Protocol):
    @property
    def __ua_function__(self, /) -> Callable[..., object]: ...

type _UABackend = ModuleType | _CanUAFunction | type[_CanUAFunction]
type _ToBackend = _UABackend | Literal["scipy"]

###

@final
class _ScipyBackend[T](_Backend[T]):
    __ua_domain__: ClassVar = "numpy.scipy.fft"

    @override
    @staticmethod
    def __ua_function__(method: Callable[..., T], args: Sequence[object], kwargs: Mapping[str, object]) -> T: ...

_named_backends: Final[dict[str, type[_Backend[Any]]]] = ...

def _backend_from_arg(backend: _ToBackend) -> _UABackend: ...
def set_global_backend(backend: _ToBackend, coerce: bool = False, only: bool = False, try_last: bool = False) -> None: ...
def register_backend(backend: _ToBackend) -> None: ...
def set_backend(backend: _ToBackend, coerce: bool = False, only: bool = False) -> op.CanWith[None, None]: ...
def skip_backend(backend: _ToBackend) -> op.CanWith[None, None]: ...
