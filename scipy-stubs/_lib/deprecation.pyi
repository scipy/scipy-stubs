from collections.abc import Callable
from types import ModuleType

__all__ = ["_deprecated"]

class _DeprecationHelperStr:
    def __init__(self, /, content: str, message: str) -> None: ...

def _deprecated[FuncT: Callable[..., object]](msg: str, stacklevel: int = 2) -> Callable[[FuncT], FuncT]: ...
def deprecate_cython_api(
    module: ModuleType, routine_name: str, new_name: str | None = None, message: str | None = None
) -> None: ...
