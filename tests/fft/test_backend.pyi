from collections.abc import Callable
from typing import Any, ClassVar, assert_type, type_check_only

from scipy.fft import register_backend, set_backend, set_global_backend, skip_backend

@type_check_only
class MyBackend:
    __ua_domain__: ClassVar[str] = ...
    @staticmethod
    def __ua_function__(method: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], /) -> Any: ...

# set_global_backend
assert_type(set_global_backend("scipy"), None)
assert_type(set_global_backend("scipy", coerce=True), None)
assert_type(set_global_backend("scipy", only=True), None)
assert_type(set_global_backend("scipy", try_last=True), None)
assert_type(set_global_backend(MyBackend), None)

# register_backend
assert_type(register_backend("scipy"), None)
assert_type(register_backend(MyBackend), None)

# set_backend
with set_backend("scipy") as ctx1:
    assert_type(ctx1, None)
with set_backend("scipy", coerce=True) as ctx2:
    assert_type(ctx2, None)
with set_backend("scipy", only=True) as ctx3:
    assert_type(ctx3, None)
with set_backend(MyBackend) as ctx4:
    assert_type(ctx4, None)

# skip_backend
with skip_backend("scipy") as ctx5:
    assert_type(ctx5, None)
with skip_backend(MyBackend) as ctx6:
    assert_type(ctx6, None)
