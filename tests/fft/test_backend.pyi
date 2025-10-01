from collections.abc import Callable
from typing import Any, ClassVar, assert_type, type_check_only

from scipy import fft

@type_check_only
class MyBackend:
    __ua_domain__: ClassVar[str] = ...
    @staticmethod
    def __ua_function__(method: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], /) -> Any: ...

# set_global_backend
assert_type(fft.set_global_backend("scipy"), None)
assert_type(fft.set_global_backend("scipy", coerce=True), None)
assert_type(fft.set_global_backend("scipy", only=True), None)
assert_type(fft.set_global_backend("scipy", try_last=True), None)
assert_type(fft.set_global_backend(MyBackend), None)

# register_backend
assert_type(fft.register_backend("scipy"), None)
assert_type(fft.register_backend(MyBackend), None)

# See https://github.com/facebook/pyrefly/issues/1221

# set_backend
with fft.set_backend("scipy") as ctx1:  # pyrefly: ignore[bad-context-manager]
    assert_type(ctx1, None)
with fft.set_backend("scipy", coerce=True) as ctx2:  # pyrefly: ignore[bad-context-manager]
    assert_type(ctx2, None)
with fft.set_backend("scipy", only=True) as ctx3:  # pyrefly: ignore[bad-context-manager]
    assert_type(ctx3, None)
with fft.set_backend(MyBackend) as ctx4:  # pyrefly: ignore[bad-context-manager]
    assert_type(ctx4, None)

# skip_backend
with fft.skip_backend("scipy") as ctx5:  # pyrefly: ignore[bad-context-manager]
    assert_type(ctx5, None)
with fft.skip_backend(MyBackend) as ctx6:  # pyrefly: ignore[bad-context-manager]
    assert_type(ctx6, None)
