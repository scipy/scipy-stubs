# This file is not meant for public use and will be removed in SciPy v2.0.0.

from typing_extensions import Any, deprecated

__all__ = ["OptimizeResult", "fmin_cobyla"]

@deprecated("will be removed in SciPy v2.0.0")
class OptimizeResult(Any): ...  # type: ignore[subclass-any]

@deprecated("will be removed in SciPy v2.0.0")
def fmin_cobyla(
    func: object,
    x0: object,
    cons: object,
    args: object = ...,
    consargs: object = ...,
    rhobeg: object = ...,
    rhoend: object = ...,
    maxfun: object = ...,
    disp: object = ...,
    catol: object = ...,
    *,
    callback: object = ...,
) -> object: ...
