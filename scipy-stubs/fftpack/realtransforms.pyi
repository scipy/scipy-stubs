# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing import Final
from typing_extensions import deprecated

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]

__MESSAGE: Final = "will be removed in SciPy v2.0.0"

@deprecated(__MESSAGE)
def dctn(
    x: object,
    type: object = ...,
    shape: object = ...,
    axes: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated(__MESSAGE)
def idctn(
    x: object,
    type: object = ...,
    shape: object = ...,
    axes: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated(__MESSAGE)
def dstn(
    x: object,
    type: object = ...,
    shape: object = ...,
    axes: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated(__MESSAGE)
def idstn(
    x: object,
    type: object = ...,
    shape: object = ...,
    axes: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated(__MESSAGE)
def dct(
    x: object,
    type: object = ...,
    n: object = ...,
    axis: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated(__MESSAGE)
def idct(
    x: object,
    type: object = ...,
    n: object = ...,
    axis: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated(__MESSAGE)
def dst(
    x: object,
    type: object = ...,
    n: object = ...,
    axis: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
@deprecated(__MESSAGE)
def idst(
    x: object,
    type: object = ...,
    n: object = ...,
    axis: object = ...,
    norm: object = ...,
    overwrite_x: object = ...,
) -> object: ...
