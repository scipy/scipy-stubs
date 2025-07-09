from typing import Literal

import optype as op
import optype.numpy as onp
import optype.numpy.compat as npc

import scipy._typing as spt

__all__ = ["differential_entropy", "entropy"]

def entropy(
    pk: onp.ToFloatND,
    qk: onp.ToFloatND | None = None,
    base: onp.ToFloat | None = None,
    axis: int = 0,
    *,
    nan_policy: spt.NanPolicy = "propagate",
    keepdims: onp.ToBool = False,
) -> float | npc.floating | onp.ArrayND[npc.floating]: ...
def differential_entropy(
    values: onp.ToFloatND,
    *,
    window_length: onp.ToInt | None = None,
    base: onp.ToFloat | None = None,
    axis: op.CanIndex = 0,
    method: Literal["vasicek", "van es", "ebrahimi", "correa", "auto"] = "auto",
    nan_policy: spt.NanPolicy = "propagate",
    keepdims: onp.ToBool = False,
) -> float | npc.floating | onp.ArrayND[npc.floating]: ...
