import scipy

# `scipy.misc` is not exported from `scipy/__init__.py`

scipy.misc  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
