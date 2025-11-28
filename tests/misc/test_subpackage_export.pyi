import scipy

# `scipy.misc` is not exported from `scipy/__init__.py`

# pyrefly: ignore[implicit-import]
scipy.misc  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
