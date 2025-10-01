import scipy

# `scipy.misc` is not exported from `scipy/__init__.py`

# NOTE: For some reason mypy does not report `[attr-defined]` when run as `uv run mypy`,
# even though the mypy plugin (which uses the same config) does report it.
scipy.misc  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # pyrefly: ignore[implicit-import]
