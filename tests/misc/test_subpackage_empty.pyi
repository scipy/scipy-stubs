from scipy import misc

# ensure there are no 1.14 remnants lefts:
# https://docs.scipy.org/doc/scipy-1.14.1/reference/misc.html

misc.__all__  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
misc.ascent  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
misc.central_diff_weights  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
misc.derivative  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
misc.face  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
misc.electrocardiogram  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

# ensure that the (empty) submodules are not exported

# NOTE: For some reason mypy does not report these `[attr-defined]` errors when run as `uv run mypy`,
# even though the mypy plugin (which uses the same config) does report it.
misc.common  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
misc.doccer  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
