"""Generate release Markdown without creating or updating a GitHub release."""

# Diagnostic messages and the release notes' Unicode typography are intentional.
# ruff: file-ignore[raise-vanilla-args, ambiguous-unicode-character-string]

import argparse
import json
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import tomllib
import unicodedata
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import batched
from pathlib import Path
from typing import NamedTuple, TypedDict, cast

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
REPO = "scipy/scipy-stubs"

KEEP_GITMOJI = {"Maintenance", "Other Changes"}
GROUP_BY_PACKAGE = {"Improvements", "Fixes"}
COLLAPSE_TESTING_AT = 10
LABEL_BATCH_SIZE = 50
VERSION_COMPONENTS = 4  # SciPy major.minor.patch plus the stubs' own patch

PR_ENTRY = re.compile(
    rf"[*-] (?P<title>.+) by @(?P<author>\S+) in "
    rf"https://github\.com/{REPO}/pull/(?P<number>\d+)[ \t]*"
)
NEW_CONTRIBUTOR = re.compile(
    r"^[*-] @(\S+) made their first contribution in ", re.MULTILINE
)
PACKAGE_LABEL = re.compile(r"scipy\.(?:\*|\w+(?:\.\w+)*)")
PYTHON_CLASSIFIER = re.compile(r"Programming Language :: Python :: 3\.(\d+)")
# Symbols, marks (variation selectors, skin tones, keycaps) and format chars (ZWJ).
EMOJI_CATEGORIES = {"So", "Sm", "Sk", "Mn", "Me", "Cf"}

type Packages = Mapping[int, tuple[str, ...]]


class Entry(NamedTuple):
    number: int
    author: str
    title: str


class Project(TypedDict):
    version: str
    classifiers: list[str]
    dependencies: list[str]


def run(*command: str) -> str:
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def github(endpoint: str, *, jq: str, **fields: str) -> str:
    arguments = [
        arg for key, value in fields.items() for arg in ("-f", f"{key}={value}")
    ]
    return run("gh", "api", endpoint, "--jq", jq, *arguments)


def strip_gitmoji(title: str) -> str:
    for index, char in enumerate(title):
        if not char.isspace() and (
            char.isascii() or unicodedata.category(char) not in EMOJI_CATEGORIES
        ):
            return title[index:]
    return title


def parse_entries(body: str) -> list[Entry]:
    entries: list[Entry] = []
    for line in body.strip("\n").splitlines():
        if not (match := PR_ENTRY.fullmatch(line)):
            raise ValueError(f"Unrecognized release note entry: {line!r}")
        entries.append(Entry(int(match["number"]), match["author"], match["title"]))
    return entries


def format_entries(entries: Iterable[Entry]) -> str:
    return "\n".join(
        f"* {title} by @{author} in https://github.com/{REPO}/pull/{number}"
        for number, author, title in entries
    )


def package_labels(numbers: Iterable[int]) -> dict[int, tuple[str, ...]]:
    # Query exact PR numbers rather than searching, so that the notes remain the
    # single definition of the release. GitHub allows at most 100 labels per PR.
    packages: dict[int, tuple[str, ...]] = {}
    for batch in batched(sorted(numbers), LABEL_BATCH_SIZE):
        fields = " ".join(
            f"pr{number}: pullRequest(number: {number}) "
            "{ labels(first: 100) { nodes { name } } }"
            for number in batch
        )
        query = (
            'query { repository(owner: "scipy", name: "scipy-stubs") { '
            + fields
            + " } }"
        )
        response = github(
            "graphql",
            query=query,
            jq=".data.repository | map_values([.labels.nodes[].name])",
        )
        labels = cast("dict[str, list[str]]", json.loads(response))
        for number in batch:
            packages[number] = tuple(
                sorted(
                    label
                    for label in labels[f"pr{number}"]
                    if PACKAGE_LABEL.fullmatch(label)
                )
            )
    return packages


def strip_package_prefix(title: str, packages: tuple[str, ...]) -> str:
    if not packages:
        return title
    names = "|".join(
        re.escape(name.removesuffix(".*").removeprefix("scipy.")) for name in packages
    )
    # Keep the opening backtick of a dotted name (`stats.foo` -> `foo`), but drop a
    # scope prefix entirely (`stats`: fix -> fix).
    pattern = rf"^(?:(`?)(?:scipy\.)?(?:{names})\.|`?(?:scipy\.)?(?:{names})`?\s*:\s*)"
    return re.sub(pattern, r"\1", title, count=1) or title


def group_by_package(entries: list[Entry], packages: Packages) -> str:
    groups: dict[tuple[str, ...], list[Entry]] = {}
    for entry in entries:
        group = packages[entry.number]
        groups.setdefault(group, []).append(
            entry._replace(title=strip_package_prefix(entry.title, group))
        )
    if not any(groups):
        return format_entries(entries)
    sections: list[str] = []
    for group in sorted(
        groups,
        key=lambda names: (
            any(name.startswith("scipy._") for name in names),
            not names,
            names,
        ),
    ):
        heading = ", ".join(f"`{name}`" for name in group) or "General"
        sections.append(f"#### {heading}\n\n{format_entries(groups[group])}")
    return "\n\n".join(sections)


def format_category(heading: str, entries: list[Entry], packages: Packages) -> str:
    category = strip_gitmoji(heading.removeprefix("### "))
    if category not in KEEP_GITMOJI:
        entries = [
            entry._replace(title=strip_gitmoji(entry.title)) for entry in entries
        ]
    body = (
        group_by_package(entries, packages)
        if category in GROUP_BY_PACKAGE
        else format_entries(entries)
    )
    count = len(entries)
    if count and (
        category in KEEP_GITMOJI
        or (category == "Testing" and count >= COLLAPSE_TESTING_AT)
    ):
        noun = "pull request" if count == 1 else "pull requests"
        body = f"<details>\n<summary>{count} {noun}</summary>\n\n{body}\n\n</details>"
    return f"{heading}\n\n{body}"


def format_contributors(entries: list[Entry], newcomers: set[str]) -> str:
    counts = Counter(
        entry.author for entry in entries if not entry.author.endswith("[bot]")
    )
    if not counts:
        return ""
    lines = ["## Contributors", ""]
    for author, count in sorted(
        counts.items(), key=lambda item: (-item[1], item[0].casefold())
    ):
        lines.append(f"* @{author} ({count}){' +' if author in newcomers else ''}")
    lines += [
        "",
        "Counts are authored pull requests; `+` marks a first-time contributor.",
    ]
    return "\n".join(lines)


def format_notes(notes: str) -> str:
    # The comparison link is not a heading, but must remain outside <details>
    # when GitHub omits the optional New Contributors section.
    body, marker, changelog = notes.partition("\n**Full Changelog**:")
    body, _, new_contributors = body.partition("\n## New Contributors")
    preamble, *blocks = re.split(r"(?=^### )", body, flags=re.MULTILINE)
    sections = [
        (heading, parse_entries(text))
        for heading, _, text in (block.partition("\n") for block in blocks)
    ]
    entries = [entry for _, section in sections for entry in section]
    packages = package_labels({entry.number for entry in entries})
    parts = [
        preamble,
        *(format_category(heading, section, packages) for heading, section in sections),
        format_contributors(entries, set(NEW_CONTRIBUTOR.findall(new_contributors))),
        marker + changelog,
    ]
    return "\n\n".join(part.strip("\n") for part in parts if part.strip("\n")) + "\n"


def dependency(requirements: list[str], name: str) -> Requirement:
    matches = [
        requirement
        for text in requirements
        if (requirement := Requirement(text)).name == name
        and requirement.marker is None
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one unconditional {name} requirement")
    return matches[0]


def version_range(specifiers: SpecifierSet) -> str:
    bounds = {
        specifier.operator: Version(specifier.version) for specifier in specifiers
    }
    if len(specifiers) != len(bounds) or bounds.keys() != {">=", "<"}:
        raise ValueError(f"Cannot describe compatibility bounds: {specifiers}")
    lower, upper = bounds[">="], bounds["<"]
    # Only whole minor-version intervals can be shortened without losing detail.
    # In particular, <3 does not tell us the final supported 2.x minor version.
    minor_versions = all(
        len(v.release) > 1 and v == Version(f"{v.major}.{v.minor}")
        for v in (lower, upper)
    )
    if not minor_versions or lower.major != upper.major or lower.minor >= upper.minor:
        raise ValueError(f"Cannot describe compatibility bounds: {specifiers}")
    first, last = f"{lower.major}.{lower.minor}", f"{upper.major}.{upper.minor - 1}"
    return first if first == last else f"{first}–{last}"


def python_versions(classifiers: list[str]) -> str:
    minors = sorted({
        int(match[1])
        for classifier in classifiers
        if (match := PYTHON_CLASSIFIER.fullmatch(classifier))
    })
    if not minors:
        raise ValueError("No Python minor-version classifiers found")
    if len(minors) > 1 and minors[-1] - minors[0] == len(minors) - 1:
        return f"3.{minors[0]}–3.{minors[-1]}"
    return ", ".join(f"3.{minor}" for minor in minors)


def header(tag: str) -> str:
    # Use tagged metadata even when previewing from the next development version.
    metadata = tomllib.loads(
        run("git", "show", "--end-of-options", f"{tag}:pyproject.toml")
    )
    project = cast("Project", metadata["project"])
    version = Version(project["version"])
    if tag != f"v{project['version']}" or len(version.release) != VERSION_COMPONENTS:
        raise ValueError(f"Tag {tag!r} does not match project version {version}")
    scipy = ".".join(map(str, version.release[:3]))
    # The NumPy range is SciPy's, as scipy-stubs only requires NumPy through optype.
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/scipy/{scipy}/json", timeout=30
    ) as response:
        requirements = cast(
            "list[str]", json.load(response)["info"]["requires_dist"] or []
        )
    numpy = dependency(requirements, "numpy")
    optype = dependency(project["dependencies"], "optype")
    return (
        f"This release targets [SciPy {scipy}]"
        f"(https://github.com/scipy/scipy/releases/tag/v{scipy}) "
        f"and supports Python {python_versions(project['classifiers'])}, "
        f"[NumPy](https://github.com/numpy/numpy) {version_range(numpy.specifier)}, "
        f"and [optype](https://github.com/jorenham/optype) "
        f"{version_range(optype.specifier)}."
    )


def generate_notes(tag: str) -> str:
    introduction = header(tag)
    notes = github(f"repos/{REPO}/releases/generate-notes", jq=".body", tag_name=tag)
    if notes.strip() in {"", "null"}:
        raise ValueError("GitHub returned no release notes")
    return f"{introduction}\n\n{format_notes(notes)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--tag", required=True, help="Existing release tag")
    args = parser.parse_args()
    try:
        notes = generate_notes(args.tag)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"release notes: {error.stderr or error}\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"release notes: {error}\n")
    _ = sys.stdout.write(notes)


if __name__ == "__main__":
    main()
