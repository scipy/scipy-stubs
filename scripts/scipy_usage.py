#!/usr/bin/env python3
# ruff: noqa: PLR6301, ERA001

import argparse
import ast
import functools
import importlib
import inspect
import json
import logging
import operator
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Generator
from pathlib import Path
from typing import cast
from typing_extensions import override

# List of repositories from the README
DEFAULT_REPOS = [
    "jorenham/Lmo",
    "librosa/librosa",
    "colour-science/colour",
    "colour-science/colour-visuals",
    "apache/spark",
    "artisan-roaster-scope/artisan",
    "arviz-devs/arviz",
    "danielhrisca/asammdf",
    "paucablop/chemotools",
    "espdev/csaps",
    "nv-legate/cupynumeric",
    "NeilGirdhar/efax",
    "circlemind-ai/fast-graphrag",
    "gerlero/foamlib",
    "freqtrade/freqtrade",
    "huntfx/MouseTracks",
    "mozilla/mozanalysis",
    "ThePornDatabase/namer",
    "neurogym/neurogym",
    "optuna/optuna",
    "bashtage/linearmodels",
    "pandas-dev/pandas",
    "pandas-dev/pandas-stubs",
    "progressivis/progressivis",
    "pysmo/pysmo",
    "radioactivedecay/radioactivedecay",
    "scverse/scanpy",
    "tqec/tqec",
    "live-image-tracking-tools/traccuracy",
    "ultralytics/ultralytics",
    "vega/altair",
    "voc/voctomix",
    "pydata/xarray",
    "jax-ml/jax",
    "unionai-oss/pandera/",
    "JohannesBuchner/imagehash",
    "hydpy-dev/hydpy",
    "static-frame/static-frame",
    "theislab/anndata2ri",
    "gdsfactory/kfactory",
    "nipreps/nireports",
    "UBC-Solar/physics",
    "nirum/jetplot",
    "flika-org/flika",
    # these projects might benefit from adding scipy-stubs as dev dependency
    "scikit-learn/scikit-learn",
    "networkx/networkx",
    "fonttools/fonttools",
    "tensorflow/tensorflow",
    "dmlc/xgboost",
]


IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".tox",
    "venv",
    "env",
    ".venv",
    "build",
    "dist",
    ".eggs",
}

logger = logging.getLogger(__name__)


@functools.cache
def is_scipy_module(name: str) -> bool:
    if not name.startswith("scipy"):
        return False

    try:
        module = importlib.import_module(name)
    except ImportError:
        return False

    return inspect.ismodule(module)


def download_repo(repo_name: str, target_dir: Path) -> Path:
    logger.info("Downloading %s...", repo_name)

    # Try main branch first, then master if main fails
    for branch in ["master", "main"]:
        zip_url = f"https://github.com/{repo_name}/archive/refs/heads/{branch}.zip"
        zip_path = target_dir / f"{repo_name.replace('/', '_')}_{branch}.zip"

        try:
            _, _ = urllib.request.urlretrieve(zip_url, zip_path)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # noqa: PLR2004
                logger.warning(
                    "Branch %s not found for %s, trying next branch", branch, repo_name
                )
                zip_path.unlink(missing_ok=True)
                continue

            raise

        extract_dir = target_dir / repo_name.replace("/", "_")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        zip_path.unlink()

        # Find the extracted directory (should be repo_name-branch)
        extracted_dirs = list(extract_dir.glob(f"*-{branch}"))
        if extracted_dirs:
            return extracted_dirs[0]

        # Fallback: return the first directory
        return next(extract_dir.iterdir())

    raise RuntimeError(f"Could not download {repo_name} - no valid branches found")  # noqa: TRY003


def find_python_files(repo_path: Path) -> Generator[Path]:
    for pattern in ["*.py", "*.pyi"]:
        for file_path in repo_path.rglob(pattern):
            if not any(part in IGNORE_DIRS for part in file_path.parts):
                yield file_path


def _create_counts_data(
    counter: Counter[str], by_repo_dict: dict[str, set[str]]
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for name, usage_count in counter.items():
        project_count = len(by_repo_dict.get(name, set()))
        counts[name] = {"references": usage_count, "projects": project_count}

    return dict(
        sorted(
            counts.items(),
            key=lambda x: (x[1]["projects"], x[1]["references"]),
            reverse=True,
        )
    )


def parse_scipy_usage(file_path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(file_path.read_text(encoding="utf-8", errors="ignore"))

    visitor = ScipyVisitor()
    visitor.visit(tree)

    calls = {name for name in visitor.calls if not is_scipy_module(name)}
    modules = {name for name in visitor.imports if is_scipy_module(name)}

    # for call_name in calls:
    #     parts = call_name.split(".")

    #     for i in range(1, len(parts)):
    #         module = ".".join(parts[: i + 1])
    #         if is_scipy_module(module):
    #             modules.add(module)

    return calls, modules


class ScipyUsageAnalyzer:
    """Analyzes scipy usage patterns in Python code."""

    def __init__(self) -> None:
        self.call_count: Counter[str] = Counter()
        self.call_repos: dict[str, set[str]] = {}
        self.module_count: Counter[str] = Counter()
        self.module_repos: dict[str, set[str]] = {}
        self.repo_stats: dict[str, dict[str, int | dict[str, int]]] = {}

        super().__init__()

    def analyze_repo(self, repo_name: str, repo_path: Path) -> None:
        """Analyze a single repository for scipy usage."""
        logger.info("Analyzing %s...", repo_name)

        repo_calls: Counter[str] = Counter()
        repo_modules: Counter[str] = Counter()

        n_total = 0
        n_relevant = 0

        for py_path in find_python_files(repo_path):
            n_total += 1

            try:
                calls, modules = parse_scipy_usage(py_path)
            except SyntaxError:
                logger.exception("Failed to parse %s", py_path)
                continue

            if calls or modules:
                n_relevant += 1
                repo_calls.update(calls)
                repo_modules.update(modules)

        # Update global counters
        self.call_count.update(repo_calls)
        self.module_count.update(repo_modules)

        # Track which repositories use each call and module
        for call_name in repo_calls:
            if call_name not in self.call_repos:
                self.call_repos[call_name] = set()
            self.call_repos[call_name].add(repo_name)

        for module_name in repo_modules:
            if module_name not in self.module_repos:
                self.module_repos[module_name] = set()
            self.module_repos[module_name].add(repo_name)

        repo_stats: dict[str, int | dict[str, int]] = {
            "files_total": n_total,
            "files_relevant": n_relevant,
            "unique_calls": len(repo_calls),
            "unique_modules": len(repo_modules),
            "modules": dict(repo_modules),
            "calls": dict(repo_calls),
        }

        self.repo_stats[repo_name] = repo_stats

        # Log warning if repository doesn't use scipy
        if n_relevant == 0:
            logger.warning("Repository %s does not use scipy", repo_name)

    def analyze_repositories(
        self, repo_list: list[str], output_file: str | None = None
    ) -> None:
        """Analyze multiple repositories and save results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for repo_name in repo_list:
                repo_path = download_repo(repo_name, temp_path)
                self.analyze_repo(repo_name, repo_path)

        self.generate_report(output_file)

    def _log_scipy_items(
        self, title: str, sorted_data: dict[str, dict[str, int]]
    ) -> None:
        """Log scipy items with usage and project counts."""
        logger.info("%s:", title)
        for i, (name, data) in enumerate(sorted_data.items(), 1):
            logger.info(
                "%2d. %-40s (%d uses, %d projects)",
                i,
                name,
                data["references"],
                data["projects"],
            )

    def generate_report(self, output_file: str | None) -> None:
        calls_with_projects = _create_counts_data(self.call_count, self.call_repos)
        modules_with_projects = _create_counts_data(
            self.module_count, self.module_repos
        )

        # Sort repository details as well
        sorted_repo_stats: dict[str, dict[str, dict[str, int]]] = {}
        for repo_name, stats in self.repo_stats.items():
            sorted_stats = cast("dict[str, dict[str, int]]", stats.copy())

            # Sort all dictionary fields
            field_names = ["calls", "modules"]
            for field_name in field_names:
                if field_name in sorted_stats:
                    field_dict = sorted_stats[field_name]
                    sorted_stats[field_name] = dict(
                        sorted(
                            field_dict.items(), key=operator.itemgetter(1), reverse=True
                        )
                    )
            sorted_repo_stats[repo_name] = sorted_stats

        # Count only repositories that actually use scipy
        repos_using_scipy = sum(
            1
            for stats in self.repo_stats.values()
            if isinstance(stats["files_relevant"], int) and stats["files_relevant"] > 0
        )

        report = {
            "summary": {
                "total_repositories": repos_using_scipy,
                "total_scipy_calls": sum(self.call_count.values()),
                "total_scipy_modules": sum(self.module_count.values()),
                "unique_calls": len(self.call_count),
                "unique_modules": len(self.module_count),
            },
            "calls": calls_with_projects,
            "modules": modules_with_projects,
            "repository_details": sorted_repo_stats,
        }

        # Save to JSON file or stdout
        if output_file:
            _ = Path(output_file).write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            logger.info("Analysis complete! Results saved to %s", output_file)
        else:
            json.dump(report, sys.stdout, indent=2)
            _ = sys.stdout.write("\n")  # Add newline after JSON output

        logger.info("Analyzed %d repositories", len(self.repo_stats))
        logger.info("Found %d unique scipy function calls", len(self.call_count))
        logger.info("Found %d unique scipy modules", len(self.module_count))

        # Print all scipy calls and modules using helper method
        self._log_scipy_items("All scipy function calls found", calls_with_projects)
        self._log_scipy_items("All scipy modules found", modules_with_projects)


class ScipyVisitor(ast.NodeVisitor):
    """AST visitor to find scipy imports and function calls."""

    def __init__(self) -> None:
        super().__init__()
        self.imports: set[str] = set()
        self.calls: set[str] = set()
        self.scipy_aliases: dict[str, str] = {}  # Maps aliases to scipy modules

    @override
    def visit_Import(self, node: ast.Import) -> None:
        """Handle 'import scipy...' statements."""
        for alias in node.names:
            if alias.name.startswith("scipy"):
                import_name = alias.name
                alias_name = alias.asname or alias.name

                self.imports.add(import_name)
                self.scipy_aliases[alias_name] = import_name

        self.generic_visit(node)

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.startswith("scipy"):
            for alias in node.names:
                import_name = f"{node.module}.{alias.name}"
                self.imports.add(f"{node.module}.*")

                if alias.name != "*":
                    alias_name = alias.asname or alias.name
                    self.scipy_aliases[alias_name] = import_name

        self.generic_visit(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        if (
            (call_name := self._get_call_name(node.func))
            and (scipy_call := self._resolve_scipy_call(call_name))
        ):  # fmt: skip
            self.calls.add(scipy_call)  # ty:ignore[possibly-unresolved-reference]  # false positive

        self.generic_visit(node)

    @override
    def visit_Name(self, node: ast.Name) -> None:
        # Only count names that are not in call position (handled by visit_Call)
        # This catches cases like: map(gammaln, x) where gammaln is passed as arg

        if scipy_ref := self._resolve_scipy_call(node.id):
            self.calls.add(scipy_ref)

        self.generic_visit(node)

    # Remove visit_Attribute - we only want actual function calls, not attribute access

    def _get_call_name(self, node: ast.expr) -> str:
        """Extract the full name of a function call."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        return ""

    def _get_attribute_name(self, node: ast.expr) -> str:
        """Extract the full name of an attribute access."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._get_attribute_name(node.value)
            if base:
                return f"{base}.{node.attr}"
            return node.attr
        return ""

    def _resolve_scipy_call(self, call_name: str) -> str | None:
        """Resolve a call name to a scipy function if applicable."""
        # Direct scipy module calls (e.g., scipy.linalg.norm)
        if call_name.startswith("scipy."):
            return call_name

        # Check if it matches any known scipy aliases
        for alias, scipy_module in self.scipy_aliases.items():
            if call_name.startswith(alias + "."):
                # Replace the alias with the actual scipy module
                return call_name.replace(alias, scipy_module, 1)
            if call_name == alias:
                # Direct call to an imported function
                return scipy_module

        return None


def main() -> None:
    """Main function to run the analysis."""
    # Configure logging to stderr so it doesn't interfere with JSON output to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Analyze scipy usage in GitHub repositories"
    )
    _ = parser.add_argument(
        "--repos", nargs="*", help="List of repositories (owner/repo format)"
    )
    _ = parser.add_argument(
        "--output", help="Output file for results (default: stdout)"
    )
    _ = parser.add_argument(
        "--sample", type=int, help="Analyze only a sample of N repositories"
    )

    args = parser.parse_args()

    # Use provided repos or default list
    repos = args.repos or DEFAULT_REPOS

    # Use sample if specified
    if args.sample:
        repos = repos[: args.sample]
        logger.info("Analyzing sample of %d repositories", len(repos))

    # Run the analysis
    analyzer = ScipyUsageAnalyzer()
    analyzer.analyze_repositories(repos, args.output)


if __name__ == "__main__":
    main()
