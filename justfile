set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

stubs := "scipy-stubs scripts"

# list all recipes
_default:
    @just --list

# run all checks: lint, typecheck, typetest, and stubtest
check: lint typecheck typetest stubtest

# ruff, dprint, repo-review, and typos
lint:
    uv run ruff check --show-fixes
    uv run ruff format --check
    uv run dprint check --incremental=false
    uv run repo-review --show=err .
    uvx typos

# auto-fix lint and formatting issues
fix:
    uv run ruff check --fix --show-fixes
    uv run ruff format
    uv run dprint fmt

# run all static type-checkers on the stubs and scripts
typecheck: ty zuban pyrefly mypy pyright

ty *paths="":
    uv run ty check --error-on-warning --exclude tests {{ paths }}

zuban *paths="":
    uv run zuban check --exclude '^tests/' {{ paths }}

pyrefly *paths=stubs:
    uv run pyrefly check {{ paths }}

mypy *paths="scipy-stubs":
    uv run mypy {{ paths }}

pyright *paths=stubs:
    uv run basedpyright {{ paths }}

# type-check the tests with pyrefly, mypy, and basedpyright
typetest: (pyrefly "tests") (mypy "tests") (pyright "tests")

# validate the stubs against the scipy runtime
stubtest:
    uv run --no-editable --reinstall-package=scipy-stubs \
        stubtest --ignore-disjoint-bases --allowlist=.mypyignore scipy

# check stub completeness
coverage:
    uv run pyrefly coverage check --public-only --fail-under=99.9 scipy-stubs
    uv run scripts/unstubbed_modules.py

# report incorrect or missing default values in the stubs
stubdefault:
    uv run stubdefaulter --packages=. --exit-zero --check

# remove cache directories and compiled bytecode
clean:
    uvx pyclean . --debris all
