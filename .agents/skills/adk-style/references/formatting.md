# Formatting Style Guide

Settings live in `pyproject.toml`; the hooks that apply them live in
`.pre-commit-config.yaml`. Both are the source of truth — this file summarizes
them.

- 2-space indentation, never tabs (`pyink-indentation = 2`).
- 80-character lines (`pyink` `line-length = 80`). Import lines are the
  exception — see the imports reference.
- `pyink` (Google's Black fork) formats Python.
- Quotes follow whichever style already dominates the file
  (`pyink-use-majority-quotes`), so pyink will not rewrite `'x'` to `"x"`.
  Match the file you are editing instead of churning quotes.
- `isort` sorts imports with `profile = "google"`.

## What each hook enforces

Run order matters less than knowing which tool rejected the commit.

| Hook | What it does |
| --- | --- |
| `ruff` | Removes unused imports only (`lint.select = ["F401"]`), auto-fixed, `src/` only. `__init__.py` is exempt because its imports are re-exports. |
| `isort` | Import order and grouping. |
| `pyink` | All other formatting. |
| `addlicense` | Adds the Apache 2.0 header to `.py`/`.sh`. Skipped with a warning if the Go binary is not installed; CI still catches it. |
| `check-new-py-prefix` | New files under `src/google/adk/` need a `_` prefix — see the visibility reference. |
| `compliance-checks` | Logger name, `from __future__ import annotations`, `cli/` import direction, mTLS endpoints. |
| `codespell` | Spelling in code and prose. Add a genuine false positive to `ignore-words-list` in `pyproject.toml`. |
| `pyproject-fmt` | Normalizes `pyproject.toml` itself. |
| `mdformat` | `README.md`, `CONTRIBUTING.md`, and `contributing/**.md` only. |
| `check-yaml`, `end-of-file-fixer`, `trailing-whitespace` | Whitespace and YAML syntax hygiene. |
| `update-constraints` | Regenerates `constraints-3.*.txt` when `pyproject.toml` changes. Needs network access. |

`src/google/adk/cli/browser/`, `src/google/adk/v1/`, and `v1_tests/` are
excluded from every hook.

## Running the formatter

Install the git hook once so formatting happens on commit:

```bash
pre-commit install
```

Then, to check work that is not yet committed:

```bash
# Staged files only (this is what the commit hook runs)
pre-commit run

# Specific files
pre-commit run --files {path/to/file.py}

# Everything
pre-commit run --all-files
```

CI runs this same config, so a clean `pre-commit run --all-files` means the
lint job will pass. Type errors are a separate CI job — see
the typing reference.

Use the `adk-setup` skill to install `pre-commit`, `addlicense`, and the rest
of the toolchain.
