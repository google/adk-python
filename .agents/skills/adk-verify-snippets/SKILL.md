---
name: adk-verify-snippets
description: >-
  Checks that every Python code block in a Markdown file actually compiles and
  runs, by extracting each block to a temporary file, executing it in an
  isolated subprocess, and writing a pass/fail report with per-snippet
  coverage. Use when the user asks to verify, test, or validate the code
  samples in a README, a guide, or a documentation page; wants to know which
  snippets in a Markdown file are broken or out of date; or asks for a snippet
  verification report. Don't use for running the project's test suite (run
  pytest directly), for checking code style or formatting (use `adk-style`),
  or for authoring a new runnable sample agent (use `adk-sample-creator`).
---

# Verify Markdown Snippets

Extracts every ` ```python ` block from a Markdown file, runs each one in its
own subprocess via the bundled `run.py` harness, and writes a report covering
load status, run status, and line coverage per snippet.

## Read-only contract

Verifying a doc must never change the doc. Do not create, modify, or delete any
file in the repository — including the Markdown being verified, its code
blocks, and this SKILL.md. Report the failures; do not fix them and do not
offer patches.

The script performs the only two writes that happen: temporary `.py` files in a
system temp directory outside the repository (removed when it exits), and the
report beside the source Markdown file.

## Prerequisites

1. An ADK development environment — run from the repository root with the `uv`
   virtual environment active (see the `adk-setup` skill).

2. `coverage`, optional. It is not a declared project dependency, so install it
   explicitly; without it the Coverage column shows `—`.

   ```bash
   uv pip install coverage
   ```

3. A Gemini API key, needed only for snippets that build an `Agent`, `App`, or
   `Workflow` — those are executed against the live API.

   ```bash
   export GEMINI_API_KEY="{your_key}"
   # or
   export GOOGLE_API_KEY="{your_key}"
   ```

   If both are set the harness drops `GOOGLE_API_KEY`, so `GEMINI_API_KEY` wins.

## Usage

```bash
uv run --no-sync python .agents/skills/adk-verify-snippets/scripts/verify_md.py {path_to_markdown_file}
```

The script prints per-snippet progress, then writes the report beside the source
file and prints its full path.

The report filename is the source file's stem lowercased with everything except
`[a-z0-9_]` stripped, plus `_REPORT.md`. `Workflow-Guide.md` therefore produces
`workflowguide_REPORT.md`, not `Workflow-Guide_REPORT.md` — read the path the
script prints rather than reconstructing it.

The report contains an Executive Summary table with one row per snippet, then a
detailed section per snippet holding the code block, the execution logs
(stdout plus stderr/traceback), and the coverage output.

## How each snippet is classified

### Runnable — has a module-level ADK component

If the snippet assigns a `Workflow`, `Agent`, or `App` to a module-level
variable, the harness executes it against the Gemini API.

-   The variable name does not matter; the harness scans `vars(module)`.
-   Precedence is `Workflow`, then root `Agent`, then `App`. A `Workflow`
    anywhere in the snippet wins over any agent in it.
-   The root agent is the first agent that appears in no other agent's
    `sub_agents`, so multi-agent snippets resolve correctly whatever order the
    agents are defined in.
-   An `App` must have been constructed with a `root_agent` or the run fails.
-   The prompt sent is `"Test input topic"`. Override it by defining a
    module-level `test_input` string in the snippet.

### Load-only — no ADK component

The harness confirms the snippet compiles and imports, and makes no API call.
The report shows `➖ NO ADK COMPONENT`.

### Skipped — annotated with ignore

Put `<!-- verify-snippets: ignore -->` alone on a line immediately before the
opening ` ```python ` fence to exclude a block. Use it for pseudo-code,
illustrative fragments, and snippets that need external setup. The report shows
`⏭️ SKIPPED`.

````markdown
<!-- verify-snippets: ignore -->
```python
# pseudo-code — not runnable as-is
my_agent = Agent(model="gemini-ultra-hypothetical", ...)
```
````

## Limitations that make correct snippets report as broken

Annotate with `<!-- verify-snippets: ignore -->` instead of editing the doc to
work around any of these.

-   **No shared state between snippets.** Each snippet runs in a fresh
    subprocess, so one that relies on an import or variable from an earlier
    block fails with `NameError` or `ImportError`.
-   **120-second timeout** per snippet, after which the process is killed and
    the snippet reports as a run failure.
-   **Annotation placement.** The annotation applies to the next ` ```python `
    fence. Blank lines between the two are fine; any prose line or heading
    between them cancels it.
-   **A bare ` ``` ` closes the block.** The parser closes a Python block at the
    first fence carrying no language tag, so a bare fence used as content inside
    a snippet truncates it. A tagged fence (for example ` ```bash `) is kept as
    literal content and is safe.
-   **Module-level `asyncio.run()`** collides with the harness's own event loop
    and reports as a run failure. Snippets should keep top-level async calls
    behind `if __name__ == "__main__":`.

## Reporting back to the user

Read the generated report and copy the Executive Summary table across exactly as
written — same six columns, same order, nothing renamed or dropped:
`Snippet | Preceding Heading | Load Phase | Run Phase | Coverage | Details`.
Present it and stop.
