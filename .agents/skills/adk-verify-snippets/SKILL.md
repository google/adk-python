---
name: adk-verify-snippets
description: >
  Extracts and verifies the runnability and code coverage of all Python code blocks inside a Markdown file.
  Generates a detailed compilation and execution report.
metadata:
  author: Antigravity
  version: 1.4.0
---

# Verify Markdown Snippets Skill

This skill extracts all ` ```python ` blocks from a Markdown file, executes each
one in a process-isolated environment using the bundled `run.py` harness, and
generates a structured report covering load status, run status, and line
coverage.

> [!CAUTION] **STRICT READ-ONLY CONSTRAINT — READ THIS BEFORE DOING ANYTHING
> ELSE**
>
> This skill is **read-only**. The agent **MUST NOT**: - **Modify** any file in
> the repository (source, test, config, docs, or skill files — including this
> SKILL.md). - **Delete** any file in the repository. - **Create** any new file
> in the repository.
>
> The **only two write operations permitted** are: 1. Writing temporary `.py`
> snippet files to a **system temp directory outside the repository**. 2.
> Writing the final `<filename>_REPORT.md` into the **same directory as the
> source Markdown file**.
>
> If in doubt, do not write. Any other mutation is a violation of this skill's
> contract.

--------------------------------------------------------------------------------

## 🔧 Prerequisites

1.  **ADK Python environment**: Run from the repository root with the `uv`
    virtual environment active.
2.  **`coverage` package** *(optional)*: Enables per-snippet coverage reporting.
    Without it, coverage columns show `—`.

    ```bash
    uv pip install coverage
    ```

3.  **Gemini API key**: Required only for snippets that instantiate an `Agent`,
    `App`, or `Workflow` (which make live Gemini API calls). Set one of:

    ```bash
    export GEMINI_API_KEY="your-key-here"
    # or
    export GOOGLE_API_KEY="your-key-here"
    ```

    If both are set, `GEMINI_API_KEY` takes precedence.

--------------------------------------------------------------------------------

## 🛠️ Usage

```bash
uv run --no-sync python .agents/skills/adk-verify-snippets/scripts/verify_md.py <path_to_markdown_file.md>
```

The script prints progress for each snippet, then writes a report to
**`<filename>_REPORT.md`** in the same directory as the source file and prints
the full path on completion.

**Report contents:** :- **Executive Summary table** — one row per snippet:
preceding heading, Load phase status, Run phase status, coverage %, and error
detail.

-   **Detailed section** — for each snippet: the extracted code block, full
    execution logs (stdout + stderr/traceback), and the coverage report.

--------------------------------------------------------------------------------

## 📝 How Snippets Are Classified

Each ` ```python ` block falls into one of these categories:

### 1. Runnability Test (has a module-level ADK component)

If the snippet assigns a `Workflow`, `Agent`, or `App` to a **module-level
variable**, the runner executes it against the Gemini API.

-   The variable name does not matter — the runner finds it automatically via
    `vars(module)`.
-   For multi-agent snippets, the runner identifies the root agent by excluding
    any agent that appears in another agent's `sub_agents` list.
-   To use a custom test prompt instead of the default `"Test input topic"`,
    define a module-level `test_input` string in the snippet.

If no module-level ADK component is found, the run phase is skipped and the
report shows `➖ NO ADK COMPONENT`.

### 2. Loadability-Only (no ADK component)

The runner verifies the snippet compiles and imports without error. No API call
is made.

### 3. Skipped (annotated with ignore)

Place `<!-- verify-snippets: ignore -->` immediately before the opening
` ```python ` fence to exclude a block entirely. Use this for pseudo-code,
illustrative examples, or snippets that require external setup.

````markdown
<!-- verify-snippets: ignore -->
```python
# pseudo-code — not runnable as-is
my_agent = Agent(model="gemini-ultra-hypothetical", ...)
```
````

The report shows these as `⏭️ SKIPPED`.

--------------------------------------------------------------------------------

## ⚠️ Known Limitations

-   **No shared state between snippets**: Each snippet runs in a fresh
    subprocess with no imports or variables carried over from previous snippets.
    A snippet that depends on code from an earlier block will fail with
    `NameError` or `ImportError`. Make each snippet self-contained, or annotate
    it with `<!-- verify-snippets: ignore -->`.
-   **120-second timeout**: Each snippet is killed after 120 seconds. Annotate
    long-running or blocking snippets with `<!-- verify-snippets: ignore -->`.
-   **Ignore annotation placement**: The `<!-- verify-snippets: ignore -->`
    annotation applies to the next ` ```python ` fence encountered. Blank lines
    between the annotation and the fence are tolerated, but any non-blank line
    (prose or a heading) cancels the annotation.
-   **Bare ` ``` ` closes the block**: The parser closes a Python block on the
    first bare ` ``` ` line (no language tag). A bare ` ``` ` appearing as
    content inside a snippet (e.g. to demonstrate Markdown syntax) will
    prematurely close the block. Annotate such snippets with
    `<!-- verify-snippets: ignore -->`.

--------------------------------------------------------------------------------

## ⚠️ Behavioral Constraints (For AI Agents)

-   **Read-only**: See the caution block at the top. The constraint is absolute.
-   **Report only, do not fix**: The agent MUST NOT rewrite the source Markdown,
    modify code blocks, or generate patches. Present the summary table to the
    user and stop.
-   **Present the summary table verbatim**: After the script completes, read the
    generated `_REPORT.md` and copy the Executive Summary table to the user
    **exactly as written** — same six columns, same order, no renaming or
    dropping: `Snippet | Preceding Heading | Load Phase | Run Phase | Coverage |
    Details`
