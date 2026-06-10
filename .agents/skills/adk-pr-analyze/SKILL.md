---
name: adk-pr-analyze
description: Analyze and triage GitHub pull requests for the adk-python repository in a strictly read-only manner. Use this skill to fetch PR details, verify the contributor's CLA, inspect the codebase, evaluate architectural and style alignment, and produce a structured analysis report. Triggers on "/adk-pr-analyze" commands. This skill is strictly read-only and must be used whenever the "/adk-pr-analyze" command is explicitly called.
---

# ADK Pull Request Analysis (adk-pr-analyze)

This skill provides a structured workflow for analyzing, verifying, and triaging GitHub pull requests (PRs) from the `google/adk-python` repository. When instructed to analyze a PR, follow this read-only workflow.

> [!IMPORTANT]
> **Strict Read-Only Constraint**:
> This skill is strictly **read-only**. You MUST NOT modify any code, create new branches, or write any implementation. Your role is only to analyze the PR and output the report. Do NOT use file creation or editing tools (e.g. `write_to_file`, `replace_file_content`, `edit_file`, etc.) in the workspace (except for assigning the PR to yourself if the user approves taking it over).
>
> **Strict Tooling Constraint**:
> Do NOT use `curl`, `wget`, or any HTTP requests to fetch PR/issue content. You MUST parse/extract the numbers and use strictly the custom `fetch_github_issue` / `fetch_github_pr` python tools, the `gh` command, or the helper scripts provided.

---

## Phase 1: Retrieve and Parse the PR & Linked Context (Read-Only)

### Step 1: Extract PR Identifier & Verify CLA Signature (Mandatory Entry Gate)
1. **Identify the PR identifier**: Parse the PR number or URL from the prompt (e.g., `https://github.com/google/adk-python/pull/5885` -> `5885`).
2. **CRITICAL COMPLIANCE GATE - Run Verification Script**:
   * **Rule**: BEFORE doing any further work, diff reading, or analysis, you MUST run the verification helper script in read-only mode to verify the contributor's Contributor License Agreement (CLA) signature:
     ```bash
     .venv/bin/python .agents/skills/adk-pr-analyze/scripts/triage_pr.py <pr_number> --skip-update
     ```
   * **Inspect the Exit Status & Verification Output**:
     * **Exit Code 2 (Refusal)**: The contributor HAS NOT signed the Google CLA. You **MUST absolutely refuse** to perform any analysis, triage, diff-fetching, checking out, or workspace operations. Stop calling tools immediately and print a clear compliance refusal message stating that the Google CLA is not signed.
     * **Exit Code 0 (Success)**: The Google CLA is verified. Proceed.
3. **Parse PR Details from Script Output**: The verification script outputs the complete PR details JSON directly to standard output, wrapped in `[PR_METADATA_JSON]` and `[/PR_METADATA_JSON]` tags. Do NOT write to or read from local cache files, and do NOT make separate network commands to fetch PR details. Parse the JSON metadata directly from the command's stdout:
   * **Key JSON Attributes**: `number`, `title`, `body`, `state`, `url`, `author`, `additions`, `deletions`, `changedFiles`, `labels`, `assignees`, `closingIssuesReferences` (used to locate linked issues).
4. **Locate and Fetch Linked Issue(s)**: Extract linked closing issues directly from the `closingIssuesReferences` array in the parsed JSON metadata from the script's stdout. If any closing issues are linked, fetch their details using the custom python tool `fetch_github_issue(issue_number=<number>)`. This is preferred as it avoids command execution policy issues.
    *If the custom python tool is not available, run the gh command:*
    ```bash
    gh issue view <issue_number> --repo google/adk-python --json number,title,body,state
    ```

### Step 2: Retrieve the Complete Diff
1. **Fetch pull request changes**: Run the `gh pr diff` command to view the actual line-by-line diff of the PR:
   ```bash
   gh pr diff <pr_number> --repo google/adk-python
   ```
2. **Review files modified**: Match the diff segments against existing repository files to identify the target components under review.

---

## Phase 2: Deep Code & Architectural Analysis (Read-Only)

Conduct an extremely thorough review of the changes by examining the diff and analyzing the local codebase. You must address the following three critical dimensions and organize your findings in a premium **PR Analysis Report**:

### 1. Objectives & Impact ("What issue does it fix, or feature does it introduce?")
- **Core Change Summary**: Define what the code modifications do, where they are applied (classes, methods, functions), and the execution flow involved.
- **Problem Resolution**: Confirm how the implementation fixes the linked issue or introduces the target feature.
- **Context Tracing**: Trace the execution flow in the active workspace and explain what modules are impacted by this patch.

### 2. Justification & Value ("Is it a justified issue or a useful feature?")
- **Codebase Verification**: Verify the bug/gap exists in the baseline code by searching the local workspace using `grep_search` and inspecting target files with `view_file`.
- **Aesthetic & Structural Value**: Analyze whether the problem represents a genuine, high-priority bug (e.g., causing hangs, memory leaks, or incorrect API validation) or if the feature adds actual, tangible utility to ADK developers.
- **Alternatives Assessment**: Assess if the PR's solution is the most elegant one, or if there is a cleaner, less intrusive, or more robust alternative pattern (e.g., utilizing an existing helper instead of introducing duplicate logic).
- **Scope & Depth Assessment**:
  - Is the implementation a localized "point fix" for this specific issue, or does it consider wider implications and fix the whole picture?
  - Does it address only the symptom, or does it fix the underlying root cause?

### 3. Architectural & Principle Alignment ("Does it align with ADK's principles?")
Evaluate the implementation against the established architectural, style, and testing guidelines. Use direct file links to code reference examples.

#### A. Public API and Visibility Principles
- **API Stability**: Does the change introduce a breaking change to any public classes, methods, or CLI structures in the `google.adk` namespace? (Breaking changes are unacceptable under Semantic Versioning without an official deprecation cycle).
- **Module and File Naming**: Are new `.py` module files under `src/google/adk/` private by default (prefixed with a leading underscore, e.g., `_my_module.py`)?
- **Explicit Exports**: Are new public symbols explicitly exposed via the package's `__init__.py` using the `__all__` list? Are internal helper classes and on-wire objects kept internal by omitting them from `__all__`?
- **Self-Containment**: Does inside-framework code import from the subsystem's specific module directly, rather than importing from `__init__.py`? (Within ADK, framework-level imports from `__init__.py` are strictly prohibited to avoid circular dependencies and maintain clean encapsulation).
- **Intuitive Naming**: Are public methods and class names concise (e.g., `Runner.run`), while private/internal methods are descriptive (e.g., `_validate_chat_agent_wiring`)?

#### B. Code Quality, Style & Pythonic Conventions
- **Future Annotations**: Does every new or heavily edited python source file include `from __future__ import annotations` immediately after the license header?
- **Strong Typing**: Are type hints used for all function arguments and return values? Is the use of `Any` avoided in favor of precise types, abstract interfaces, or generics?
- **Modern Types**: Is the modern union syntax `X | None` preferred for new code over the legacy `Optional[X]`?
- **Keyword-Only Arguments**: Are swaps and parameter mismatches prevented by enforcing keyword-only arguments using `*` for constructors with multiple attributes?
- **Mutable Defaults**: Are mutable defaults (like `list`, `dict`, `set`) avoided? (Use `None` and instantiate within the method body).
- **Runtime Discrimination**: Does type validation use `isinstance(obj, Type)` instead of `type(obj) is Type` to support subclasses, and is a fallback `else` raise handled?
- **Pydantic v2 Idioms**: For Pydantic models:
  - Do they use `Field()` constraints for simple boundary checks?
  - Do validation rules use `@field_validator` (with `mode='after'`) and `@model_validator`?
  - Is `use_attribute_docstrings=True` configured in the model `ConfigDict` so that docstrings are utilized as field descriptions?
  - Are internal mutable states declared with `PrivateAttr()` and constructor logic mapped in `model_post_init()`?
- **Lazy Logging**: Does logging utilize lazy-evaluated `%`-based templates rather than eager `f-strings`? (e.g., `logging.info("Completed in %s ms", duration)` is correct; `logging.info(f"Completed in {duration} ms")` is a violation).
- **Error Handling**: Are specific exceptions caught with context, avoiding bare `except:` constructs?

#### C. Test Integrity & Verification Quality
- **Behavior-Focused Testing**: Do the new unit or integration tests under `tests/` target public boundaries rather than internal execution states?
- **No Mocking of Core Components**: Are real ADK modules (`BaseNode`, `Event`, `Context`) used, restricting mocking exclusively to external web or network dependencies?
- **Minimal Fixtures & Locality**: Are test helper classes and fixtures kept close to the test functions (defined inline inside the test function when utilized by a single test) to improve discoverability?
- **Structure**: Do tests follow the clean **Arrange-Act-Assert** pattern separated by clear logical blocks?

---

## Report Template

Present the analysis using the following structured format:

```markdown
# 🔍 ADK Pull Request Analysis: PR #<pr_number>
**Title**: <PR Title>
**Author**: @<author_username>
**Status**: `<state>`
**Impact**: `<additions> additions`, `<deletions> deletions` across `<changedFiles> files`

## Executive Summary
1. **Core Objective**: [Briefly summarize what issue is fixed or feature is introduced]
2. **Justification & Value**: [Justified Fix / Valuable Feature / Duplicate / Redundant] - [1-sentence explanation]
3. **Alignment with Principles**: [Pass / Pass with Nits / Major Changes Required] - [1-sentence architecture alignment summary]
4. **Recommendation**: [Approve / Approve with Nits / Push Back (Request Changes)]

<details>
<summary><b>Detailed Findings & Analysis</b></summary>

### 1. Objectives & Impact ("What does it do?")
- **Context & Background**: [Briefly explain the background and the problem it targets. Reference linked Issue #<number> using markdown links if available]
- **Implementation Mechanism**: [Detail precisely which modules are modified and how the execution flow is altered]
- **Affected Surface**: [Highlight any changes to public classes, CLI interfaces, state models, or setup pipelines]

### 2. Justification & Value ("Is it a valid and useful change?")
- **Workspace Verification**:
  - Investigated current workspace files: [file_name.py](file:///absolute/path/to/src/google/adk/...#L123-L145) (using `view_file` / `grep_search`).
  - Found that: [Describe the baseline condition that proves the bug exists or the feature is missing]
- **Value Assessment**: [Explain why this is a good addition. Does it solve a genuine real-world developer problem, improve performance, or prevent resources leaks?]
- **Alternative Approaches**: [Evaluate if there is an alternative implementation path. Did the author choose the cleanest design?]
- **Scope & Depth**: [Point Fix / Systematic Fix] & [Symptom / Root Cause] (Explain whether the implementation targets only the specific symptom/point-issue or addresses the underlying root cause and wider implications).

### 3. Principle & Style Alignment Checklist ("Does it follow rules?")
*   **Public API & Visibility Boundaries**:
    *   *Status*: [Pass / Fail / N/A]
    *   *Analysis*: [Check for breaking changes, private module conventions `_`, and explicit exports in `__init__.py` using `__all__`]
*   **Code Quality, Typing & Conventions**:
    *   *Status*: [Pass / Fail / Nits]
    *   *Analysis*: [Check for `from __future__ import annotations`, absence of `Any`, modern unions `X | None`, lazy logging `%`, specific exception catching, and Pydantic v2 structures]
*   **Robustness & Edge Cases**:
    *   *Status*: [Pass / Fail]
    *   *Analysis*: [Check for type discrimination (`isinstance`), boundaries, null checks, fallback else routes, and thread/async safety]
*   **Test Integrity & Quality**:
    *   *Status*: [Pass / Fail / N/A]
    *   *Analysis*: [Check coverage, testing through public interfaces, minimal inline fixtures, and Arrange-Act-Assert formatting]

</details>
```

---

## Tips & Best Practices
> [!IMPORTANT]
> **Command Sandbox Policy**:
> When running commands via `run_command`, you MUST ONLY use `gh` or `git` commands. Commands like `curl`, `wget`, or direct HTTP network requests are strictly forbidden and will be automatically denied.
> Furthermore, you MUST ONLY use simple commands without special characters (such as `;`, `&`, `|`, `$`, `` ` ``, `<`, `>`, `\n`, `\r`, `(`, `)`, `{`, `}`, `\`). The runner environment runs a security policy that automatically denies any commands containing these characters. Always run clean `gh` or `git` commands directly with arguments, without redirections, command chaining, or shell expansions.

> [!IMPORTANT]
> **Strict Read-Only Enforcement**:
> When executing the `adk-pr-analyze` skill, you MUST NOT use any file modification or editing tools (such as `edit_file`, `replace_file_content`, `write_to_file`, `notebook_edit`, etc.) in the workspace. Your output must strictly be a text markdown report following the template provided, without editing any workspace files or writing/fixing code.

> [!TIP]
> Always verify the baseline behavior in your active workspace before claiming something is a bug or invalid. Reading the current source files using `view_file` gives you full context.

> [!IMPORTANT]
> When presenting code files and lines, always use markdown file links that point directly to the files in the workspace. Make sure the link is clickable and formatted as `[filename.py](file:///absolute/path/to/file#L100-L120)` without surrounding backticks around the brackets.
