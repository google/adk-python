---
name: adk-review
description: Reviews all local changes in the repository for errors, styling compliance, unintended outcomes, and necessary documentation/test/sample updates. Generates a report and assists in fixing identified issues on-demand. Triggers on "adk-review", "review changes", "pr review", "check code style", "verify changes".
---

# ADK Change Reviewer (adk-review)

This skill guides AI assistants in performing a comprehensive, rigorous review of local repository changes before they are committed or submitted. It evaluates code correctness, style guidelines, architectural impact, and checks if associated tests, samples, and documentation need updates. It generates a detailed report and, upon explicit user request, assists in automatically fixing the identified issues.

> [!NOTE]
> Always read this skill and follow its steps when asked to review local changes or before finalizing a PR/commit.

---

## Review Checklist Dimensions

### 1. Code Correctness & Errors
- **Syntax & Types**: Ensure the code is free of syntax errors and conforms to strong typing guidelines. Avoid using `Any`, and prefer specific/abstract types. Use `X | None` instead of `Optional[X]`.
- **Imports**: Verify there are no circular imports. Ensure absolute imports are used where appropriate.
- **Exception Handling**: Avoid bare `except:`. Always catch specific exceptions and log them properly with context.
- **Visibility**: Ensure internal modules and package-private attributes use proper naming (e.g., prefixed with `_`) per ADK rules.
- **Edge Cases & Defensive Programming**:
  - **Type & Attribute Discrimination**: Explicitly verify an object's type (e.g., using `isinstance`) before checking type-specific or custom attributes (e.g., checking if a node is an `LlmAgent` before inspecting its `mode`), avoiding errors on unexpected types.
  - **Boundary and Null Conditions**: Ensure robust handling for boundary conditions and null values (e.g., `None`, empty collections, zero, or empty strings) using validation or fallback defaults.
  - **Preconditions & Invariants**: Validate that preconditions and state invariants are checked before performing core logic.

### 2. Style and Convention Compliance
- **ADK Style Guide**: Cross-reference all code changes with the guidelines in the `adk-style` skill (including Pydantic v2 patterns, lazy logging evaluation, and file structure).
- **Pre-commit Hooks**: Ensure changed files are formatted and linted. Remind the user to run `pre-commit run --files <files>` if hooks like `isort`, `pyink`, `addlicense`, or `mdformat` are not configured automatically.

### 3. Architectural Integrity & Unintended Outcomes
- **Public API Stability**: Verify whether changes modify, remove, or restrict public-facing interfaces, classes, methods, argument lists, or CLI structures (e.g., in the public package namespaces under `src/google/adk/`). Breaking changes are unacceptable without a formal deprecation cycle under Semantic Versioning.
- **Execution & Resumption**: If changing workflows, nodes, or state management, ensure compatibility with the ADK 2.0 event execution lifecycle and session resumption (HITL/checkpoints).
- **Concurrency & Safety**: Check for race conditions or resource leaks. Ensure long-running or shared resources (like plugins, exporters, and connections) are closed/disposed of safely.

### 4. Documentation Impact (`docs/design` and `docs/guides`)
- **Design & Architecture**: Determine if the change updates a core design contract. If so, check if design docs under `docs/design/` require updates or new documents need to be written.
- **Guides**: If the changes introduce a new feature or change a public API/workflow pattern, check if the guides under `docs/guides/` need updates.

### 5. Sample Compatibility & Updates
- **Sample Integrity**: Verify if existing samples under `contributing/samples/` are affected by the change.
- **New Samples**: If the changes introduce a key new capability, assess whether a new sample should be added to demonstrate the feature (following `adk-sample-creator` conventions).

### 6. Test Coverage & Quality
- **Coverage**: Ensure that all modified or new code paths have corresponding unit or integration tests under `tests/`.
- **ADK Test Rules**: Ensure test implementations adhere to the 9 rules in the `adk-style` testing reference (e.g., using deterministic IDs, event normalization, and clean up utilities).

---

## Execution Workflow

When the `adk-review` skill is triggered, you MUST execute the following steps:

### Step 1: Retrieve Local Changes
Run `git status` and `git diff` to identify exactly which files have been modified, added, or deleted.

### Step 2: Perform the Multi-Dimensional Review
Analyze the retrieved diffs file-by-file against the six dimensions in the Checklist. Identify any errors, deviations, or missing files (such as docs, tests, or samples).

### Step 3: Generate and Present a Review Report
Generate a clear, beautifully formatted Markdown report categorized by priority:
- 🔴 **Critical Errors / Bugs**: Syntax, type safety violations, race conditions, or resource leaks.
- 🟡 **Style & Conventions**: Lints, formatting issues, non-lazy logging, or typing mismatches.
- 🔵 **Documentation, Tests, & Samples**: Missing or stale test coverage, design docs, or user guides.

Include the specific filename and line number/context for each finding.

### Step 4: Present Findings and Stop
Stop execution here. Do **NOT** call any code editing tools or modify the codebase automatically. Present the generated review report clearly to the user, highlighting key takeaways, and stop.

Do **NOT** ask the user if they want you to fix the issues, and do **NOT** offer interactive fixing options by default. Simply stop and wait for the user to explicitly command or ask you to fix the changes.

### Step 5 (Optional): Implement Authorized Fixes & Verify
If, and only if, the user explicitly instructs or requests you to apply a fix for some or all of the identified findings:
1. Perform the necessary edits using precise code editing tools. Ensure all fixes strictly comply with the established `adk-style` and `adk-architecture` rules.
2. Verify correctness by running associated unit and integration tests (e.g., via `pytest` or pre-commit hooks) before concluding.
