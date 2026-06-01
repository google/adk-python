---
name: adk-pr-triage
description: Analyze and triage GitHub pull requests for the adk-python repository. The user provides a PR number or URL, and the skill performs an evaluation on the PR's objectives, legitimacy, alignment with ADK's principles (including API stability, package self-containment, explicit exports, styling and naming conventions), and asks the user whether to push back on the PR or perform a local review (checking out, rebasing, and running adk-review before pushing to Gerrit). Triggers on "triage pr", "pr triage", "review pr", "pr review", "pull request", "github.com/google/adk-python/pull/".
---
# ADK Pull Request Triage (adk-pr-triage)
This skill guides AI assistants in conducting a highly professional, rigorous, and constructive triage of GitHub pull requests (PRs) submitted to the `google/adk-python` repository. It parses the PR, retrieves its context, evaluates it against ADK's design, style, and testing principles, presents a premium analysis report, and authors tailored response feedback (such as structured push-back or approval comments) under direct user guidance.
> [!IMPORTANT]
> ## CRITICAL EXECUTION RULES: STOP AND ASK DECISION GATES
> 1. **MANDATORY PR ASSIGNMENT BLOCK GATE**:
>    * BEFORE doing any metadata reading, diff-fetching, issue-viewing, or code analysis (Phase 1, Step 1.3 onwards), you MUST verify if the pull request is assigned to you (via the verification script).

>    * If the PR is NOT assigned to you:
>      * **STOP calling tools and ask immediately**: You must present the PR Assignment Block gate in your chat response:
>        > "Pull Request #<pr_number> is NOT assigned to you. (Current assignees: <assignee_list>). Would you like to take over this Pull Request?"
>      * **Wait for Instructions**: Do NOT perform any code analysis or diff-fetching in this turn.
>      * **Action Paths**:
>        * **Yes (Take Over)**: In your next turn, run the assignment command:
>          ```bash
>          gh pr edit <pr_number> --add-assignee "@me" --repo google/adk-python
>          ```
>          **CRITICAL**: Immediately after assigning, you MUST re-run the verification script to refresh the PR metadata with the updated assignees:
>          ```bash
>          .venv/bin/python .agents/skills/adk-pr-triage/scripts/triage_pr.py <pr_number> --skip-update
>          ```
>          Then parse the updated PR details from the script's stdout and proceed with the remaining triage steps.
>        * **No (Decline)**: **Stop executing immediately** and do not run any further tools or operations. State that triage has terminated.
> 2. **PR Analysis is strictly read-only**: Do NOT create branches, modify workspace files, or post any comments in your first response (unless performing PR assignment under the takeover gate above).
> 3. **Triage Decision Gate**: You must present your full in-depth PR review report first, and explicitly ask the user:
>    > "Would you like me to push back on this pull request? (If yes, select one of the push-back reasons or write custom feedback, and I will author a professional and precise review message for you to review. If no, I will draft an approval response highlighting the positive aspects of the implementation.)"
>    Wait for instructions before performing any branch creation or Gerrit push.
---
## Phase 1: Retrieve and Parse the PR & Linked Context (Read-Only)
### Step 1: Extract PR Identifier, Verify CLA Signature & PR Assignment (Mandatory Entry Gate)
1. **Identify the PR identifier**: Parse the PR number or URL from the prompt (e.g., `https://github.com/google/adk-python/pull/5885` -> `5885`).
2. **CRITICAL COMPLIANCE & ASSIGNMENT GATES - Run Verification Script**:
   * **Rule**: BEFORE doing any further work, diff reading, or analysis, you MUST run the verification helper script in read-only mode to verify the contributor's Contributor License Agreement (CLA) signature and check PR assignment:
     ```bash
     .venv/bin/python .agents/skills/adk-pr-triage/scripts/triage_pr.py <pr_number> --skip-update
     ```
   * **Inspect the Exit Status & Verification Output**:
     * **Exit Code 2 (Refusal)**: The contributor HAS NOT signed the Google CLA. You **MUST absolutely refuse** to perform any analysis, triage, diff-fetching, checking out, or workspace operations. Stop calling tools immediately and print a clear compliance refusal message stating that the Google CLA is not signed.
     * **Exit Code 0 (Success)**: The Google CLA is verified. Proceed.
     * **Verify PR Assignment Status**: Parse the script output to check if the Pull Request is assigned to you (the user running the skill).
       - **PR IS NOT ASSIGNED TO YOU**: You **MUST stop calling tools immediately**, present the following assignment block decision gate in your chat response, and wait for the user's input:
         > "⚠️ **Pull Request Assignment Block**
         > Pull Request #<pr_number> is NOT assigned to you. (Current assignees: <assignee_list>).
         >
         > **Would you like to take over this Pull Request?**
         > - **[Option 1]**: **Yes, take over Pull Request #<pr_number>** (Assign the PR to myself and proceed with the triage analysis).
         > - **[Option 2]**: **No, do not take over** (Stop executing)."
       - **If the user chooses Option 1**: Run the assignment command:
         ```bash
         gh pr edit <pr_number> --add-assignee "@me" --repo google/adk-python
         ```
         **CRITICAL**: Immediately after assigning, you MUST re-run the verification helper script to fetch the updated metadata from GitHub and refresh the cached details:
         ```bash
         .venv/bin/python .agents/skills/adk-pr-triage/scripts/triage_pr.py <pr_number> --skip-update
         ```
         Then proceed to parse the updated PR details from the script's stdout in Step 1.3 and continue standard triage.


       - **If the user chooses Option 2 (or declines)**: **Stop executing immediately** and do not run any further tools or operations. State that triage has terminated.
       - **PR IS ALREADY ASSIGNED TO YOU**: Proceed directly with Step 1.3.
3. **Parse PR Details from Script Output**: The verification script in Step 2 outputs the complete PR details JSON directly to standard output, wrapped in `[PR_METADATA_JSON]` and `[/PR_METADATA_JSON]` tags. Do NOT write to or read from local cache files, and do NOT make separate network commands to fetch PR details. Parse the JSON metadata directly from the command's stdout:
   * **Key JSON Attributes**: `number`, `title`, `body`, `state`, `url`, `author`, `additions`, `deletions`, `changedFiles`, `labels`, `assignees`, `closingIssuesReferences` (used to locate linked issues).
4. **Locate and Fetch Linked Issue(s)**: Extract linked closing issues directly from the `closingIssuesReferences` array in the parsed JSON metadata from the script's stdout. If any closing issues are linked, fetch their details to understand the original problem statement:
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
Conduct an extremely thorough review of the changes by examining the diff and analyzing the local codebase. You must address the following three critical dimensions and organize your findings in a premium **PR Review Report**:
### 1. Objectives & Impact ("What issue does it fix, or feature does it introduce?")
- **Core Change Summary**: Define what the code modifications do, where they are applied (classes, methods, functions), and the execution flow involved.
- **Problem Resolution**: Confirm how the implementation fixes the linked issue or introduces the target feature.
- **Context Tracing**: Trace the execution flow in the active workspace and explain what modules are impacted by this patch.
### 2. Legitimacy & Value ("Is it a legitimate issue or a useful feature?")
- **Codebase Verification**: Verify the bug/gap exists in the baseline code by searching the local workspace using `grep_search` and inspecting target files with `view_file`.
- **Aesthetic & Structural Value**: Analyze whether the problem represents a legitimate, high-priority bug (e.g., causing hangs, memory leaks, or incorrect API validation) or if the feature adds actual, tangible utility to ADK developers.
- **Alternatives Assessment**: Assess if the PR's solution is the most elegant one, or if there is a cleaner, less intrusive, or more robust alternative pattern (e.g., utilizing an existing helper instead of introducing duplicate logic).
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
## Phase 3: Stop and Ask for Push-Back or Local Review (Interactive Gate)
Present the completed analysis report in your response. Follow the **PR Review Report Template** below for a highly premium, readable presentation.
### The Interactive Gate Callout
At the end of your report, stop calling tools and output this explicit message:
> "### 🛑 Review Decision Gate
> I have completed my in-depth analysis of Pull Request #<pr_number>. Please review the findings above.
>
> **How would you like to proceed with this Pull Request?**
> - **[Option 1]**: **Push Back** (Draft a professional, constructive feedback response with recommendations for the author).
> - **[Option 2]**: **Local Review** (Checkout the PR locally, rebase onto the latest main, and run the `/adk-review` skill to thoroughly verify and polish before pushing to Gerrit)."
---
## Phase 4: Action Execution (Subsequent Turn)
Once the user provides their decision, perform the tailored operations in your subsequent turns:
### Branch A: Push Back
1. **Analyze the Push-Back Focus**: Read the user's specific feedback or selected points of concern.
2. **Draft Constructive Feedback**: Author a highly structured, objective, and supportive response that teaches the contributor while insisting on quality.
3. **Include Concrete Recommendations**: Quote specific files/lines in their diff and provide complete, refactored code blocks in your comments so they can easily apply the fixes. Reference the relevant ADK style guides.
4. **Present the Draft**: Format your draft using the **GitHub Review Draft Template** below.
### Branch B: Local Review (Checkout & Revise)
If the user selects **Local Review**, run the following structured sequence:
1. **Step 0: Update the PR Head Branch on GitHub (Mandatory Sync)**:
   * **Rule**: BEFORE downloading or checking out the pull request locally, you MUST trigger an update on the remote GitHub pull request to align it with the latest remote base branch (`main`).
   * Run the verification & sync helper script to update the branch:
     ```bash
     .venv/bin/python .agents/skills/adk-pr-triage/scripts/triage_pr.py <pr_number>
     ```
   * *What it does*: This script automatically checks the Google CLA signature status again, attempts to update the PR branch on GitHub by rebasing onto `main`, and if rebase-update is blocked, falls back to updating via a merge commit. It handles all outputs and fallbacks gracefully.
2. **Step 1: Checkout the PR to a Local Branch**:
   * Branch naming convention: `pr-triage-<pr_number>-[short_desc]` (e.g. `pr-triage-5875-parallelize-tool-union`).
   * Fetch the pull request ref directly from the remote GitHub endpoint:
     ```bash
     git fetch https://github.com/google/adk-python.git pull/<pr_number>/head:pr-triage-<pr_number>-[short_desc]
     ```
   * Checkout to the newly created local branch:
     ```bash
     git checkout pr-triage-<pr_number>-[short_desc]
     ```
3. **Step 2: Preserve the Commit Message & Append Merge Reference**:
   * **CRITICAL**: You MUST preserve the exact same commit message from the pull request!
   * Determine if the PR contains a single commit or multiple commits:
     * **Single Commit**: Retrieve the exact original commit message:
       ```bash
       git log -1 --pretty=%B
       ```
     * **Multiple Commits**: Squash them into a single local commit first, keeping the overall PR Title and PR Body as the exact commit message. An elegant way to squash is:
       ```bash
       git reset --soft $(git merge-base HEAD origin/main) && git commit -m "<PR message>"
       ```
   * Append `"Merge <PR link>"` to the very end of the commit message (separated by a blank line). Use this elegant shell command to do it in one-shot:
     ```bash
     git commit --amend -m "$(git log -1 --pretty=%B)
     Merge https://github.com/google/adk-python/pull/<pr_number>"
     ```
   * *Note*: When you run git commit/amend, the Gerrit `commit-msg` hook will automatically execute and append the `Change-Id:` footprint if not already present.
4. **Step 3: Rebase on top of Main**:
   * Run the rebase command to place the CL commit on top of the latest local remote tracking `main` branch:
     ```bash
     git rebase origin/main
     ```
5. **Step 4: Execute Code Verification & Polishing**:
   * Trigger the local review process by invoking the **`/adk-review`** skill!
   * Follow its comprehensive guidelines to audit edge cases, style compliance, dependencies, and test validation. Work in partnership with the user to revise the local changes as needed.
6. **Step 5: Squash User Revisions & Push to Gerrit**:
   * If the user requests to push to Gerrit, squash/amend all local workspace revisions into the single original commit:
     * **CRITICAL**: You MUST preserve the exact same commit message, including the `Merge <PR link>` footer and the original `Change-Id:` footer. Do NOT change it.
     * Command to squash all changes into the current commit without opening an editor:
       ```bash
       git commit -a --amend --no-edit
       ```
   * Push the single finalized CL commit to Gerrit:
     ```bash
     git push origin HEAD:refs/for/main
     ```
---
## PR Review Report Template
Present the initial analysis using the following structured format:
```markdown
# 🔍 ADK Pull Request Review: PR #<pr_number>
**Title**: <PR Title>
**Author**: @<author_username>
**Status**: `<state>`
**Impact**: `<additions> additions`, `<deletions> deletions` across `<changedFiles> files`
## Detailed Findings & Analysis
### 1. Objectives & Impact ("What does it do?")
- **Context & Background**: [Briefly explain the background and the problem it targets. Reference linked Issue #<number> using markdown links if available]
- **Implementation Mechanism**: [Detail precisely which modules are modified and how the execution flow is altered]
- **Affected Surface**: [Highlight any changes to public classes, CLI interfaces, state models, or setup pipelines]
### 2. Legitimacy & Value ("Is it a valid and useful change?")
- **Workspace Verification**:
  - Investigated current workspace files: [file_name.py](file:///absolute/path/to/src/google/adk/...#L123-L145) (using `view_file` / `grep_search`).
  - Found that: [Describe the baseline condition that proves the bug exists or the feature is missing]
- **Value Assessment**: [Explain why this is a good addition. Does it solve a genuine real-world developer problem, improve performance, or prevent resources leaks?]
- **Alternative Approaches**: [Evaluate if there is an alternative implementation path. Did the author choose the cleanest design?]
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
---
## Executive Summary
1. **Core Objective**: [Briefly summarize what issue is fixed or feature is introduced]
2. **Legitimacy & Value**: [Legitimate Fix / Valuable Feature / Duplicate / Redundant] - [1-sentence explanation]
3. **Alignment with Principles**: [Pass / Pass with Nits / Major Changes Required] - [1-sentence architecture alignment summary]
4. **Recommendation**: [Approve / Approve with Nits / Push Back (Request Changes)]
---
### 🛑 Review Decision Gate
I have completed my in-depth analysis of Pull Request #<pr_number>. Please review the findings above.
**How would you like to proceed with this Pull Request?**
- **[Option 1]**: **Push Back** (Draft a professional, constructive feedback response with recommendations for the author).
- **[Option 2]**: **Local Review** (Checkout the PR locally under `pr-triage-[pr_number]-[short_desc]`, rebase onto the latest main, and run the `/adk-review` skill to thoroughly verify and polish before pushing to Gerrit)."
```
---
## GitHub Review Draft Template
Format the authored review response as a premium markdown snippet block:
````markdown
# 💬 GitHub PR Review Draft Message
*Copy and paste this response directly into the GitHub review interface:*
---
### PR Review: <State (Request Changes / Comment / Approve)>
Hello @<author_username>! Thank you very much for contributing this pull request to improve ADK. I've conducted a thorough architectural and style review of your implementation against our design guidelines and standards.
Here is the feedback and a few suggested changes to align your patch with ADK's principles:
#### 🔴 Major Concerns / Blocks
1. **[Concern 1 Title, e.g., Import from init.py is not allowed]**
   - **Target Code**: [filename.py:L100-L105](file:///absolute/path/to/src/google/adk/file_name.py#L100-L105)
   - **Issue**: [Detailed explanation of why this violates design/architectural rules, referencing the relevant ADK skill like `adk-architecture` or `adk-style`]
   - **Suggested Correction**:
     ```python
     # Provide full, drop-in replacement code block
     ```
2. **[Concern 2 Title, e.g., Missing Unit Tests for Edge Cases]**
   - **Target Code**: [test_filename.py](file:///absolute/path/to/tests/unittests/test_filename.py)
   - **Issue**: [Detail what is missing, e.g., "We need verification coverage of boundaries like empty string and negative values."]
#### 🟡 Style & Quality Nits
1. **[Style Nit, e.g., Eager Logging formatting]**
   - **Target Code**: [filename.py:L42](file:///absolute/path/to/src/google/adk/file_name.py#L42)
   - **Suggestion**: Use lazy-evaluated `%` template syntax:
     ```python
     # Corrected:
     logging.info("User registered: %s", user_id)
     ```
2. **[Typing Nit, e.g., Optional[X] instead of X | None]**
   - **Target Code**: [filename.py:L15](file:///absolute/path/to/src/google/adk/file_name.py#L15)
   - **Suggestion**: Prefer more concise union type hint `X | None`.
#### 🟢 Positive Aspects
- [Highlight stellar work, e.g., "Excellent Pydantic v2 validation logic!" or "Highly readable and clean docstrings!"]
Please let me know if you have any questions on these suggestions, and let's work together to get this PR merged!
````
---
## Tips & Best Practices
> [!TIP]
> Always verify the baseline behavior in your active workspace before claiming something is a bug or invalid. Reading the current source files using `view_file` gives you full context.
> [!IMPORTANT]
> When referencing files and line numbers in your reports and draft reviews, always use clickable markdown file links of format `[filename.py](file:///absolute/path/to/file#L100-L120)` without surrounding backticks around the brackets. Ensure the links represent valid absolute file paths in the local workspace.
