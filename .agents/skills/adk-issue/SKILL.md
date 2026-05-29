---
name: adk-issue
description: Analyze and triage GitHub issues for the adk-python repository. Use this skill when the user provides an issue number or link to investigate whether the issue is legitimate, whether it should be fixed, and if there is an existing PR addressing it. Triggers on "analyze issue", "issue #", "github issue", "github.com/google/adk-python/issues/".
---

# ADK Issue Analyzer & Triager

This skill provides a structured workflow for analyzing, verifying, and triaging GitHub issues from the `google/adk-python` repository. When a user provides a GitHub issue number or link, use this skill to perform deep investigation and report your findings.

> [!IMPORTANT]
> ## CRITICAL EXECUTION RULE: STOP AFTER STEP 2
> * **Phase 1 (Triage & Report) is strictly read-only**: Do NOT modify any code, create new branches, or write any implementation in your first response.
> * **STOP and ask**: You must present the analysis report first and explicitly ask the user:
>   > "Would you like me to create and implement a fix for this issue in the workspace? (Note: The changes and tests will be created in a new branch but NOT committed, so you can review and iterate on them.)"
> * **Wait for Approval**: Only proceed with Phase 2 (Step 3: Implementation) in a subsequent turn *after* the user explicitly approves the recommendation.

## Phase 1: Triage and Analysis (Read-Only)

### Step 1: Retrieve and Parse the Issue
1. **Extract the issue number**: Parse the number from the link or prompt (e.g., `https://github.com/google/adk-python/issues/5882` -> `5882`).
2. **Fetch issue details**: Use the `gh` CLI tool to fetch issue details in JSON format:
   ```bash
   gh issue view <issue_number> --repo google/adk-python --json number,title,body,state,labels,comments,assignees,createdAt,closedAt
   ```
   *If the `gh` CLI is not available or errors out, use `read_url_content` to fetch the public GitHub issue page:*
   ```
   https://github.com/google/adk-python/issues/<issue_number>
   ```

---

### Step 2: Deep Investigation & Analysis
Address the following four critical questions and present your findings in a structured, premium report.

#### 1. What is broken?
Explain the root cause of the issue or failure:
- **Trace the execution flow**: Identify which components, classes, or functions are malfunctioning.
- **Pinpoint the bug**: Detail why the system is behaving incorrectly and where the failure originates (e.g. incorrect logic, missing configuration, unhandled states).

#### 2. Is the issue legitimate?
Inspect the codebase to confirm if the issue represents a real problem:
- **Examine the description**: Identify the component, class, function, or file referenced.
- **Search the codebase**: Use `grep_search` to locate the relevant files/functions in the local workspace.
- **Inspect the code**: Open the files using `view_file` to analyze the code's current logic.
- **Verify the bug**:
  - Is the reported problem actually present in the code?
  - Does it produce the reported error or behavior under the current version (ADK 2.0)?
  - Is it a documentation typo, setup discrepancy, or a genuine code/logic bug?
- **Document your code evidence**: Reference specific file paths and line ranges (using clickable markdown file links, e.g., `[skill_toolset.py](file:///path/to/file#L123)`) in your report.

#### 3. Should we fix it?
Formulate a recommendation on whether the issue should be addressed:
- **Assess the impact**:
  - Does it break core functionality?
  - Does it affect standard developer workflows or introduce brittle workarounds?
  - Is it a high-priority bug or a low-impact cosmetic/feature request?
- **Check alignment**:
  - Does the suggested solution align with `adk-architecture` and `adk-style`?
  - Is it consistent with Python idioms and Pydantic validation rules?
- **Evaluate workarounds**: Is there a clean workaround, or is a core fix necessary?
- **Final Recommendation**: Clearly declare whether we should fix it, along with the reasoning and estimated complexity/scope of the fix.

#### 4. Is there a linked PR that fixes this issue?
Search for any existing pull requests that attempt to resolve the issue:
- **Search PRs**: Run `gh pr list --repo google/adk-python --search "<issue_number>"` to list pull requests mentioning the issue number in the branch name, title, or body.
- **Verify the PR details**: If PRs are found, fetch their details:
  ```bash
  gh pr view <pr_number> --repo google/adk-python --json number,title,state,url,body,author
  ```
- **Analyze progress**: Check if the PR is open, merged, or closed, and if it successfully fixes the issue according to the repository's testing patterns.
- **Present the structured report**: Format and present your findings structured as a premium report following the **Report Template** below.
- **Offer to create a fix**: If no existing PR is found, you MUST explicitly ask the user: "Would you like me to create and implement a fix for this issue in the workspace? (Note: The changes and tests will be created in a new branch but NOT committed, so you can review and iterate on them.)"

---

## Phase 2: Implementation (After User Approval)

### Step 3: Propose and Implement Fix
Once the user has approved the implementation of the fix in the workspace, follow these rules:
   - **Do NOT commit the changes**: Leave them uncommitted in the workspace so the user can review and iterate on them.
   - **Base the branch on remote HEAD**: When creating the new branch, ensure it is based on the remote tracking branch HEAD (`origin/main`), not the current local branch. For example:
     ```bash
     git checkout -b fix/<issue_number>-<desc> origin/main
     ```
   - **Follow these implementation steps**:
     1. **Create the fix**: Modify the necessary source files implementing clean, robust logic following `adk-style` and `adk-architecture`.
     2. **Add or update unittests**: Write comprehensive unit tests to verify the behavior and prevent regressions.
     3. **Update documentation**: Update `/docs/design` and `/docs/guides` if applicable to the changes.
     4. **Update samples**: Update `/contributing/samples` if applicable to demonstrate the new capability or updated behavior.

---

## Report Template

Present your final analysis as a high-quality markdown response using the following structure:

```markdown
# GitHub Issue #<issue_number> Analysis: <Issue Title>

## Detailed Analysis

### 1. Root Cause Analysis ("What is broken?")
- Explanation of the failure or bug (what is failing and why).
- Pinpoint the exact file, function, or design component that is malfunctioning.

### 2. Legitimacy Analysis
- **Status**: [Legitimate Bug / Feature Request / Duplicate / Invalid / Not Reproducible]
- **Evidence**:
  - Code references: [filename.py](file:///absolute/path/to/file#L100-L120)
  - Analysis of code behavior and why the issue occurs.

### 3. Fix Recommendation
- **Recommendation**: [Should Fix (High Priority) / Should Fix (Medium/Low Priority) / Won't Fix / Needs Discussion]
- **Rationale**:
  - Impact on user experience, workflows, or architecture.
  - Implementation complexity and risk of side effects.

### 4. Existing Pull Requests
- **Linked PR**: [None / Pull Request #<pr_number> - <PR Title> (<state>)]
- **PR URL**: <PR URL>
- **Analysis**: Brief summary of the PR's approach and status (e.g., "Fixes the bug by implementing X in Y, currently awaiting review").

---

## Executive Summary
1. **What is broken?** [Brief explanation of the root cause or error]
2. **Is the issue legitimate?** [Yes / No - brief explanation]
3. **Should we fix it?** [Yes / No / Needs Discussion - priority & brief reasoning]
4. **Is there a linked PR that fixes this issue?** [None / Yes, PR #<pr_number> - <state>]
```

---

## Tips & Best Practices
> [!TIP]
> Always use explicit repository qualifiers (`--repo google/adk-python`) when running `gh` commands to avoid failures due to custom internal or local git remotes.

> [!IMPORTANT]
> When presenting code files and lines, always use markdown file links that point directly to the files in the workspace. Make sure the link is clickable and formatted as `[filename.py](file:///absolute/path/to/file#L100-L120)` without surrounding backticks around the brackets.
