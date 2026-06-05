---
name: adk-issue-analyze
description: Analyze and triage a GitHub issue for the adk-python repository. Use this skill to retrieve issue details, inspect the codebase, evaluate justification, check for existing PRs, and produce a structured analysis report. Triggers on "/adk-issue-analyze" commands. This skill is strictly read-only and must be used whenever the "/adk-issue-analyze" command is explicitly called.
---

# ADK Issue Triage & Analysis (Read-Only)

This skill provides a structured workflow for analyzing, verifying, and triaging GitHub issues from the `google/adk-python` repository. When instructed to analyze/triage an issue, follow this read-only workflow.

> [!IMPORTANT]
> **Strict Read-Only Constraint**:
> This skill is strictly **read-only**. You MUST NOT modify any code, create new branches, or write any implementation. Your role is only to analyze the issue and output the report. Do NOT use file creation or editing tools (e.g. `write_to_file`, `replace_file_content`, `edit_file`, etc.).
>
> **Strict Tooling Constraint**:
> Do NOT use `curl`, `wget`, or any HTTP requests to fetch issue/PR content. You MUST parse/extract the issue number and use strictly the custom `fetch_github_issue` / `fetch_github_pr` python tools (or the `gh` command).

## Step 1: Retrieve and Parse the Issue
1. **Extract the issue number**: Parse the number from the link or prompt (e.g., `https://github.com/google/adk-python/issues/5882` -> `5882`).
2. **Fetch issue details**: Use the custom python tool `fetch_github_issue(issue_number=<number>)` to get the issue metadata. This is the preferred method as it avoids command execution policy issues.
   *If the custom python tool is not available, fall back to running the gh command:*
   ```bash
   gh issue view <issue_number> --repo google/adk-python --json number,title,body,state,labels,comments,assignees,createdAt,closedAt
   ```

---

## Step 2: Deep Investigation & Analysis
Address the following three critical questions and present your findings in a structured, premium report.

### 1. What is broken?
Explain the root cause of the issue or failure:
- **Trace the execution flow**: Use `grep_search` and `view_file` to locate and analyze the malfunctioning components, classes, or functions in the local workspace.
- **Pinpoint the bug**: Detail why the system is behaving incorrectly and where the failure originates (e.g., incorrect logic, missing configuration, unhandled states).
- **Document code evidence**: Reference specific file paths and line ranges using clickable markdown file links, e.g., `[filename.py](file:///absolute/path/to/file#L100-L120)`.

### 2. Is there a linked PR that fixes this issue?
Search for any existing pull requests that attempt to resolve the issue:
- **Search PRs**: Run `gh pr list --repo google/adk-python --search "<issue_number>"` to list pull requests mentioning the issue number in the branch name, title, or body.
- **Verify the PR details**: If PRs are found, fetch their details:
  ```bash
  gh pr view <pr_number> --repo google/adk-python --json number,title,state,url,body,author
  ```
- **Analyze progress**: Check if the PR is open, merged, or closed, and if it successfully fixes the issue according to the repository's testing patterns.

### 3. Recommendation
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

---

## Report Template

Present your final analysis as a high-quality markdown response using the following structure:

```markdown
# GitHub Issue #<issue_number> Analysis: <Issue Title>

## Executive Summary
1. **What is broken?** [Brief explanation of the root cause or error]
2. **Is there a linked PR that fixes this issue?** [None / Yes, PR #<pr_number> - <state>]
3. **Recommendation**: [Should Fix (High Priority) / Should Fix (Medium/Low Priority) / Won't Fix / Needs Discussion - priority & brief reasoning]

<details>
<summary><b>Detailed Analysis</b></summary>

### 1. Root Cause Analysis ("What is broken?")
- Explanation of the failure or bug (what is failing and why).
- Pinpoint the exact file, function, or design component that is malfunctioning.
- Code references: [filename.py](file:///absolute/path/to/file#L100-L120)

### 2. Existing Pull Requests ("Is there a linked PR that fixes this issue?")
- **Linked PR**: [None / Pull Request #<pr_number> - <PR Title> (<state>)]
- **PR URL**: <PR URL>
- **Analysis**: Brief summary of the PR's approach and status (e.g., "Fixes the bug by implementing X in Y, currently awaiting review").

### 3. Recommendation
- **Recommendation**: [Should Fix (High Priority) / Should Fix (Medium/Low Priority) / Won't Fix / Needs Discussion]
- **Rationale**:
  - Impact on user experience, workflows, or architecture.
  - Implementation complexity and risk of side effects.
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
> When executing the `adk-issue-analyze` skill, you MUST NOT use any file modification or editing tools (such as `edit_file`, `replace_file_content`, `write_to_file`, `notebook_edit`, etc.). Your output must strictly be a text markdown report following the template provided, without editing any workspace files or writing/fixing code.

> [!TIP]
> Always use explicit repository qualifiers (`--repo google/adk-python`) when running `gh` commands to avoid failures due to custom internal or local git remotes.

> [!IMPORTANT]
> When presenting code files and lines, always use markdown file links that point directly to the files in the workspace. Make sure the link is clickable and formatted as `[filename.py](file:///absolute/path/to/file#L100-L120)` without surrounding backticks around the brackets.
