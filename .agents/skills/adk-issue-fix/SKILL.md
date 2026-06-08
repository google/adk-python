---
name: adk-issue-fix
description: Implement a bug fix or feature for a GitHub issue in the adk-python repository. Use this skill after the triage/analysis is complete and approved. It creates a new branch, implements code changes, adds tests, and updates relevant documentation/samples. Triggers on "/adk-issue-fix" commands.
---

# ADK Issue Fix Implementation

This skill provides a structured workflow for implementing bug fixes or new features for GitHub issues in the `google/adk-python` repository. Only invoke/use this skill once the user has approved the fix.

## Implementation Steps

### 1. Check for Existing Pull Requests (Entry Gate)
- **Check linked pull requests**: Before creating a branch or implementing changes, check if the issue is already linked to any pull requests by querying the issue details:
  ```bash
  gh issue view <issue_number> --repo google/adk-python --json closedByPullRequestsReferences
  ```
- **Refuse to Proceed**: If the `closedByPullRequestsReferences` list is not empty (which indicates that there are already pull requests linked to this issue):
  - **Stop immediately**: You MUST refuse to proceed with the fix implementation.
  - **Output details**: Report the linked PR details (PR number, URL) to the user and terminate the skill execution.

### 2. Base the Branch on Remote HEAD & Create Branch
- **Do NOT commit the changes**: Leave them uncommitted in the workspace so the user can review and iterate on them.
- **Base the branch on remote HEAD**: When creating the new branch, ensure it is based on the remote tracking branch HEAD (`origin/main`), not the current local branch. For example:
  ```bash
  git checkout -b fix/issue-<issue_number> origin/main
  ```

### 3. Implement the Fix
- Modify the necessary source files implementing clean, robust logic following `adk-style` and `adk-architecture`.

### 4. Add or Update Unittests
- Write comprehensive unit tests to verify the behavior and prevent regressions. Refer to the testing patterns in the testing guides.

### 5. Update Documentation & Samples
- Update `/docs/design` and `/docs/guides` if applicable to the changes.
- Update `/contributing/samples` if applicable to demonstrate the new capability or updated behavior.
