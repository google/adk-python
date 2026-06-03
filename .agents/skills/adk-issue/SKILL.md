---
name: adk-issue
description: Orchestrate analyzing and triaging GitHub issues for the adk-python repository. Use this skill when a user provides a GitHub issue number or link. It coordinates triage analysis via `adk-issue-analyze` and implementation via `adk-issue-fix`. Triggers on "analyze issue", "issue #", "github issue", "github.com/google/adk-python/issues/".
---

# ADK Issue Resolution Orchestrator

This skill orchestrates the analysis, triage, and resolution of GitHub issues for the `google/adk-python` repository. When a user provides a GitHub issue number or link, follow this two-phase workflow by delegating/calling the specific sub-skills:

## Phase 1: Triage and Analysis (Read-Only)
1. **Delegate to `adk-issue-analyze`**: Follow the instructions in the `adk-issue-analyze` skill (located at `.agents/skills/adk-issue-analyze/SKILL.md`) to fetch the issue, inspect the codebase, evaluate legitimacy, search for existing PRs, and present a structured analysis report.
2. **CRITICAL**: Do NOT modify any code, create new branches, or write any implementation yet.
3. **Ask for Approval**: Present the report and explicitly ask the user:
   > "Would you like me to create and implement a fix for this issue in the workspace? (Note: The changes and tests will be created in a new branch but NOT committed, so you can review and iterate on them.)"
4. **Wait for Approval**: Do not proceed to Phase 2 until the user explicitly approves.

## Phase 2: Implementation (After User Approval)
1. **Delegate to `adk-issue-fix`**: Once the user approves, follow the instructions in the `adk-issue-fix` skill (located at `.agents/skills/adk-issue-fix/SKILL.md`) to create the branch, implement the fix, add/update tests, update docs, and update samples.
