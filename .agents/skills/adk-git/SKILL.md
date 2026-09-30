---
name: adk-git
description: >-
  Writes commit messages and pull request descriptions for the adk-python
  repository: Conventional Commits types and scopes, subject lines that say why
  a change was made, and the linked-issue and testing-plan sections the PR
  template requires. Use when writing or rewording a commit message, squashing
  commits before a pull request, drafting a PR description, or checking that a
  change is shaped to land. Don't use for generic git mechanics such as
  rebasing, resolving conflicts, cherry-picking, or branch surgery; those need
  no skill. Don't use to judge the content of a change (use adk-review) or for
  code style and naming (use adk-style).
---

# Commit and Pull Request Conventions

## Commit message format

Conventional Commits:

```text
<type>(<scope>): <description>
```

The type decides where the commit lands in `CHANGELOG.md`. `release-please`
generates the changelog from merged commit subjects, so the wrong type either
files the change under the wrong heading or drops it from the release notes
entirely.

| Type                                               | Changelog section        |
| -------------------------------------------------- | ------------------------ |
| `feat`                                             | Features                 |
| `fix`                                              | Bug Fixes                |
| `perf`                                             | Performance Improvements |
| `docs`                                             | Documentation            |
| `refactor`, `test`, `build`, `ci`, `style`, `chore` | hidden, no entry         |

The mapping lives in `.github/release-please-config.json`. A type that is not
listed there produces no changelog entry at all.

Scope is optional. Use a short module name with no underscores
(`fix(cli):`, `feat(a2a):`, `fix(sessions):`) or leave it off.

## Subject line

Say why the change exists, not which lines moved. A reviewer who reads only the
subject should understand the motivation.

| Write                                                        | Not                                                            |
| ------------------------------------------------------------ | -------------------------------------------------------------- |
| `fix(sessions): prevent duplicate events when resuming HITL` | `fix(sessions): check interrupt_id before appending`           |
| `feat(workflow): support parallel tool execution`            | `feat(workflow): add asyncio.gather call in execute_tools_node` |
| `refactor: make graph public for dev UI serialization`       | `refactor: make graph a public field on Workflow`              |

Rules:

1. Imperative mood: `add`, not `added`.
2. Lowercase the first word after the colon. `release-please` copies the
   subject into `CHANGELOG.md` verbatim, and the great majority of merged
   commits are lowercase, so capitalizing makes one line stand out.
3. No trailing period.
4. Keep the subject under about 72 characters. Nothing enforces this, but each
   commit renders as one changelog line.
5. Reference the issue in the body, not the subject: `Fixes #1234` or
   `Closes #1234`, or the full issue URL when the issue lives in another
   repository.

Self-check: read the subject back and ask whether it says *why* someone made
the change. If it only names the edit, rewrite it.

## Commit body

Add a blank line, then a short concrete explanation. For a feature, show the
new capability or a usage line. For a fix, say what caused the failure and how
the change addresses it.

```text
feat(workflow): support JSON string parsing in schema validation

Parse JSON strings into dicts or Pydantic models when input_schema or
output_schema is defined on a node.
```

```text
fix(sessions): prevent duplicate events when resuming HITL

interrupt_id was not checked before appending, so resuming twice appended the
same event twice. Ignore interrupts that were already processed.

Fixes #1234
```

## Before committing

`pre-commit` reformats and checks staged files, and the same hooks run again in
CI on every pull request, so a commit made with hooks skipped fails there.

```bash
pre-commit install                  # once per clone
pre-commit run --files {paths}      # check only what changed
```

The hooks include `isort`, `pyink`, `addlicense`, `mdformat`, `ruff`,
`codespell`, and repository-local compliance checks; see
`.pre-commit-config.yaml`. If `pre-commit` is not installed, point the user at
the `adk-setup` skill rather than committing unformatted code.

## Pull requests

- Every PR except a small documentation or typo fix needs a linked issue.
  Put `Closes: #{issue_number}` in the PR description, or describe the problem
  and solution inline following the issue templates.
- Fill in the Testing Plan section of `.github/pull_request_template.md`,
  including a summary of passing `pytest` results.
- Do not merge on GitHub. The `Do Not Merge on GitHub` check fails on every PR
  to `main` by design; a maintainer lands the change and it is synced back to
  the repository. GitHub then shows the PR as closed with a `merged` label
  rather than merged, and the landed commit carries the original authorship.
  That red check is expected and is not something to fix.
