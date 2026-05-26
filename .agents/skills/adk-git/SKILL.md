---
name: adk-git
description: Use for any git operation (commit, push, pull, rebase, branch, PR, cherry-pick, etc.). Provides commit message format and conventions.
---

# Git Operations for adk-python

## Commit Message Format

Use **Conventional Commits**:

```
<type>(<scope>): <description>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructure without behavior change
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `chore`: Build, config, dependencies
- `ci`: CI/CD changes

### Description Phrasing

**CRITICAL**: The subject line must answer **why**, not just **what**.
A reviewer reading only the subject should understand the motivation.

- **State the outcome**, not the mechanics:
  - Good: `Fix race condition when two agents write to same session`
  - Bad: `Update session.py to add lock`
- **Name the capability added**, not the implementation:
  - Good: `Support parallel tool execution in workflows`
  - Bad: `Add asyncio.gather call in execute_tools_node`
- **For refactors, state the reason**, not just the action:
  - Good: `Make graph public for dev UI serialization`
  - Bad: `Make graph a public field on new Workflow`
- **For bug fixes, state what was broken**:
  - Good: `Prevent duplicate events when resuming HITL`
  - Bad: `Check interrupt_id before appending`

### Detailed Commit Messages

Promote detailed commit messages by including a short, concrete explanation in the body:
- For **features**: Give a sample usage or explain the new capability.
- For **fixes**: Explain what caused the error and how the fix addresses it.

**Example (Feature):**
```
feat(workflow): Support JSON string parsing in schema validation

Automatically parse JSON strings into dicts or Pydantic models when input_schema or output_schema is defined on a node.
```

**Example (Fix):**
```
fix(sessions): Prevent duplicate events when resuming HITL

The interrupt_id was not checked before appending, causing duplicates if the user resumed multiple times. Added a check to ignore already processed interrupts.
```

Self-check before committing: read your subject line and ask "does this tell me _why_ someone made this change?" If it only describes _what_ changed, rewrite it.

### Rules

1. **Imperative mood** - "Add feature" not "Added feature".
2. **Capitalize** first letter of description (for release-please changelog).
3. **No period** at end of subject line.
4. **50 char limit** on subject line when possible, max 72.
5. **Use body for context** - Add a blank line then explain _why_,
   not _how_, when the subject alone isn't enough.

### Examples

```
feat(agents): Support App pattern with lifecycle plugins
fix(sessions): Prevent memory leak on concurrent session cleanup
refactor(tools): Unify env var checks across tool implementations
docs: Add contributing guide for first-time contributors
```

## Pre-commit Hooks

> [!IMPORTANT]
> Before performing any commit, check if `pre-commit` is installed and configured with the expected hooks (`isort`, `pyink`, `addlicense`, `mdformat`). If not, remind the user to set up pre-commit hooks using the `adk-setup` skill.
