# Design: First-Party (1P) Skills for ADK Toolsets

## Problem

ADK toolsets like `BigQueryToolset` provide raw tools (e.g., `execute_sql`,
`list_dataset_ids`) but no guidance on how to use them effectively. Developers
must re-invent prompt engineering for each toolset, embedding workflow
knowledge directly in agent instructions. This leads to:

- Duplicated effort across agent builders.
- Inconsistent quality of analysis workflows.
- No standard way to share toolset expertise.
- Agent instructions that grow unwieldy as guidance accumulates.

## Solution

Pre-packaged skills that follow the
[agentskills.io specification](https://agentskills.io/specification),
consumed via ADK's existing `SkillToolset`. Zero new APIs, zero new classes.

A 1P skill is simply a spec-compliant skill directory that ships with ADK
alongside its corresponding toolset. Users add both the toolset (for tools)
and a `SkillToolset` (for guided workflows) to their agent.

```python
# Before: raw toolset, no guidance
root_agent = LlmAgent(tools=[bigquery_toolset])

# After: toolset + 1P skill for guided workflows
bq_skill_toolset = SkillToolset(skills=[get_bigquery_skill()])
root_agent = LlmAgent(tools=[bigquery_toolset, bq_skill_toolset])
```

## How It Works

### Progressive Disclosure

The skill content is loaded in three levels, keeping context efficient:

1. **L1 - Metadata** (always in context): Skill name and description are
   returned by `list_skills`. The LLM sees what skills are available without
   loading full instructions.

2. **L2 - Instructions** (loaded on activation): When the LLM calls
   `load_skill(name="bigquery-data-analysis")`, it receives the SKILL.md
   body with step-by-step workflow guidance.

3. **L3 - References** (loaded on demand): When the LLM needs detailed
   patterns, it calls `load_skill_resource` to load specific reference
   files (e.g., `sql_patterns.md`, `error_handling.md`).

### Runtime Flow

```
1. Agent starts -> SkillToolset injects skill system instruction
2. User asks question -> LLM sees list_skills tool available
3. LLM calls list_skills -> sees "bigquery-data-analysis" skill
4. LLM calls load_skill("bigquery-data-analysis") -> gets workflow steps
5. LLM follows steps, using BigQuery tools (execute_sql, etc.)
6. LLM calls load_skill_resource for detailed patterns as needed
```

### Directory Structure

```
src/google/adk/tools/bigquery/
├── bigquery_toolset.py          # Existing: raw tools
├── bigquery_skill.py            # New: get_bigquery_skill() loader
└── skills/
    └── bigquery-data-analysis/  # Spec-compliant skill directory
        ├── SKILL.md             # Frontmatter + workflow instructions
        └── references/
            ├── sql_patterns.md
            ├── schema_exploration.md
            └── error_handling.md
```

## API Usage

### Before (tools only)

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset

bigquery_toolset = BigQueryToolset(credentials_config=creds)

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="analyst",
    instruction="""You are a data analyst. When analyzing data:
    1. First explore schemas with list_dataset_ids, list_table_ids...
    2. Use get_table_info before writing queries...
    3. Always use LIMIT on exploratory queries...
    4. Use CTEs for complex queries...
    5. Handle errors by checking get_job_info...
    ... (many lines of hand-written guidance)""",
    tools=[bigquery_toolset],
)
```

### After (tools + 1P skill)

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset
from google.adk.tools.bigquery.bigquery_skill import get_bigquery_skill
from google.adk.tools.skill_toolset import SkillToolset

bigquery_toolset = BigQueryToolset(credentials_config=creds)
bq_skill_toolset = SkillToolset(skills=[get_bigquery_skill()])

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="analyst",
    instruction="You are a data analyst. Use your tools and skills.",
    tools=[bigquery_toolset, bq_skill_toolset],
)
```

The curated guidance moves from fragile inline instructions into a
structured, versioned, spec-compliant skill that the agent discovers
and loads at runtime.

### Composability

`BigQueryToolset` and `SkillToolset` are fully independent — neither
depends on nor references the other. The 1P skill is opt-in; nothing
auto-includes it. This means all of the following patterns work:

```python
# BigQuery toolset + your own custom skills (no 1P BQ skill)
my_skill = load_skill_from_dir("path/to/my-custom-skill")
root_agent = LlmAgent(
    tools=[
        BigQueryToolset(credentials_config=creds),
        SkillToolset(skills=[my_skill]),
    ],
)
```

```python
# BigQuery toolset + 1P BQ skill + your own skills (all together)
root_agent = LlmAgent(
    tools=[
        BigQueryToolset(credentials_config=creds),
        SkillToolset(skills=[get_bigquery_skill(), my_skill]),
    ],
)
```

```python
# BigQuery toolset alone, no skills at all
root_agent = LlmAgent(
    tools=[BigQueryToolset(credentials_config=creds)],
)
```

Users choose exactly which skills to include. The `get_bigquery_skill()`
loader is a convenience, not a coupling.

## Why This Design Is Minimal

This design achieves guided workflows with the absolute minimum change
to the existing API surface:

1. **No behavioral changes** to `BigQueryToolset`, `SkillToolset`,
   `LlmAgent`, or the runner flow.
2. **No signature changes** or breaking changes to existing public APIs.
3. **Entirely additive**: a packaged skill directory + a thin loader +
   sample + tests.
4. **Opt-in**: existing user patterns work unchanged; the new pattern
   is `tools=[bigquery_toolset, skill_toolset]`.

### Trade-off: Minimalism vs. Ergonomics

For minimum API churn, this is the right design. A more ergonomic
single-line UX (e.g., `BigQueryToolset(include_skill=True)`) would
require new convenience APIs, increasing surface area and review risk.
The current two-line pattern keeps the toolset and skill concerns
cleanly separated.

### Public Surface Note

`get_bigquery_skill` is exported from `google.adk.tools.bigquery` for
discoverability alongside `BigQueryToolset`. This is still additive and
acceptable. For absolute-minimum public surface, it could instead be
kept as an import from `google.adk.tools.bigquery.bigquery_skill` only.

## Repeatable Template for New Toolsets

The pattern scales cleanly to Spanner, Bigtable, PubSub, and other
toolsets without changing existing core APIs. Follow these steps per
toolset:

1. Add a spec-compliant skill directory under
   `src/google/adk/tools/<toolset>/skills/<skill-name>/`.
2. Add a thin loader `get_<toolset>_skill()` that calls
   `load_skill_from_dir(...)`.
3. (Optional but recommended) Export the loader in
   `src/google/adk/tools/<toolset>/__init__.py`.
4. Add tests for skill validity + `SkillToolset` integration.
5. Add a sample showing
   `tools=[<Toolset>(...), SkillToolset(skills=[get_<toolset>_skill()])]`.

### 1. Create a Spec-Compliant Skill Directory

```
src/google/adk/tools/<toolset>/skills/<skill-name>/
├── SKILL.md              # Required: YAML frontmatter + instructions
└── references/           # Optional: detailed reference materials
    └── ...
```

The directory name must match the `name` field in SKILL.md frontmatter.

### 2. Add a Convenience Loader

```python
# src/google/adk/tools/<toolset>/<toolset>_skill.py

import pathlib
from google.adk.skills import Skill, load_skill_from_dir

_SKILL_DIR = pathlib.Path(__file__).parent / "skills" / "<skill-name>"

def get_<toolset>_skill() -> Skill:
    return load_skill_from_dir(_SKILL_DIR)
```

### 3. Users Combine Toolset + SkillToolset

```python
from google.adk.tools.<toolset> import <Toolset>
from google.adk.tools.<toolset>.<toolset>_skill import get_<toolset>_skill
from google.adk.tools.skill_toolset import SkillToolset

toolset = <Toolset>(...)
skill_toolset = SkillToolset(skills=[get_<toolset>_skill()])
agent = LlmAgent(tools=[toolset, skill_toolset])
```

### Candidate Toolsets

- **Spanner**: Schema design, transaction patterns, query optimization.
- **Bigtable**: Row key design, filter patterns, scan optimization.
- **PubSub**: Topic/subscription setup, message handling, dead-letter queues.

## Spec Compliance

The skill directory maps to [agentskills.io](https://agentskills.io/specification)
fields as follows:

| Spec Field | Source |
|------------|--------|
| `name` | SKILL.md frontmatter `name` (must match directory name) |
| `description` | SKILL.md frontmatter `description` |
| `license` | SKILL.md frontmatter `license` |
| `metadata` | SKILL.md frontmatter `metadata` |
| `instructions` | SKILL.md body (after frontmatter) |
| `references` | `references/` directory (loaded by `load_skill_resource`) |
| `assets` | `assets/` directory (not used by this skill) |
| `scripts` | `scripts/` directory (not used by this skill) |

ADK's `load_skill_from_dir()` validates name-directory match, parses YAML
frontmatter, and loads all resource directories. `SkillToolset` provides
the standard tools for skill discovery, loading, and resource access.
