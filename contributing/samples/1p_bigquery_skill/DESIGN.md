# First-Party (1P) Skills for ADK Toolsets

Please refer to [go/orcas-hermes-guide](http://goto.google.com/orcas-hermes-guide). Please submit your RFC at [go/hermes-orcas](http://goto.google.com/hermes-orcas)   
---

# Summary

This RFC proposes a standardized method for bundling and consuming "First-Party (1P) Skills" alongside existing ADK toolsets (e.g., `BigQueryToolset`, `SpannerToolset`). These 1P skills, compliant with the [agentskills.io specification](https://agentskills.io/specification), will encapsulate best practices and guided workflows for using the toolset's raw tools. This approach enhances developer experience by providing discoverable, versioned guidance without requiring any changes to core ADK APIs or classes. Developers opt-in by adding both the base toolset and the associated `SkillToolset` to their agent. Example implementation based on this RFC you can refer to [this PR.](https://github.com/google/adk-python/pull/4678) 

# Motivation

Currently, ADK toolsets provide powerful but low-level tools (e.g., `execute_sql`). Developers are responsible for engineering the prompts and logic to use these tools effectively, often embedding complex workflow guidance directly into agent instructions. This leads to:

* **Duplicated Effort:** Each developer reinvents common usage patterns.  
* **Inconsistent Quality:** Lack of standardized workflows results in varying reliability.  
* **Poor Discoverability:** Expertise about toolset usage is not easily shared or found.  
* **Bloated Instructions:** Agent prompts become long and hard to maintain.

By shipping 1P Skills with toolsets, we can provide reusable, curated knowledge on how to best utilize ADK components.

# Proposal

We propose to package spec-compliant skill directories within the ADK library, alongside the toolsets they guide. These skills will be loaded using the existing `SkillToolset` and `load_skill_from_dir` mechanisms.

## Key Concepts:

1. **Co-location:** 1P skill directories will reside within the corresponding toolset's module path (e.g., `google/adk/tools/bigquery/skills/bigquery-data-analysis/`).  
2. **Standard Specification:** Skills will adhere to the [agentskills.io specification](https://agentskills.io/specification).  
3. **Existing Mechanisms:** Consumption is via the standard `SkillToolset`. No new ADK classes or APIs are introduced.  
4. **Opt-In Usage:** Developers explicitly add the `SkillToolset` with the desired 1P skill(s) to their agent. There is no automatic inclusion.  
5. **Convenience Loaders:** A simple function like `get_bigquery_skill()` will be provided for easy loading.

## Directory Structure Example:

```
src/google/adk/integration/bigquery/    # Canonical location
├── __init__.py                         # Exports BigQueryToolset, etc.
├── bigquery_toolset.py                 # Raw tools
├── bigquery_credentials.py             # Credentials config
├── bigquery_skill.py                   # Skill loader
├── client.py                           # BQ client helper
├── config.py                           # Tool configuration
├── data_insights_tool.py               # Data insights tool
├── metadata_tool.py                    # Metadata tools
├── query_tool.py                       # Query tools
└── skills/
    └── bigquery-data-analysis/         # Spec-compliant skill directory
        ├── SKILL.md                    # Frontmatter + workflow instructions
        └── references/
            ├── sql_patterns.md
            └── error_handling.md

src/google/adk/tools/bigquery/
└── __init__.py                         # Alias → integration.bigquery
                                        # (registers canonical modules
                                        #  in sys.modules for compat)
```

## Runtime Flow:

1P Skills leverage `SkillToolset`'s progressive disclosure:

* **L1 Metadata:** Skill name/description visible via `list_skills`.  
* **L2 Instructions:** Main `SKILL.md` content loaded via `load_skill(name=...)`.  
* **L3 References:** Detailed guides in `references/` loaded via `load_skill_resource(skill_name=..., resource_name=...)`.

This allows the agent to access guidance on demand without overloading the context window.

# API Usage

## Before:

```py
from google.adk.agents.llm_agent import LlmAgent
from google.adk.integration.bigquery import BigQueryToolset

bigquery_toolset = BigQueryToolset(credentials_config=creds)

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="analyst",
    instruction="""You are a data analyst. When analyzing data:
    1. First explore schemas...
    2. Use get_table_info...
    ... (many lines of hand-written guidance)""",
    tools=[bigquery_toolset],
)
```

## After:

```py
from google.adk.agents.llm_agent import LlmAgent
from google.adk.integration.bigquery import BigQueryToolset
from google.adk.integration.bigquery import get_bigquery_skill
from google.adk.tools.skill_toolset import SkillToolset

bigquery_toolset = BigQueryToolset(credentials_config=creds)
bq_skill_toolset = SkillToolset(skills=[get_bigquery_skill()])

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="analyst",
    instruction="You are a data analyst. Use your tools and skills.",
    tools=[bigquery_toolset, bq_skill_toolset], # Add both
)
```

The detailed guidance is now encapsulated within the skill, accessible through standard skill tools.

## Composability:

This design maintains a clear separation of concerns. Developers can mix and match:

* Toolset only.  
* Toolset \+ 1P Skill.  
* Toolset \+ Custom Skills.  
* Toolset \+ 1P Skill \+ Custom Skills.

# Implementation Pattern for Toolsets

1. **Create Skill Directory:** Add `src/google/adk/integration/<toolset>/skills/<skill-name>/` with `SKILL.md` and optional `references/`.
2. **Add Loader:** Create `src/google/adk/integration/<toolset>/<toolset>_skill.py`:

```py
import pathlib
from google.adk.skills import Skill, load_skill_from_dir

_SKILL_DIR = pathlib.Path(__file__).parent / "skills" / "<skill-name>"

def get_<toolset>_skill() -> Skill:
    return load_skill_from_dir(_SKILL_DIR)
```

3. **Add Alias (Optional):** Re-export from `src/google/adk/tools/<toolset>/` for backward compatibility.  
4. **Add Tests & Sample:** Validate skill structure and demonstrate usage.

**Candidate Toolsets for 1P Skills:** Spanner, Bigtable, PubSub.

# Backward Compatibility

All toolset code has moved to `google.adk.integration.bigquery` as the canonical location. The old `google.adk.tools.bigquery` path remains as a fully transparent alias: its `__init__.py` registers the canonical modules in `sys.modules` so that all existing imports (including `from google.adk.tools.bigquery.config import BigQueryToolConfig`) resolve to the same module objects where the real code lives. This ensures `mock.patch.object` and all other patterns continue to work without changes to existing tests or user code.

# Alternatives Considered

* **Embedding guidance in Toolset:** Would tightly coupled tools with specific workflows, reducing flexibility.  
* **New API/Class for 1P Skills:** Would increase API surface area unnecessarily, as existing `SkillToolset` fits the need perfectly.

The proposed approach is a minimalist design, maximizing reuse of existing components.

# Timeline

* **Phase 1:** Implement 1P Skill for `BigQueryToolset` as a proof-of-concept.  
* **Phase 2:** Develop 1P Skills for other key toolsets (Spanner, PubSub, etc.).  
* **Phase 3:** Document the pattern for community contributions.

# Outcome

* Improved developer experience for using complex toolsets.  
* Standardized, versioned, and discoverable best practices.  
* Reduced boilerplate in agent instructions.  
* A clear pattern for extending other ADK toolsets with 1P skills.
