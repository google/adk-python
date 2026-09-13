# SkillRegistry

`SkillRegistry` is the interface behind a searchable catalog of skills, the kind
an agent discovers at runtime instead of being handed up front. You implement it
to plug your own skill store into a
`SkillToolset`.

## Introduction

A `SkillToolset` normally carries a fixed list of skills you loaded from disk,
which is the right arrangement when the agent owns its own skills. It starts to
strain in three situations:

*   A catalog shared across many agents.
*   A store that should grow without anyone redeploying an agent.
*   A catalog so large that sending every description into the prompt is a waste
    on most turns.

A registry gets around all three by turning discovery into a tool call. Give the
toolset a registry and the agent gains a `search_skills` tool, and its
`load_skill` tool becomes able to fetch a skill that was never in the
constructor's list at all.

`SkillRegistry` is an abstract base class with two abstract methods and one
optional one. ADK ships a single implementation, `GCPSkillRegistry`, backed by
the Agent Registry API. You write your own when your skills live somewhere else,
whether that is a database, an internal service, or a bucket with an index of
its own.

```python
from google.adk.skills import SkillRegistry
```

## Get started

The registry below is about the smallest one that does anything useful, serving
skills out of an in-memory dictionary. Notice the asymmetry between the two
methods: `get_skill` returns a whole [`Skill`](../skill/index.md), while
`search_skills` returns only `Frontmatter`, which is the name and description,
and that is what keeps discovery cheap.

```python
import pathlib

from google.adk import Agent
from google.adk.skills import Frontmatter
from google.adk.skills import Skill
from google.adk.skills import SkillRegistry
from google.adk.skills import load_skills_from_dir
from google.adk.tools.skill_toolset import SkillToolset

catalog = {
    skill.name: skill
    for skill in load_skills_from_dir(pathlib.Path(__file__).parent / "skills")
}


class DictSkillRegistry(SkillRegistry):
  """Serves skills out of a dictionary keyed by skill name."""

  def __init__(self, skills: dict[str, Skill]):
    self._skills = skills

  async def get_skill(self, *, name: str) -> Skill:
    if name not in self._skills:
      raise KeyError(f"No skill named {name!r}")
    return self._skills[name]

  async def search_skills(self, *, query: str) -> list[Frontmatter]:
    terms = query.lower().split()
    return [
        skill.frontmatter
        for skill in self._skills.values()
        if any(t in skill.description.lower() for t in terms)
    ]


root_agent = Agent(
    name="support_agent",
    description="Answers customer questions.",
    instruction="Search for a skill before answering an unfamiliar question.",
    tools=[SkillToolset(skills=[], registry=DictSkillRegistry(catalog))],
)
```

Both methods are async and both take keyword-only arguments. Passing `skills=[]`
alongside a registry is perfectly reasonable, and it means the agent starts with
an empty local list and finds everything it uses through search.

## How it works

Nothing is ever pushed into the registry; it only ever gets pulled on. The
toolset calls it in response to the model's own tool calls, in this order:

1.  **`search_skills`** runs when the model calls the `search_skills` tool,
    which only exists when a registry is configured. It gets the model's query
    string and returns `Frontmatter` objects. The model sees names and
    descriptions, which is level 1 of the progressive disclosure the
    [Skill](../skill/index.md) guide describes.
2.  **`get_skill`** runs when the model calls `load_skill` for a name that is
    not in the toolset's local list. The full `Skill` comes back, and its
    instructions go to the model. Fetched definitions are cached per turn for up
    to sixteen turns, so a model that loads the same skill repeatedly in one
    conversation hits the registry once.
3.  **Resources** are then served from the fetched `Skill` object, not from the
    registry. `load_skill_resource` and `run_skill_script` read
    `skill.resources`, so whatever `get_skill` returned is the whole of what the
    model can reach.

The contract that matters is what you hand back. `get_skill` has to produce a
validated `Skill`, which means a `Frontmatter` whose `name` is kebab-case, the
`instructions` string, and a `Resources` holding the references, assets and
scripts. You can build those models directly, and if your store keeps skill
folders you unpack them first and construct the same objects.

### The two methods have different error contracts

`get_skill` is asked for one specific name that the model chose, so failing
loudly is the right behavior. Raise if the name does not exist, which is what
the abstract method's docstring asks for.

`search_skills` is a different situation, because the caller had no say in what
your catalog contains. One malformed entry must not break discovery for
everything else in the store. The shipped GCP implementation logs a warning and
skips any result that fails frontmatter validation, and yours should do the
same.

## Configuration options

`SkillRegistry` introduces no configuration of its own. It is an interface, and
these are the three methods:

| Method | Required | Signature | Returns |
| :--- | :--- | :--- | :--- |
| `get_skill` | yes | `async (*, name: str)` | `Skill` |
| `search_skills` | yes | `async (*, query: str)` | `list[Frontmatter]` |
| `search_tool_description` | no | `(self)` | `str \| None` |

**`search_tool_description`** is a plain synchronous method rather than an
abstract one, and it returns `None` by default. Override it to replace the
description the model sees on the `search_skills` tool. That is worth the effort
whenever your search has a syntax the model would benefit from knowing about,
such as matching whole words only, or supporting a tag filter. The default
description cannot know any of that, and a model left to guess your query
language burns a turn finding nothing.

Because both required methods are `@abstractmethod`, a subclass that implements
only one of them cannot be instantiated at all. Python raises `TypeError` at
construction, so you find out immediately rather than at the first call.

## Advanced applications

The first section below is the implementation ADK ships, which is worth reading
even if you are writing your own, because it shows what the interface expects of
an implementation. The second is the design question every registry runs into.

### The shipped implementation: GCPSkillRegistry

*   **Problem solved**: your skills are published to the Google Cloud Agent
    Registry and you want an agent to discover them without a local copy.
*   **Implementation**: construct it with a project and a location, and pass it
    as the toolset's registry.

```python
from google.adk.integrations.skill_registry import GCPSkillRegistry
from google.adk.tools.skill_toolset import SkillToolset

registry = GCPSkillRegistry(project_id="my-project", location="us-central1")
skill_toolset = SkillToolset(skills=[], registry=registry)
```

All three arguments are keyword-only. `project_id` and `location` fall back to
the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` environment variables,
and the constructor raises `ValueError` if neither the argument nor the variable
supplies a value. `credentials` defaults to application default credentials,
resolved lazily on the first request rather than at construction.

`get_skill` downloads the skill's default revision and unpacks it into a
`Skill`, so it costs more than one network round trip. A skill with no default
revision raises `ValueError`. The name is validated against the skill naming
pattern before it goes anywhere near a URL, since it arrives straight from a
model-issued tool call.

`search_skills` calls the registry's search endpoint and builds a `Frontmatter`
per result, skipping and logging any entry that fails validation.

### Decide what `search_skills` should return

*   **Problem solved**: with a large catalog, the search results become the whole
    prompt budget. Return too many and you have re-created the exact problem
    skills were invented to avoid.
*   **Implementation**: return the fewest entries that could plausibly answer the
    query, and put the "when to use this" content into each `description`, since
    that string is all the model has to choose from. The return type carries no
    relevance score and there is no pagination, which leaves ranking and
    truncation entirely on your side of the interface.

## Limitations

*   **Search returns no score and no cursor.** `list[Frontmatter]` is the whole
    contract, so the model sees your results in the order you return them with
    no notion of confidence, and there is no way to ask for a second page.
*   **A registry cannot list its whole catalog.** The interface has no
    `list_skills`, and the model's `list_skills` tool reports only the toolset's
    local skills. A model that never thinks to search never learns the registry
    is there at all.
*   **Skill names must satisfy the frontmatter rules.** A store whose keys are
    not kebab-case cannot round-trip through `Frontmatter` validation, so map
    them at the boundary.
*   **Caching is the toolset's, not yours.** Fetched skills are cached per turn
    for sixteen turns inside `SkillToolset`. A registry has no way to invalidate
    that, so a skill edited mid-conversation may not be picked up until later.
*   **`GCPSkillRegistry` needs an endpoint that exists.** It targets the Agent
    Registry API and honors the `AGENT_REGISTRY_ENDPOINT` environment variable;
    a failed HTTP call surfaces as a `RuntimeError` carrying the status and
    body.
*   **Experimental.** The skills package is marked experimental and under active
    development, so this interface may change.

## Related samples

*   [GCP Skill Registry agent](../../../../contributing/samples/integrations/gcp_skill_registry_agent)
    wires the shipped registry into a `SkillToolset` that has an empty local
    skill list.
*   [Skills](../../../../contributing/samples/environment_and_skills/skills) sits
    at the other end of the trade-off, supplying skills directly with no registry
    involved.

## Related guides

*   [Skill](../skill/index.md) describes what a `Skill`, a `Frontmatter` and a
    `Resources` actually hold, which is exactly what `get_skill` has to produce.
*   `SkillToolset` is the consumer, and it
    covers the tools the model gets once a registry is configured.
