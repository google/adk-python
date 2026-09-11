# Skill, Frontmatter, and Resources

A skill is a folder of instructions, reference documents, and scripts that an
agent can pull in at runtime, but only on the turns where it is relevant.
`Skill` is the in-memory representation of that folder, and the loader functions
in `google.adk.skills` are how you get one. Authoring a skill is a different job
from wiring one up, and it starts with the `SKILL.md` file at the root of that
folder.

## Introduction

Putting every set of instructions an agent might need into its system prompt
costs tokens on every single turn, and on any given turn most of them are dead
weight. Skills get around that by splitting each set of instructions into the
three numbered levels below, and loading each level only at the point where it
earns its place. The model works down them in order.

1.  **The frontmatter.** A name and a one-paragraph description. This is the
    only part the model sees up front, and it sees it for every skill you have
    installed. It stays small enough that a catalog of twenty skills costs
    almost nothing.
2.  **The instructions.** The body of `SKILL.md`. The model reads this only
    after it has decided, on the strength of the description alone, that the
    skill applies to what it is doing.
3.  **The resources.** Reference documents, data files, and executable scripts.
    Each one is fetched individually, by name, and only when the instructions
    tell the model to go and get it.

`Skill` is a Pydantic model with exactly those three parts: `frontmatter` (a
`Frontmatter`), `instructions` (a `str`), and `resources` (a `Resources`). You
can either write a skill folder on disk and load it, or build a `Skill` in
Python. Whichever route you take, you then hand the result to `SkillToolset`
(`google.adk.tools.skill_toolset`), which is the piece that actually puts the
skill in front of the model.

Skills follow the [Agent Skills](https://agentskills.io/) convention, so a skill
folder written for another agent runtime will usually load here without any
changes.

## The SKILL.md format

Everything the model ever learns about a skill comes out of this one file,
either directly or through the resources it points at. It has two halves, the
YAML frontmatter and the markdown body, and getting both right is most of what
authoring a skill amounts to.

A skill on disk is a directory, and the directory name is the skill's name. The
one file the loader insists on is `SKILL.md` at the top of it. Everything else
is optional, and the three subdirectory names below are the only ones the loader
looks at:

```
weather-skill/
  SKILL.md          # required
  references/       # optional
    weather_info.md
  assets/           # optional
    station_codes.csv
  scripts/          # optional
    get_humidity.py
```

`SKILL.md` itself is YAML frontmatter followed by a markdown body. The following
is a complete one for the folder above:

```markdown
---
name: weather-skill
description: A skill that provides weather information based on reference data and scripts.
metadata:
  adk_additional_tools:
    - get_wind_speed
---

Step 1: Check 'references/weather_info.md' for the current weather.
Step 2: If humidity is requested, run 'scripts/get_humidity.py' with the `location` argument.
Step 3: If wind speed is requested, use the `get_wind_speed` tool.
Step 4: Provide the complete weather update to the user.
```

Reading that from the top: the `name` is how the model refers to the skill, the
`description` is the sales pitch that decides whether it ever gets loaded, and
`metadata.adk_additional_tools` names a tool that appears only once the skill is
active. The body below the frontmatter is written as instructions addressed to a
model, and it mentions its resources by path relative to the skill directory,
which is how the model knows what to ask for at level 3.

The loader is strict about the frontmatter, and every rule below raises rather
than warns, so a mistake shows up as a `ValueError` at load time instead of as
odd behavior later:

*   The file must start with `---`, and a second `---` must close the
    frontmatter. Anything else raises `ValueError`.
*   The frontmatter must parse as a YAML mapping.
*   `name` and `description` are both required.
*   `name` must be lowercase kebab-case, meaning `a-z`, `0-9`, and single
    hyphens, with no leading, trailing, or consecutive hyphens. The loader
    Unicode-normalizes it (NFKC) before checking, and it must be at most 64
    characters. Underscores are rejected unless the `SNAKE_CASE_SKILL_NAME`
    feature is enabled, and even with that on you cannot mix hyphens and
    underscores in one name.
*   `name` must equal the directory name. A `weather-skill/SKILL.md` that
    declares `name: weather` raises `ValueError`, so renaming a skill is always
    two edits.
*   `description` must be non-empty and at most 1024 characters.
*   `compatibility`, if you use it, is at most 500 characters.

The filename may be spelled `SKILL.md` or `skill.md`, and `SKILL.md` is checked
first.

Everything after the closing `---` becomes `skill.instructions`, stripped of
surrounding whitespace. There is no template to follow for the body and no
structure the loader expects, so write whatever a model would need to be told to
carry the task out, in the order it should do things.

Unknown top-level keys in the frontmatter are allowed rather than rejected, and
they are preserved on the `Frontmatter` object. That is what lets a skill
carrying keys meant for another runtime load here anyway.

## Get started

Loading the `weather-skill` folder from above and putting it in front of a model
takes one loader call and one toolset. The `get_wind_speed` function is passed
as an additional tool because the skill's frontmatter asked for it by name.

```python
import pathlib

from google.adk.agents import Agent
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset


def get_wind_speed(location: str) -> str:
  """Returns the current wind speed for a given location."""
  return f"The wind speed in {location} is 10 mph."


weather_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "weather-skill"
)

root_agent = Agent(
    name="weather_agent",
    description="An agent that answers weather questions.",
    tools=[
        SkillToolset(
            skills=[weather_skill],
            additional_tools=[get_wind_speed],
        )
    ],
)
```

If you want to see what the loader actually produced, the `Skill` object exposes
each of the three levels directly:

```python
weather_skill.name                                  # "weather-skill"
weather_skill.description                           # the description line
weather_skill.instructions                          # the markdown body
weather_skill.resources.list_references()           # ["weather_info.md"]
weather_skill.resources.get_script("get_humidity.py").src  # the script source
```

## How it works

A loaded `Skill` mirrors the folder it came from. The frontmatter and the body
of `SKILL.md` become `frontmatter` and `instructions`, and the three resource
subdirectories become `resources`.

Resources recurse into subfolders, but what is found is flattened into keys
rather than kept as a tree. Every file under `references/` becomes one entry
in `resources.references`, keyed by its path relative to that subdirectory and
always written with forward slashes, so `references/deep/note.md` ends up under
the key `deep/note.md`. The `assets/` and `scripts/` folders work the same way,
filling `resources.assets` and `resources.scripts`. Anything in a directory
named `__pycache__` is skipped.

Whether a file is treated as text or binary comes down to whether it decodes as
UTF-8. Text is stored as `str` and everything else as `bytes`, which is how a
PNG under `assets/` survives the trip intact. Scripts behave differently, because
`resources.scripts` holds `Script` objects whose `src` field is source code. A
script that is not valid UTF-8 has no sensible `src`, so it is dropped with a log
warning instead of being stored.

Once the skill reaches a `SkillToolset`, the toolset publishes four tools to the
model: `list_skills`, `load_skill`, `load_skill_resource`, and
`run_skill_script`. A fifth, `search_skills`, appears when a registry is
configured. The model reaches the three levels by calling those tools in order:

1.  `list_skills` returns the name and description of every installed skill,
    which is level 1.
2.  `load_skill` returns the body of the one it picked, which is level 2.
3.  `load_skill_resource` or `run_skill_script` reaches a single file or
    script, which is level 3.

Those calls are the progressive disclosure across the three levels, and `Skill`
itself is the passive data structure the tools read from. Each of them hands its
answer back as a tool response, so a skill body arrives in the conversation
rather than being written into the system instruction.

## Frontmatter fields

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | *required* | Kebab-case identifier, at most 64 characters. Must match the directory name. |
| `description` | `str` | *required* | What the skill does and when to use it, at most 1024 characters. |
| `license` | `str \| None` | `None` | License of the skill content. Not interpreted by ADK. |
| `compatibility` | `str \| None` | `None` | Free text, at most 500 characters. Not interpreted by ADK. |
| `allowed_tools` | `str \| None` | `None` | Space-delimited list of pre-approved tools. Also accepted as the YAML key `allowed-tools`. |
| `metadata` | `dict[str, Any]` | `{}` | Client-specific properties. ADK reads two keys from it, below. |

**`description` is the field that decides whether your skill is ever used.** It
is the only thing the model has to go on when it chooses, so write it as "what
this does and when to use it" rather than as a title. The 1024-character
cap leaves plenty of room for that, and a description that uses only a fraction
of it is usually a description that will lose out to a better-written one.

**`metadata.adk_additional_tools`** is a list of tool names. When the model
activates the skill, `SkillToolset` looks each name up among the tools you passed
as `additional_tools=` and adds the matches to the tool list for the rest of the
conversation. Naming tools that way is how a skill brings its own along without
them crowding the model's tool list on every other turn. The value has to be a list,
and a bare string raises a validation error. Names that match nothing you
supplied are quietly not added, with no warning, so check the spelling against
your Python function names if a tool the skill promised never turns up.

**Adding a tool this way costs you a context cache hit.** If you enabled context
caching by setting `context_cache_config` on the `App`, the cached prefix is
identified by the system instruction, the tool declarations and the tool config
along with the leading conversation, so the tool list has to stay identical from
one request to the next. A skill that activates a tool changes that list partway
through the conversation, and the request after it misses the cache and pays
full price for the prefix. A skill that names no additional tools costs you
nothing here, because its instructions come back as a tool response and leave
the prefix alone.

**`metadata.adk_inject_state`** must be a boolean. Set it to `true` and the skill
body is rendered through
`inject_session_state` at the moment
the model calls `load_skill`, which replaces `{dev_name}` in the body with the
value of `dev_name` in session state as it stands at that point. Writing `{var?}`
instead substitutes an empty string when the key is missing rather than raising,
and for a skill that is usually the behavior you want, because a skill often
loads before every key it mentions has been set.

**`allowed_tools`** is stored and exposed but ADK does not enforce it. It comes
from the Agent Skills specification, and it is kept so that a skill folder
survives a round trip through ADK unchanged.

## Resources

`Resources` is three dictionaries and a handful of accessors over them, one
dictionary per subdirectory the loader recognizes:

| Member | Type | Description |
| :--- | :--- | :--- |
| `references` | `dict[str, str \| bytes]` | Markdown or text guidance the model reads. |
| `assets` | `dict[str, str \| bytes]` | Data files: schemas, templates, examples. |
| `scripts` | `dict[str, Script]` | Executable source, run through the toolset's code executor or environment. |

`get_reference`, `get_asset`, and `get_script` all return `None` for a missing
key rather than raising, and `list_references`, `list_assets`, and `list_scripts`
give you the keys. `Script` wraps a single field, `src`, and its `__str__`
returns that source, so you can drop a `Script` straight into a prompt without
unwrapping it.

The split between references and assets is a convention rather than a mechanism.
References are what you expect the model to read as prose, and assets are what
you expect it to consult as data, but the same tool fetches both and nothing
enforces the distinction. Put a file wherever it will make more sense to the
next person reading the folder.

## Advanced applications

The sections below cover the ways of getting a `Skill` that are not the single
`load_skill_from_dir` call above: building one in Python, loading a whole tree
at once, reading only the frontmatter, reading from a bucket, and doing any of
those without blocking the event loop.

### Define a skill in Python

*   **Problem solved**: the skill content is generated, or it comes from a
    database, or it is short enough that a whole folder feels like too much
    ceremony.
*   **Implementation**: construct the models directly. Nothing requires a file on
    disk, though the name still has to be kebab-case.

```python
from google.adk.skills import Frontmatter
from google.adk.skills import Resources
from google.adk.skills import Skill

support_hours_skill = Skill(
    frontmatter=Frontmatter(
        name="support-hours-skill",
        description=(
            "A skill to check customer support hours for a given location."
        ),
        metadata={"adk_additional_tools": ["get_timezone"]},
    ),
    instructions=(
        "Step 1: Look up the timezone for the user's location using"
        " 'get_timezone'. Step 2: Read 'references/support_policy.txt' to"
        " understand the support hours policy. Step 3: Explain the support"
        " hours relative to the location's timezone."
    ),
    resources=Resources(
        references={
            "support_policy.txt": (
                "Customer support is available Monday through Friday, "
                "from 9:00 AM to 5:00 PM local time."
            ),
        },
    ),
)
```

### Load a whole directory of skills

*   **Problem solved**: you keep a `skills/` tree and you want every skill in it.
*   **Implementation**: `load_skills_from_dir` walks the immediate
    subdirectories in sorted order and loads each one that contains a `SKILL.md`.
    A subdirectory without one is skipped silently, but a subdirectory that has a
    `SKILL.md` and fails validation raises and takes the whole call down with it,
    so one broken skill means none of them load.

```python
from google.adk.skills import load_skills_from_dir

skills = load_skills_from_dir(pathlib.Path(__file__).parent / "skills")
```

### List without loading

*   **Problem solved**: you want a catalog of names and descriptions without
    paying to read every body, every reference, and every script.
*   **Implementation**: `list_skills_in_dir` returns `dict[str, Frontmatter]`,
    reading only the frontmatter of each `SKILL.md`. It is far more forgiving
    than `load_skills_from_dir`, because an invalid skill is logged and skipped,
    and a base path that is not a directory produces a warning and an empty dict
    instead of an exception.

### Load from Cloud Storage

*   **Problem solved**: skills are deployed separately from the agent, or shared
    across several agents.
*   **Implementation**: `list_skills_in_gcs_dir` and `load_skill_from_gcs_dir`
    mirror the local functions against a bucket. Both require the optional
    `google-cloud-storage` package and raise `ImportError` with an install hint
    if it is absent.

```python
from google.adk.skills import list_skills_in_gcs_dir
from google.adk.skills import load_skill_from_gcs_dir

available = list_skills_in_gcs_dir(
    bucket_name="my-skills-bucket", skills_base_path="static-skills"
)
skills = [
    load_skill_from_gcs_dir(
        bucket_name="my-skills-bucket",
        skills_base_path="static-skills",
        skill_id=skill_id,
    )
    for skill_id in available
]
```

The GCS layout mirrors the local one, with one prefix per skill, `SKILL.md`
directly under it, and `references/`, `assets/` and `scripts/` beneath that. The
last path segment of the prefix plays the part the directory name plays locally,
so it has to equal the declared `name`.

### Load off the event loop

*   **Problem solved**: every loader above is blocking, so calling one from
    inside an async handler stalls the event loop. It is most noticeable with
    GCS, where loading a single skill is several network round trips.
*   **Implementation**: each loader has an `_async` twin that runs the blocking
    version in a worker thread. They are `load_skill_from_dir_async`,
    `load_skills_from_dir_async`, `list_skills_in_dir_async`,
    `load_skill_from_gcs_dir_async`, and `list_skills_in_gcs_dir_async`, and the
    arguments and return values are identical to their blocking counterparts.

## Limitations

*   **The directory name is part of the contract.** Renaming a skill means
    renaming the folder and the `name` field together, or the load fails.
*   **`allowed_tools` is inert.** ADK parses and preserves it but does not use
    it to restrict anything.
*   **Non-UTF-8 scripts are dropped.** A binary under `scripts/` is skipped with
    a log warning and never appears in `resources.scripts`. Under `references/`
    or `assets/` the same file would be kept as `bytes`.
*   **Scripts need somewhere to run.** Loading a skill with a `scripts/` folder
    works anywhere, but the model can only execute one if the `SkillToolset`
    was given a `code_executor` or an `environment`, or the agent has a
    `code_executor`. Otherwise `run_skill_script` returns a `NO_CODE_EXECUTOR`
    error to the model.
*   **`metadata.adk_additional_tools` invalidates a context cache.** The tool
    declarations are part of the cached prefix, so the request after a skill
    activates a tool misses the cache. See the field description above.
*   **State injection is opt-in and load-time.** Without
    `metadata.adk_inject_state: true`, a `{placeholder}` in the body is passed
    to the model literally. With it, the substitution happens when the model
    calls `load_skill`, so later changes to session state do not re-render an
    already-loaded skill.
*   **Skill names are kebab-case by default.** Snake_case requires enabling the
    `SNAKE_CASE_SKILL_NAME` feature; see
    the feature registry guide.
*   **Experimental.** The package's own
    [README](../../../../src/google/adk/skills/README.md) marks skills as
    experimental and under active development, so the API may change without
    notice.
*   **Importing `DEFAULT_SKILL_SYSTEM_INSTRUCTION` from `google.adk.skills` is
    deprecated.** It emits a `DeprecationWarning`; import it from
    `google.adk.tools.skill_toolset` instead.

## Related samples

*   [Skills](../../../../contributing/samples/environment_and_skills/skills) is a
    real `skills/` tree, with a directory-loaded skill and a Python-defined one
    combined in a single toolset.
*   [Skills agent](../../../../contributing/samples/environment_and_skills/skills_agent)
    is a small agent driven entirely by skills.
*   [Skills agent (GCS)](../../../../contributing/samples/environment_and_skills/skills_agent_gcs)
    lists and loads its skills from a Cloud Storage bucket.
*   [Skills with state injection](../../../../contributing/samples/environment_and_skills/skills_inject_state)
    has a `SKILL.md` that personalizes itself from session state.
