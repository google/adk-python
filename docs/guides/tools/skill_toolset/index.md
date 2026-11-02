# SkillToolset

The `SkillToolset` class gives an agent a set of skills, which are packaged
instructions, reference files and scripts that the agent loads only when a
request calls for them. The agent knows each skill's name and description from
the start, and it reads a skill's full instructions only after it decides that
the skill applies.

## Introduction

A skill is a folder with a `SKILL.md` file and, optionally, reference
documents, assets and scripts.
[Skill, Frontmatter, and Resources](../../skills/skill/index.md) describes what
goes into a skill and how to write one. The `SkillToolset` class is the part
that makes skills available to an agent. When you add it to an agent's `tools`,
the agent gains the following tools:

- The `load_skill` tool returns a skill's instructions.
- The `load_skill_resource` tool returns a file from a skill's `references/`,
  `assets/` or `scripts/` folder.
- The `list_skills` tool returns the name and description of every skill. It is
  present only in the default discovery mode.
- The `run_skill_script` tool runs one of a skill's scripts. It is present only
  when the toolset has somewhere to run scripts.
- The `search_skills` tool searches a skill registry. It is present only when
  you pass a registry.

The toolset also adds a block of text to the system instruction that tells the
model what skills are and how to use these tools.

The `google.adk.tools` package does not export the `SkillToolset` class. Import
it from its own module, `google.adk.tools.skill_toolset`.

## Get started

The simplest setup loads every skill in a folder next to your agent. In the
following layout, `skills/` holds one skill, and each subfolder of `skills/`
becomes one skill:

```
my_agent/
  agent.py
  skills/
    weather-skill/
      SKILL.md
      references/
        weather_info.md
```

The agent definition loads the folder and passes the skills to the toolset:

```python
import pathlib

from google.adk.agents import LlmAgent
from google.adk.skills import load_skills_from_dir
from google.adk.tools.skill_toolset import SkillToolset

skills = load_skills_from_dir(pathlib.Path(__file__).parent / "skills")

root_agent = LlmAgent(
    name="assistant",
    instruction="Use your skills when a request matches one of them.",
    tools=[SkillToolset(skills=skills)],
)
```

You run the agent in the usual way, for example with `adk web`.

This agent can list skills, load them and read their reference files. It cannot
run a skill's scripts yet, because it has nowhere to run them. In that case ADK
leaves `run_skill_script` out of the tool list and tells the model that it can
read a script but not run it. The section
[Give scripts somewhere to run](#give-scripts-somewhere-to-run) explains how to
add an environment or a code executor.

## How it works

When the model decides that a skill applies, the following steps take place:

1. The model calls `load_skill` with the skill's name. The tool returns the
   skill's instructions and its frontmatter.
2. ADK records in session state that the skill is active for this agent. ADK
   keeps the record per agent name, so two agents in the same session activate
   skills independently.
3. Before the next model call, the toolset rebuilds its tool list. If the
   active skill names extra tools in the `adk_additional_tools` entry of its
   frontmatter, those tools are now in the list.

Most toolsets build their tool list once per invocation. A `SkillToolset`
rebuilds its list before every model call, and that is what lets a skill's
extra tools appear in the same turn in which the model loaded the skill. The
cost is a small amount of work on every model call.

## Configuration options

The `skills` argument is the only positional one, and every other argument is
keyword-only.

### Supply the skills

Pass the skills that the agent should have in `skills`. Two skills with the same
name raise `ValueError` when you build the toolset.

Pass a `SkillRegistry` in `registry` when your skills live in a catalog that the
agent should search at runtime. The toolset then adds the `search_skills` tool,
and `load_skill` fetches a skill from the registry when the name is not among
the local skills. When a local skill and a registry skill share a name, the
local one wins, and search results leave the registry copy out.
[SkillRegistry](../../skills/skill_registry/index.md) describes how to implement
a registry.

### Give scripts somewhere to run

A skill's scripts need somewhere to run, and the toolset supports the following
kinds of backend. Pass one of them if any of your skills ships a script.

- The `environment` argument takes a `BaseEnvironment`. The `LocalEnvironment`
  class from `google.adk.environment` runs commands on the host machine. The
  `E2BEnvironment` class from `google.adk.integrations.e2b` and the
  `DaytonaEnvironment` class from `google.adk.integrations.daytona` run them in
  a remote sandbox.
- The `code_executor` argument takes a `BaseCodeExecutor`. ADK sends the
  executor a small Python program that unpacks the skill's files into a
  temporary folder and runs the script there. ADK supports Python and shell
  scripts on this path.

The two are mutually exclusive, and passing both raises `ValueError`. If you
pass neither, `run_skill_script` falls back to the agent's own `code_executor`.
If the agent has none either, ADK leaves the tool out of the tool list.

The `skills_folder` argument sets where ADK copies skill files inside an
environment. It defaults to a `skills` folder under the environment's working
directory. It must be an absolute path, and it is valid only together with
`environment`. ADK raises `ValueError` in either case otherwise.

The `script_timeout` argument is the number of seconds a script may run, 300 by
default. It applies to every command in an environment and to shell scripts on
the code-executor path. It does not limit a Python script on the code-executor
path, which runs inside the executor's own process.

Choose the backend with the risk in mind. In environment mode the model writes
the shell command itself, so the command can do anything the environment
allows. A `LocalEnvironment` and a local code executor such as
`UnsafeLocalCodeExecutor` both run on the machine that hosts your agent with no
isolation. Use a sandboxed environment or executor when the skills, or the
users talking to the agent, are not fully trusted.

### Let a skill bring its own tools

Some tools are useful only while a particular skill is in use. A skill names
such tools under `metadata.adk_additional_tools` in its frontmatter, and you
pass the tools themselves to the toolset in `additional_tools`.
[Skill, Frontmatter, and Resources](../../skills/skill/index.md) shows both
halves with a working example.

The `additional_tools` argument accepts `BaseTool` instances, `BaseToolset`
instances and plain functions. The toolset wraps a plain function in a
`FunctionTool` and matches it by its function name. If a name in the
frontmatter matches no tool, the toolset skips it without a warning, so check
the spelling if a tool does not appear.

When a skill that brings extra tools becomes active, or is released, the tool
declarations change, and the next request misses the prompt cache.

### Choose how the model discovers skills

The `discovery_mode` argument decides how the model learns which skills exist.
It takes a `SkillDiscoveryMode` from `google.adk.tools.skill_toolset`. The
default value, `SkillDiscoveryMode.LAZY`, gives the model the `list_skills`
tool. The `SkillDiscoveryMode.EAGER` value writes the catalog into the system
instruction and removes `list_skills`, so the model can load a skill
immediately. [Skill, Frontmatter, and Resources](../../skills/skill/index.md)
explains when to choose each mode, and why you should not filter `list_skills`
out with `tool_filter` to get the eager behavior.

### Control how long skills stay active

The `lifecycle_config` argument takes a `SkillLifecycleConfig`, which decides
how long a loaded skill stays active. Each skill gets one of the following modes
from `SkillLifecycleMode`:

- The `PERSISTENT` mode is the default. The skill stays active for the rest of
  the session.
- The `BOUNDED` mode works like `PERSISTENT`, except that at most
  `max_active_skills` bounded skills can be active for an agent at once.
  Loading one more releases the bounded skill that was loaded least recently,
  and loading an active skill again counts as a new use.
- The `EPHEMERAL` mode keeps the skill active for the rest of the turn that
  loaded it and releases it when the next turn begins.

The following configuration makes skills bounded by default, allows three at
once, and makes one skill ephemeral:

```python
from google.adk.tools.skill_toolset import SkillLifecycleConfig
from google.adk.tools.skill_toolset import SkillLifecycleMode
from google.adk.tools.skill_toolset import SkillToolset

toolset = SkillToolset(
    skills=skills,  # your own list of skills
    lifecycle_config=SkillLifecycleConfig(
        default_mode=SkillLifecycleMode.BOUNDED,
        max_active_skills=3,
        skill_overrides={"incident-report": SkillLifecycleMode.EPHEMERAL},
    ),
)
```

The `default_mode` field applies to every skill that `skill_overrides` does not
name, and the names in `skill_overrides` can include registry skills. Setting
`enabled=False` makes every skill persistent, whatever the rest of the
configuration says.

When you set the `revalidate_skills` field of `SkillLifecycleConfig` to `True`,
ADK re-states an active skill whose definition has changed since the model
loaded it, as [Skill, Frontmatter, and Resources](../../skills/skill/index.md)
describes. The field has no effect when `enabled` is `False`. It costs a
registry lookup for each active registry skill on each turn, and each
re-statement changes the system instruction, which causes a prompt-cache miss.

The `SKILL_LIFECYCLE` feature flag turns on related behaviors that are
experimental and off by default. When you turn on the flag, for example by
setting the environment variable `ADK_ENABLE_SKILL_LIFECYCLE=1`, the toolset
adds an `unload_skill` tool with which the model can release a skill it has
finished with. ADK also replaces the instructions of every released skill in the
history it sends to the model with a short notice, while the stored session
events keep the original text. Without the flag, the lifecycle modes still
decide which skills' tools are available, but a released skill's instructions
stay in the conversation history.

### Rename or filter the skill tools

The `tool_name_prefix` argument puts `<prefix>_` in front of the name of every
tool the toolset returns, including the extra tools that skills bring. The text
that the toolset adds to the system instruction uses the prefixed names. Use a
prefix when two skill toolsets are on the same agent.

The `tool_filter` argument limits which tools the agent sees, and it matches
names without the prefix. The filter also applies to the extra tools that
skills bring, so a list of names has to include those tools as well, or they
never appear. When the filter removes one of the skill tools, the system
instruction tells the model not to call it.

## Advanced applications

The basic setup leaves activation to the model. You can also activate and
release skills from your own code.

### Activate a skill from your own code

The toolset has methods for managing skills without going through the model.
Calling `await toolset.load_skill(ctx, name)` activates a skill for the agent in
`ctx`, calling `toolset.unload_skill(ctx, name)` releases it, and calling
`toolset.list_active_skills(ctx)` returns the active names. You call them from a
tool or callback with the context it was given, for example to give the agent a
billing skill's tools only when the user's account allows it.

When you call the `load_skill` method this way, it registers the skill's extra
tools, but it does not put the skill's instructions into the conversation. The
model gets the tools without being told what they are for, so give it that
guidance yourself. The method raises `ValueError` if the skill does not exist.

## Limitations

- ADK records activation as one list per agent in session state, and parallel
  calls rewrite that list. If the model loads or unloads two skills in parallel
  in the same step, the last write wins, so one activation can be lost, or a
  skill can stay active while both calls report success. The race is a current
  limitation.
- ADK keys activation by the agent's name. If you rename an agent, its existing
  sessions lose their active skills.
- Under `run_live`, the whole stream is one invocation, so an ephemeral skill
  stays active until the stream ends.

## Related samples

- [Skills agent](../../../../contributing/samples/environment_and_skills/skills_agent/agent.py)
  combines a skill built in Python with one loaded from a folder, and uses
  `adk_additional_tools` to bring in a `BaseTool` and a plain function.
- [Skills with a local environment](../../../../contributing/samples/environment_and_skills/local_env_skill_toolset/agent.py)
  runs skill scripts through `LocalEnvironment`.
- [Skills with an E2B sandbox](../../../../contributing/samples/environment_and_skills/e2b_env_skill_toolset/agent.py)
  runs the same kind of agent against a remote sandbox.
- [Skills from Cloud Storage](../../../../contributing/samples/environment_and_skills/skills_agent_gcs/agent.py)
  loads skills from a bucket and runs scripts through the agent's Agent Engine
  sandbox code executor.
- [State-injected skills](../../../../contributing/samples/environment_and_skills/skills_inject_state/agent.py)
  uses `adk_inject_state` to personalize a skill's instructions for each
  session.
