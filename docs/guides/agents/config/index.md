# Creating agents with configuration files

ADK can build an agent, or a whole multi-agent graph, from a YAML file instead
of Python. `from_config()` reads the file, resolves every field against the
target class, and returns a live agent you can run.

This is useful when the shape of a workflow changes more often than the code
underneath it: CI/CD pipelines, per-tenant routing, or an operator tuning a
graph without cutting a new package release.

## Get started

Two files -- a workflow and the agent it calls.

**root_agent.yaml**

```yaml
agent_class: Workflow
name: my_sample_workflow
edges:
  - - START
    - my_module.functions.process_data
    - sub_agent.yaml
```

**sub_agent.yaml**

```yaml
agent_class: LlmAgent
name: summarizer_agent
description: Summarizes incoming data payloads.
instruction: Please summarize the following input concisely.
```

Load it:

```python
from google.adk.agents import config_agent_utils

root_workflow = config_agent_utils.from_config("root_agent.yaml")
```

### Running it from the CLI

If the file is named `root_agent.yaml` and sits in a directory named after the
agent, the ADK CLI finds it with no Python at all:

```
my_agents/
  my_sample_workflow/
    root_agent.yaml
    sub_agent.yaml
```

```bash
adk run my_agents/my_sample_workflow      # interactive terminal session
adk web my_agents                         # dev UI, one entry per directory
adk api_server my_agents                  # HTTP server exposing every agent
```

The loader tries the Python forms first -- `{agent_name}/__init__.py` exposing a
`root_agent`, then `{agent_name}/agent.py` -- and reaches
`{agent_name}/root_agent.yaml` only when neither is present. A directory can
therefore move from Python to YAML without changing how it is launched, but the
Python definition has to go, or it keeps winning.

## Writing edges

`edges` is the part with syntax of its own, and it is worth reading before
writing a graph by hand. Each entry in the list is one of three things.

The first two are shorthands; the explicit form below can express anything they
can.

### A chain

A list of nodes. Consecutive pairs become edges, so a three-element chain is two
edges:

```yaml
edges:
  - - START
    - fetch.yaml
    - summarize.yaml     # START -> fetch, and fetch -> summarize
```

Write a chain in the block form above. An inline list -- `[a, b]` -- always
means a fan-out, so keeping the two apart on sight is worth the extra lines.

### A routing map

A mapping whose keys are route names, for a node that fans out by result:

```yaml
edges:
  - - START
    - classifier.yaml
    - refund: refund_handler.yaml
      question: faq_agent.yaml
      other: [logger.yaml, escalation_agent.yaml]   # a list is a fan-out
```

A key here is a route value, not a field name, so a route called `name` is read
as a route, not as an inline node. One exception: a map with a single entry
`code` whose value is a string is an agent reference, so a route by that exact
shape is not available.

### An explicit edge

The full form the two shorthands expand to: a mapping with `from_node` and
`to_node`, plus `route` when the source node picks between outgoing paths.

```yaml
edges:
  - - START
    - classifier.yaml                  # introduces the node named "classifier"
  - from_node: classifier
    to_node: refund_handler.yaml
    route: refund
```

### What can appear as a node

Anywhere a node is expected, all of these work:

| Form              | Example                            | Meaning           |
| ----------------- | ---------------------------------- | ----------------- |
| `START`           | `START`                            | the graph entry   |
:                   :                                    : point             :
| a name            | `summarizer_agent`                 | a node named      |
:                   :                                    : elsewhere in this :
:                   :                                    : file              :
| a config path     | `sub_agent.yaml`                   | another config,   |
:                   :                                    : relative to this  :
:                   :                                    : file              :
| a function        | `my_module.functions.process_data` | wrapped in a      |
: reference         :                                    : `FunctionNode`    :
| an inline mapping | `{agent_class: LlmAgent, name: x,  | a node defined in |
:                   : ...}`                              : place             :

Prefer one agent per file and refer to it by path: a graph reads better when the
node definitions are not inlined into it. The inline mapping is there for the
cases a path cannot cover.

A bare function reference does **not** need a `name`: the node takes the
function's own name, so `my_module.functions.process_data` becomes a node called
`process_data`, which is what later edges refer to. Use the mapping form only
when you want a different name or extra fields, and note that `name` is required
there:

```yaml
- name: preprocess
  agent_class: FunctionNode
  func_code: my_module.functions.process_data
```

Nodes are cached by name and by reference, so naming a node once in a chain and
referring to it again from a later edge gives you the same node, not a copy. The
definition has to come first: a name an earlier edge has not introduced is
rejected, not resolved later.

## How it works

### Choosing the class

The top-level `agent_class` names the class to build, defaulting to `LlmAgent`.
Shorthands (`Workflow`, `FunctionNode`, `LlmAgent`) resolve against
`google.adk.agents` and `google.adk.workflow`; anything else is treated as a
fully-qualified name and imported.

### Filling in the fields

Rather than each class parsing its own config, the mapper reads the target
class's field annotations and resolves each YAML value to whatever that field
expects -- a list of tools, a callback, a schema, sub-agents, a model. A field
added to a class is configurable immediately, with no parser to update.

Keys are validated against the class's config schema first, so a misspelled key
is reported rather than silently ignored. How strict that is depends on the
schema: the built-in configs forbid unknown keys, while a custom agent class
inherits `BaseAgentConfig`, which permits extras so the class can define its
own.

The node classes -- `Workflow` and `FunctionNode` -- declare no config schema at
all, so there is nothing to reject a stray key up front. There reflection is the
only gate, and a key it cannot place is logged as a warning naming the class and
the key.

### Referencing Python

Keys ending in `_code` (and `_callbacks`) hold a fully-qualified reference to
something in Python -- a function, a callback, a schema class. Each is checked
against the module denylist below before it is imported.

### Resolving sub-agents

`sub_agents` and `edges` are hydrated through the same mapper, so a referenced
config file is parsed exactly as a top-level one would be. Paths are relative to
the file that mentions them, which lets a repository organise agents into
folders (`writers/`, `critics/`) and reference across them.

## Security and limitations

### Module denylist

Every `_code` reference is resolved by name, which would otherwise be a direct
path to arbitrary code execution. Before importing, the top-level module is
checked against a denylist covering the entire standard library plus third-party
packages with known execution or deserialization entry points.

The whole standard library is blocked rather than a curated list of dangerous
modules, because a curated list does not hold: `cProfile.run`, `timeit.timeit`
and `trace.Trace.run` each execute a string handed to them, and every Python
release can add more. Agent configs have no legitimate need for the standard
library -- they name the agent's own package, `google.adk`, or an integration.

It is still a denylist, and a denylist cannot cover third-party packages in
general. Treat a config that can name arbitrary modules as trusted input.

### The `args` key

`args` is still a supported key -- it is how a tool or toolset configuration
passes constructor arguments. What is in flux is the guard around it, not the
key itself.

Because `args` can reach code execution there is a denylist for it, but that
denylist is **off by default** today (`_ENFORCE_YAML_KEY_DENYLIST` is `False`).
A host that loads configs it does not control should enable it explicitly.
Making it default-deny is in progress.

### Path handling

Absolute paths in file references are rejected, so a config can reach sibling
and descendant files rather than anywhere on disk.

## Related samples

-   [Workflow loop config](../../../contributing/samples/workflows/loop_config/README.md)
    -- looping and conditional routing declared in YAML.
-   [Multi-agent loop config](../../../contributing/samples/multi_agent/multi_agent_loop_config/README.md)
    -- sequential and loop workflows across several config files.
