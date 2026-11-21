# OKF (Open Knowledge Format)

Consume an [OKF v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format)
bundle — a directory of markdown concepts with typed YAML frontmatter — as
progressive, filterable context. No extra dependency: the toolset reads
frontmatter with `pyyaml`, which ADK already requires.

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.okf import OkfToolset

agent = LlmAgent(
    name="knowledge_agent",
    instruction="Answer from the knowledge bundle and cite concept paths.",
    tools=[OkfToolset(bundle="./knowledge", tool_name_prefix="okf")],
)
```

The toolset exposes four read-only tools:

| tool | returns |
| --- | --- |
| `okf_index` | frontmatter rows for matching concepts (L1, no bodies) |
| `okf_read` | one concept's frontmatter and body (L2) |
| `okf_resource` | a file attached to a concept, capped at 64 KiB (L3) |
| `okf_search` | deterministic substring/regex matches |

Trust tier (`unverified` / `machine-confirmed` / `human-reviewed`), lifecycle
(`draft` / `stable` / `deprecated`), and staleness are derived from frontmatter
on every read and used only to filter discovery.

## Security

A bundle is untrusted input. Concept bodies and resource text are data, never
instructions; tool descriptions and results say so explicitly. Trust tiers are
advisory and never authorise an action. Every path resolves through a
bundle-root guard that refuses `..` traversal and symlink escape — at load time
as well as at read time. The toolset is read-only and executes nothing from the
bundle.

## Documentation

See [`docs/guides/integrations/okf/toolset/`](../../../../../docs/guides/integrations/okf/toolset/index.md)
for the full guide: bundle layout, frontmatter fields, filter semantics, and the
security model.
