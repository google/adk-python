# Agent Skills in ADK — Design & Implementation

**Status:** Implemented (PR #4575, `feat/execute-skill-script`)
**Spec:** [agentskills.io/specification](https://agentskills.io/specification)

---

## 1. Overview

ADK implements the open [Agent Skills](https://agentskills.io) standard —
the same specification adopted by Claude Code, OpenAI Codex, Gemini CLI,
GitHub Copilot, Cursor, and 20+ other platforms. A skill built for any
conforming platform works identically in ADK.

A skill is a directory containing instructions, resources, and scripts
that extend an agent's capabilities for specialized tasks. ADK surfaces
skills to the LLM through four tools in `SkillToolset`:

| Tool | Purpose |
|------|---------|
| `list_skills` | Discover available skills (names + descriptions) |
| `load_skill` | Read full SKILL.md instructions |
| `load_skill_resource` | Access individual files (references, assets, scripts) |
| `run_skill_script` | Execute Python or shell scripts from `scripts/` |

---

## 2. Directory Structure (Spec-Compliant)

```
my-skill/
├── SKILL.md              # Required — YAML frontmatter + markdown instructions
├── references/           # Optional — additional documentation
│   └── api-guide.md
├── assets/               # Optional — templates, schemas, data files
│   └── schema.json
└── scripts/              # Optional — executable code
    ├── analyze.py
    └── setup.sh
```

The directory name **must match** the `name` field in the SKILL.md
frontmatter. ADK validates this at load time.

---

## 3. SKILL.md Format & Metadata

Each skill's `SKILL.md` contains YAML frontmatter followed by markdown:

```yaml
---
name: my-skill
description: What this skill does and when to use it.
license: Apache-2.0
compatibility: Requires Python 3.10+
metadata:
  category: data-analysis
  author: team-x
allowed-tools: Bash(python *) Read
---

# My Skill Instructions

Step-by-step instructions the agent follows...
```

### Frontmatter Fields

| Field | Required | Constraints | Purpose |
|-------|----------|-------------|---------|
| `name` | Yes | 1-64 chars, kebab-case, must match directory name | Unique skill identifier |
| `description` | Yes | 1-1024 chars | Discovery — helps LLM decide when to use the skill |
| `license` | No | Free-form string | License information |
| `compatibility` | No | Max 500 chars | Environment requirements |
| `metadata` | No | `dict[str, str]` | Client-specific key-value pairs |
| `allowed-tools` | No | Space-delimited tool patterns | Tools the skill requires |

### Validation

ADK validates frontmatter via Pydantic models in `skills/models.py`:

- **Name format:** Lowercase letters, numbers, hyphens only. No leading/
  trailing/consecutive hyphens.
- **Name-directory match:** `name` field must equal the parent directory
  name.
- **Required fields:** `name` and `description` must be non-empty.
- **No unknown keys:** Extra frontmatter keys produce validation errors.
- **Duplicate detection:** `SkillToolset` rejects duplicate skill names
  at initialization time.

---

## 4. Progressive Loading (Three-Tier Architecture)

The core design principle is **progressive disclosure** — the agent only
loads what it needs, when it needs it. This allows hundreds of skills to
be registered without significant context overhead.

```
┌──────────────────────────────────────────────────────────────────┐
│                    Context Window Budget                         │
│                                                                  │
│  Tier 1: ~50-100 tokens per skill (always loaded)               │
│  ┌────────────────────────────────────────────────┐              │
│  │  <skill>                                       │              │
│  │    <name>my-skill</name>                       │  list_skills │
│  │    <description>What it does</description>     │              │
│  │  </skill>                                      │              │
│  └────────────────────────────────────────────────┘              │
│                         │                                        │
│                   Agent decides to use skill                     │
│                         ▼                                        │
│  Tier 2: ~2,000-5,000 tokens (loaded on demand)                 │
│  ┌────────────────────────────────────────────────┐              │
│  │  Full SKILL.md body with step-by-step          │  load_skill  │
│  │  instructions, examples, workflows             │              │
│  └────────────────────────────────────────────────┘              │
│                         │                                        │
│                   Agent follows instructions                     │
│                         ▼                                        │
│  Tier 3: Variable size (loaded as needed)                        │
│  ┌────────────────────────────────────────────────┐              │
│  │  Individual files from references/,            │  load_skill  │
│  │  assets/, scripts/                             │  _resource   │
│  │                                    ┌───────────┤              │
│  │  Script execution with args        │run_skill  │              │
│  │                                    │_script    │              │
│  └────────────────────────────────────┴───────────┘              │
└──────────────────────────────────────────────────────────────────┘
```

### How Each Tier Works

**Tier 1 — Discovery (~100 tokens/skill)**

At startup, `SkillToolset.process_llm_request()` injects a system
instruction listing all skills as XML:

```xml
<available_skills>
<skill>
  <name>statistical-calc</name>
  <description>Compute descriptive statistics for numeric datasets.</description>
</skill>
<skill>
  <name>log-parsing</name>
  <description>Parse and analyze structured log files.</description>
</skill>
</available_skills>
```

Only the `name` and `description` from the frontmatter are included.
Full instructions and resources are not loaded. This means **registering
100 skills costs ~5,000-10,000 tokens** — a small fraction of a typical
context window.

**Tier 2 — Activation (<5,000 tokens)**

When the LLM identifies a relevant skill, it calls `load_skill`:

```json
{"name": "statistical-calc"}
```

This returns the full SKILL.md markdown body along with the frontmatter.
The agent now has step-by-step instructions for using the skill.

**Tier 3 — Execution (variable)**

The agent accesses individual resources on demand:

- `load_skill_resource` — reads a specific file from `references/`,
  `assets/`, or `scripts/`
- `run_skill_script` — executes a script with structured arguments

Resources are loaded individually, not in bulk. A skill with 10
reference files only loads the ones the agent actually needs.

### Why This Matters

Without progressive loading, every skill's full content would need to
be in the system prompt. For 50 skills averaging 3,000 tokens each,
that's 150,000 tokens before the conversation even starts. With
three-tier loading, the same 50 skills cost ~5,000 tokens at Tier 1,
and only the actively-used skill's content enters the context.

---

## 5. Skill Script Execution

`RunSkillScriptTool` enables agents to execute code bundled with skills.
This is the key differentiator that turns skills from passive
instruction sets into active, executable tools.

### Architecture

```
LLM calls run_skill_script(skill_name, script_path, args)
        │
        ▼
┌─ RunSkillScriptTool.run_async() ─────────────────────────┐
│  1. Validate params (skill_name, script_path, args)      │
│  2. Resolve skill → locate script in resources           │
│  3. Resolve executor: toolset → agent fallback           │
│  4. Build self-extracting wrapper code                   │
│  5. await asyncio.to_thread(executor.execute_code, ...)  │
│  6. Parse result (JSON envelope for shell scripts)       │
│  7. Return {stdout, stderr, status}                      │
└──────────────────────────────────────────────────────────┘
```

### Parameter Design

```json
{
  "skill_name": "statistical-calc",
  "script_path": "scripts/stats.py",
  "args": {"data": "10,20,30,40,50"}
}
```

**`script_path`** uses the full relative path (not just the filename)
so scripts can access sibling resources (`references/`, `assets/`)
via relative paths from the skill root.

**`args`** is a structured JSON object, not a raw string. This design:
- Improves LLM reliability (structured JSON > command-line flag arrays)
- Eliminates shell injection (args are flattened to
  `['--key', 'value']` arrays, passed with `shell=False`)

### Script Types

| Type | Extension | Execution Method | Timeout |
|------|-----------|------------------|---------|
| Python | `.py` | `runpy.run_path()` via code executor | Executor-level |
| Shell | `.sh`, `.bash` | `subprocess.run()` with JSON envelope | `script_timeout` (default 300s) |
| Other | any | Rejected with `UNSUPPORTED_SCRIPT_TYPE` | N/A |

### Self-Extracting Wrapper

The tool generates a self-contained Python script that:

1. **Materializes** all skill files (references, assets, scripts) into
   a temporary directory
2. **Sets working directory** to the temp dir so relative paths work
3. **Executes** the target script with proper argument injection
4. **Captures** stdout/stderr through the code executor

This design is executor-agnostic — the same wrapper works with
`UnsafeLocalCodeExecutor`, `ContainerCodeExecutor`,
`VertexAiCodeExecutor`, or any `BaseCodeExecutor` implementation.

### Shell Script JSON Envelope

Shell scripts face a challenge: code executors capture stdout via
`redirect_stdout`, but stderr and exit codes need separate channels.
The wrapper solves this by serializing both streams as JSON through
stdout:

```json
{
  "__shell_result__": true,
  "stdout": "actual script output",
  "stderr": "any error messages",
  "returncode": 0
}
```

The tool parses this envelope and extracts the real stdout/stderr.
On timeout, the wrapper catches `TimeoutExpired`, captures partial
output, and returns a structured error — ensuring the LLM always
receives actionable feedback.

### Status Model

| Status | Condition |
|--------|-----------|
| `success` | No stderr, exit code 0 |
| `warning` | Both stdout and stderr present, exit code 0 |
| `error` | Non-zero exit code, or stderr-only output |

### Code Executor Resolution

```
1. SkillToolset(code_executor=...)    ← explicit, highest priority
2. agent.code_executor                ← fallback to agent's executor
3. None → NO_CODE_EXECUTOR error      ← actionable error message
```

### Security

- **Structured argument arrays** prevent shell injection
  (`subprocess.run` with `shell=False`)
- **`SystemExit` handling** prevents scripts from terminating the host
  (`sys.exit(0)` → success; `sys.exit(N)` → `EXECUTION_ERROR`)
- **`CancelledError`/`KeyboardInterrupt` propagation** — these are not
  swallowed; only `SystemExit` and `Exception` are caught
- **Pluggable executors** for isolation levels appropriate to the
  deployment context
- **Payload size guard** — warns when inlined resources exceed 16 MB

---

## 6. Spec Compliance

### Agent Skills Spec Alignment

| Spec Requirement | ADK Implementation |
|------------------|--------------------|
| `SKILL.md` with YAML frontmatter | `_parse_skill_md()` in `skills/_utils.py` |
| Required fields: `name`, `description` | Pydantic validation in `Frontmatter` model |
| `name` must be kebab-case, match directory | Custom validator + load-time check |
| Optional `references/` directory | `Resources.references` dict, loaded recursively |
| Optional `assets/` directory | `Resources.assets` dict, loaded recursively |
| Optional `scripts/` directory | `Resources.scripts` dict with `Script` model |
| Optional `license`, `compatibility` | Supported in `Frontmatter` model |
| Optional `metadata` dict | `Frontmatter.metadata: dict[str, str]` |
| Progressive loading (3 tiers) | `list_skills` → `load_skill` → `load_skill_resource` |
| Script execution | `run_skill_script` with `_SkillScriptCodeExecutor` |

### What ADK Adds Beyond the Spec

| Feature | Description |
|---------|-------------|
| `allowed-tools` frontmatter field | Declare tool dependencies (experimental) |
| Executor-agnostic script execution | Same skill works across local, container, Vertex AI, GKE |
| JSON envelope for shell scripts | Reliable stdout/stderr capture across all executors |
| Agent fallback executor chain | Skills work even without explicit executor config |
| LLM system instruction injection | `process_llm_request()` auto-injects skill list |

---

## 7. Data Model

```
Skill
├── frontmatter: Frontmatter     # Tier 1 — discovery metadata
│   ├── name: str                 #   kebab-case, 1-64 chars
│   ├── description: str          #   1-1024 chars
│   ├── license: Optional[str]
│   ├── compatibility: Optional[str]
│   ├── metadata: dict[str, str]
│   └── allowed_tools: Optional[str]
├── instructions: str             # Tier 2 — SKILL.md body (markdown)
└── resources: Resources          # Tier 3 — on-demand files
    ├── references: dict[str, str]
    ├── assets: dict[str, str]
    └── scripts: dict[str, Script]
                      └── src: str
```

---

## 8. Key Files

| File | Purpose |
|------|---------|
| `src/google/adk/skills/models.py` | `Skill`, `Frontmatter`, `Resources`, `Script` data models |
| `src/google/adk/skills/_utils.py` | `load_skill_from_dir()`, SKILL.md parsing, validation |
| `src/google/adk/skills/prompt.py` | `format_skills_as_xml()` for LLM prompt injection |
| `src/google/adk/tools/skill_toolset.py` | `SkillToolset` and all four tool implementations |
| `tests/unittests/tools/test_skill_toolset.py` | 61-test suite covering all tools and edge cases |
| `docs/design/skill_execution_script.md` | Design doc for script execution architecture |
| `docs/design/rfc_runskillscript_p0.md` | RFC for production-readiness (timeout, sandboxing) |

---

## 9. Usage Example

```python
from google.adk.skills import load_skill_from_dir, Skill
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.code_executors.unsafe_local_code_executor import (
    UnsafeLocalCodeExecutor,
)

# Load skills from disk
skill = load_skill_from_dir("./skills/statistical-calc")

# Create toolset with executor for script support
toolset = SkillToolset(
    skills=[skill],
    code_executor=UnsafeLocalCodeExecutor(),
    script_timeout=300,
)

# Attach to agent
agent = LlmAgent(
    name="analyst",
    model="gemini-2.0-flash",
    tools=[toolset],
)
```

The agent will automatically:
1. See "statistical-calc" in its available skills list
2. Load instructions when the user asks about statistics
3. Run `scripts/stats.py` with the user's data
4. Return formatted results

---

## 10. Future Work

See [RFC: Production-Readiness for RunSkillScriptTool](rfc_runskillscript_p0.md)
for planned improvements:

- **P0-A:** Uniform timeout support across all executors (including
  Python scripts)
- **P0-B:** `LocalSandboxCodeExecutor` — subprocess-based isolation
  with resource limits, replacing `UnsafeLocalCodeExecutor` as the
  recommended local default
- **`allowed_tools` resolution** — dynamically resolve tools declared
  in skill frontmatter from `additional_tools` or built-in tools
