# ADK Skills - Agent Skills Standard Integration

A comprehensive guide to the ADK Skills module, which implements support for the [Agent Skills standard](https://agentskills.io) - an open format for extending AI agent capabilities.

## Table of Contents

- [Overview](#overview)
- [Agent Skills Standard](#agent-skills-standard)
- [Directory Structure](#directory-structure)
- [SKILL.md Specification](#skillmd-specification)
- [Progressive Disclosure Architecture](#progressive-disclosure-architecture)
- [ADK Implementation](#adk-implementation)
- [Usage Examples](#usage-examples)
- [Security Considerations](#security-considerations)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Overview

**Agent Skills** are organized folders of instructions, scripts, and resources that agents can discover and load dynamically to perform better at specific tasks. The ADK Skills module provides full support for this open standard.

### Key Benefits

| Stakeholder | Benefit |
|-------------|---------|
| **Skill Authors** | Build capabilities once, deploy across multiple agent products |
| **ADK Users** | Give agents new capabilities using standard SKILL.md format |
| **Teams & Enterprises** | Capture organizational knowledge in portable, version-controlled packages |

### Module Components

| Component | Description |
|-----------|-------------|
| `MarkdownSkill` | Load skills from SKILL.md files with YAML frontmatter |
| `AgentSkillLoader` | Discover and load skills from directories |
| `ScriptExecutor` | Safe sandboxed execution of bundled scripts |
| `SkillTool` | Wrap skills as ADK tools for LLM invocation |
| `SkillsManager` | Unified registry for skill management |

---

## Agent Skills Standard

### What Is a Skill?

At its core, a skill is a **folder containing a `SKILL.md` file** that provides:

- **Metadata**: `name` and `description` (minimum required)
- **Instructions**: Markdown documentation on how to perform a task
- **Optional resources**: scripts, templates, and reference materials

### Capabilities Enabled

1. **Domain Expertise** - Package specialized knowledge into reusable instructions
2. **New Capabilities** - Enable agents to create presentations, build MCP servers, analyze datasets
3. **Repeatable Workflows** - Turn multi-step tasks into consistent, auditable workflows
4. **Interoperability** - Reuse the same skill across different skills-compatible agent products

### Code Integration

Skills can include pre-written Python scripts and other code that agents execute deterministically. This approach proves more efficient than token-based generation for operations like:
- Sorting lists
- Extracting PDF form fields
- Data transformation
- File manipulation

---

## Directory Structure

### Minimal Structure

```
skill-name/
└── SKILL.md          # Required
```

### Full Structure with Optional Directories

```
skill-name/
├── SKILL.md          # Required: instructions + metadata
├── scripts/          # Optional: executable code
│   ├── extract.py
│   └── transform.sh
├── references/       # Optional: additional documentation
│   ├── REFERENCE.md
│   ├── FORMS.md
│   └── domain-specific.md
└── assets/           # Optional: static resources
    ├── templates/
    ├── images/
    └── data/
```

### Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| `scripts/` | Executable code agents can run. Should be self-contained, include helpful error messages, and handle edge cases gracefully. Supported languages: Python, Bash, JavaScript |
| `references/` | Additional documentation loaded on demand. Keep individual files focused for efficient context use |
| `assets/` | Static resources: templates, images, diagrams, data files, lookup tables, schemas |

---

## SKILL.md Specification

Every skill starts with **YAML frontmatter** followed by **Markdown content**.

### Basic Format

```markdown
---
name: skill-name
description: A description of what this skill does and when to use it.
---

# Skill Title

## When to use this skill
Use this skill when the user needs to...

## How to perform the task
1. Step one...
2. Step two...

## Examples
...
```

### Frontmatter Fields

| Field | Required | Constraints | Description |
|-------|----------|-------------|-------------|
| `name` | **Yes** | Max 64 characters. Lowercase letters, numbers, and hyphens only. | Short identifier for the skill |
| `description` | **Yes** | Max 1024 characters. Non-empty. | Describes what the skill does and when to use it (used for discovery) |
| `license` | No | - | License name or reference to bundled license file |
| `compatibility` | No | Max 500 characters | Environment requirements (product, system packages, network access, etc.) |
| `metadata` | No | Arbitrary key-value mapping | Additional metadata (author, version, etc.) |

### ADK-Specific Extensions

ADK extends the standard frontmatter with additional fields:

```yaml
---
name: advanced-pdf-processing
description: Advanced PDF processing with OCR and form filling capabilities.
license: Apache-2.0
compatibility: Requires Python 3.8+, Tesseract OCR

# Standard metadata
metadata:
  author: google-adk
  version: "2.0"
  category: documents

# ADK-specific extensions
adk:
  # Execution configuration
  config:
    max_parallel_calls: 5
    timeout_seconds: 120
    allow_network: true
    memory_limit_mb: 512

  # Tool declarations for scripts
  tools:
    - name: extract_text
      script: scripts/extract_text.py
      description: Extract text from PDF pages
      parameters:
        input_file: Path to PDF file
        pages: Optional page range (e.g., "1-5")
---
```

### Description Field Best Practices

The description should describe both **what the skill does** and **when to use it**.

**Good example:**
```yaml
description: Extracts text and tables from PDF files, fills PDF forms, and merges multiple PDFs. Use when working with PDF documents or when the user mentions PDFs, forms, or document extraction.
```

**Poor example:**
```yaml
description: PDF processing  # Too vague, no usage context
```

---

## Progressive Disclosure Architecture

Skills use a **context-efficient, three-stage approach** to information loading:

### Stage 1: Discovery (Startup)

- Agents load only the `name` and `description` of available skills
- Minimal context overhead (~100 tokens per skill)
- Enables agents to identify relevant skills without full loading

### Stage 2: Activation (Task Matching)

- When a task matches a skill's description, the agent reads the full `SKILL.md`
- Complete instructions are loaded into context
- Recommended: Keep under 5000 tokens for the body

### Stage 3: Execution (Implementation)

- Agent follows instructions
- Optionally loads referenced files from `scripts/`, `references/`, `assets/`
- Resources loaded only when required

**Benefit**: The amount of context that can be bundled into a skill is effectively unbounded since agents with filesystem access don't require everything in their context window simultaneously.

### Progressive Disclosure Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Progressive Disclosure in ADK                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: Discovery (~100 tokens per skill)                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ <available_skills>                                                     │  │
│  │   <skill>                                                              │  │
│  │     <name>pdf-processing</name>                                        │  │
│  │     <description>Extract text, fill forms, merge PDFs</description>   │  │
│  │   </skill>                                                             │  │
│  │ </available_skills>                                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                   │                                          │
│                      User: "I need to extract text from a PDF"              │
│                                   │                                          │
│                                   ▼                                          │
│  STAGE 2: Activation (~2000-5000 tokens)                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ LLM calls: skill_pdf_processing(action="activate")                     │  │
│  │                                                                        │  │
│  │ Returns full SKILL.md instructions:                                    │  │
│  │ - When to use this skill                                               │  │
│  │ - Prerequisites                                                        │  │
│  │ - Step-by-step instructions                                            │  │
│  │ - Available scripts: [extract_text.py, merge_pdfs.py]                 │  │
│  │ - Available references: [FORMS.md]                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                   │                                          │
│                      LLM reads instructions, decides to run script          │
│                                   │                                          │
│                                   ▼                                          │
│  STAGE 3: Execution (on-demand resources)                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ LLM calls: skill_pdf_processing(                                       │  │
│  │     action="run_script",                                               │  │
│  │     script="extract_text.py",                                          │  │
│  │     args=["--input", "document.pdf"]                                   │  │
│  │ )                                                                      │  │
│  │                                                                        │  │
│  │ ScriptExecutor runs extract_text.py in sandbox                        │  │
│  │ Returns: {"stdout": "Extracted text...", "success": true}             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ADK Implementation

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ADK Skills Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Skill Sources                                 │    │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │    │
│  │  │  SKILL.md Files │    │  BaseSkill      │    │  Remote Skills  │  │    │
│  │  │  (Agent Skills  │    │  Classes        │    │  (Future)       │  │    │
│  │  │   Standard)     │    │  (Python)       │    │                 │  │    │
│  │  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘  │    │
│  └───────────┼──────────────────────┼──────────────────────┼───────────┘    │
│              │                      │                      │                 │
│              ▼                      ▼                      ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      AgentSkillLoader                                │    │
│  │  • Discovers skills from directories                                │    │
│  │  • Parses SKILL.md frontmatter and content                          │    │
│  │  • Creates unified MarkdownSkill instances                          │    │
│  │  • Manages progressive disclosure stages                            │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        SkillsManager                                 │    │
│  │  • Unified registry for all skill types                             │    │
│  │  • Skill discovery and lookup                                       │    │
│  │  • Execution coordination                                           │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│              ┌────────────────────┼────────────────────┐                    │
│              ▼                    ▼                    ▼                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐     │
│  │   SkillTool     │  │  ScriptExecutor │  │  ProgrammaticTool       │     │
│  │   (LLM-facing)  │  │  (scripts/)     │  │  Executor (PTC)         │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### MarkdownSkill

Loads skills from SKILL.md files with progressive disclosure support:

```python
from google.adk.skills import MarkdownSkill

# Load a skill (Stage 1 - metadata only)
skill = MarkdownSkill.from_directory("/path/to/pdf-processing")

# Access metadata
print(skill.name)         # "pdf-processing"
print(skill.description)  # "Extract text from PDFs..."

# Stage 2: Full instructions
instructions = skill.get_instructions()

# Stage 3: Access scripts/references
script = skill.get_script("extract_text.py")
reference = skill.get_reference("FORMS.md")
```

#### AgentSkillLoader

Discovers and loads skills from directories:

```python
from google.adk.skills import AgentSkillLoader

loader = AgentSkillLoader()

# Discover skills from directories
loader.add_skill_directory("/path/to/skills")
loader.add_skill_directory("/path/to/custom-skills")

# Get discovered skills
skills = loader.get_all_skills()
print(loader.get_skill_names())  # ['pdf-processing', 'data-analysis', ...]

# Generate discovery prompt for LLM
prompt = loader.generate_discovery_prompt()
```

#### ScriptExecutor

Safely executes bundled scripts with sandboxing:

```python
from google.adk.skills import ScriptExecutor

executor = ScriptExecutor(
    timeout_seconds=30.0,
    allow_network=False,
)

result = await executor.execute_script(
    script_path=Path("/path/to/skill/scripts/extract.py"),
    args=["--input", "file.pdf"],
    working_dir=Path("/tmp/workspace"),
)

if result.success:
    print(result.stdout)
else:
    print(f"Error: {result.stderr}")
```

#### SkillTool

Wraps skills as ADK tools for LLM invocation:

```python
from google.adk.skills import SkillTool, MarkdownSkill

skill = MarkdownSkill.from_directory("/path/to/pdf-processing")
tool = SkillTool(skill)

# LLM can invoke with these actions:
# - {"action": "activate"} → Returns full instructions
# - {"action": "run_script", "script": "extract.py", "args": [...]}
# - {"action": "load_reference", "reference": "FORMS.md"}
```

### Module Exports

```python
from google.adk.skills import (
    # Core abstractions
    BaseSkill,
    SkillConfig,
    SkillsManager,
    SkillInvocationResult,
    # Agent Skills standard support
    MarkdownSkill,
    MarkdownSkillMetadata,
    AgentSkillLoader,
    # Execution
    ScriptExecutor,
    ScriptExecutionResult,
    ScriptExecutionError,
    # Tool integration
    SkillTool,
    create_skill_tools,
    # Path to bundled skills
    SKILLS_DIR,
)
```

---

## Usage Examples

### Basic Usage with LlmAgent

```python
from google.adk.agents import LlmAgent
from google.adk.skills import AgentSkillLoader, SkillTool, SkillsManager

# Load Agent Skills standard skills
loader = AgentSkillLoader()
loader.add_skill_directory("./skills")

# Create skills manager and register
skills_manager = SkillsManager()
loader.register_all(skills_manager)

# Convert skills to tools for LLM
skill_tools = [
    SkillTool(skill) for skill in skills_manager.get_all_skills()
]

# Create agent with skills
agent = LlmAgent(
    name="skilled_agent",
    model="gemini-2.0-flash",
    instruction=f"""You are a helpful assistant with access to skills.

{loader.generate_discovery_prompt()}

To use a skill:
1. First activate it to get full instructions
2. Then use run_script or load_reference as needed
""",
    tools=skill_tools,
)
```

### Using Built-in Skills

```python
from google.adk.skills import AgentSkillLoader, SkillTool, SKILLS_DIR

# Load built-in skills (bqml, bq-ai-operator)
loader = AgentSkillLoader()
loader.add_skill_directory(SKILLS_DIR)

print(f"Loaded skills: {loader.get_skill_names()}")
# Output: ['bq-ai-operator', 'bqml']

# Create tools
skill_tools = [SkillTool(s) for s in loader.get_all_skills()]
```

### Using Anthropic Skills

```python
# Clone the Anthropic skills repository first:
# git clone https://github.com/anthropics/skills ./anthropic-skills

from google.adk.skills import AgentSkillLoader, SkillTool

# Load skills from Anthropic's repository
loader = AgentSkillLoader()
loader.add_skill_directory("./anthropic-skills/skills")

# Check what was loaded
print(f"Discovered {len(loader.get_skill_names())} skills:")
for name in loader.get_skill_names():
    skill = loader.get_skill(name)
    print(f"  - {name}: {skill.description[:50]}...")
```

### Complete Demo Agent

See `contributing/samples/agent_skills_demo/` for a complete example that:
- Loads built-in BQML and BQ AI Operator skills
- Integrates with BigQuery toolset
- Demonstrates progressive disclosure in action

---

## Security Considerations

Script execution introduces security risks. The ADK Skills module implements multiple safeguards:

### ScriptExecutor Security Features

| Feature | Description |
|---------|-------------|
| **Execution Timeout** | Configurable timeout prevents runaway scripts |
| **Working Directory Isolation** | Scripts run in specified directories |
| **Environment Filtering** | Only allowed environment variables passed through |
| **Container Sandboxing** | Optional Docker isolation for untrusted scripts |
| **Memory Limits** | Configurable memory limits for containerized execution |
| **Network Isolation** | Optional network access restriction |

### Best Practices

| Measure | Description |
|---------|-------------|
| **Sandboxing** | Use `use_container=True` for untrusted skills |
| **Allowlisting** | Only execute scripts from trusted skills |
| **Confirmation** | Ask users before running potentially dangerous operations |
| **Logging** | Record all script executions for auditing |
| **Source Verification** | Install skills only from trusted sources |
| **Audit** | Review bundled files, dependencies, and external network connections |

---

## Best Practices

### Writing Effective Skills

| Guideline | Description |
|-----------|-------------|
| **Clear Descriptions** | Write descriptions that help agents determine when to activate the skill |
| **Concise Instructions** | Keep the main SKILL.md focused; use references for detailed content |
| **Concrete Examples** | Include input/output examples to demonstrate expected behavior |
| **Handle Edge Cases** | Document known limitations and workarounds |
| **Self-Contained Scripts** | Scripts should document dependencies and include helpful error messages |
| **Logical Organization** | Group related information and use clear section headers |

### Context Efficiency

- Load metadata at startup (~100 tokens per skill)
- Keep main instructions under 5000 tokens
- Split large reference materials into separate files
- Use progressive disclosure - load details only when needed

### SKILL.md Template

```markdown
---
name: my-skill-name
description: Clear description of what this skill does and when to use it.
license: Apache-2.0
compatibility: List any environment requirements
metadata:
  author: your-name
  version: "1.0"
---

# Skill Title

## When to use this skill
Describe the situations where this skill should be activated.

## Prerequisites
List any required tools, packages, or access.

## Instructions
Step-by-step guide for performing the task.

## Examples
Concrete examples with inputs and outputs.

## Common Edge Cases
Known limitations and how to handle them.

## Additional Resources
- [Reference Guide](references/REFERENCE.md)
- [Scripts](scripts/)
```

---

## Resources

### Official Agent Skills Documentation

- **Agent Skills Website**: https://agentskills.io
- **Specification**: https://agentskills.io/specification
- **Integration Guide**: https://agentskills.io/integrate-skills

### GitHub Repositories

- **Agent Skills Framework**: https://github.com/agentskills/agentskills
- **Example Skills (Anthropic)**: https://github.com/anthropics/skills
- **Google ADK**: https://github.com/google/adk-python

### ADK Skills Module Files

```
src/google/adk/skills/
├── __init__.py              # Module exports
├── adk_skills.md            # This documentation
├── base_skill.py            # BaseSkill abstract class
├── skill_manager.py         # SkillsManager registry
├── markdown_skill.py        # SKILL.md file loader
├── agent_skill_loader.py    # Discovery and loading
├── script_executor.py       # Script execution
├── skill_tool.py            # Tool wrapper
├── bqml/                    # Built-in BQML skill
│   ├── SKILL.md
│   ├── scripts/
│   └── references/
└── bq-ai-operator/          # Built-in BQ AI Operator skill
    ├── SKILL.md
    ├── scripts/
    └── references/
```

### Related Documentation

- **Anthropic Engineering Blog**: https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- **Claude Support - Skills**: https://support.claude.com/en/articles/12512176-what-are-skills
