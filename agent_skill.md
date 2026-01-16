# Agent Skills Specification

A comprehensive guide to understanding and implementing Agent Skills - an open standard for extending AI agent capabilities.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Directory Structure](#directory-structure)
- [SKILL.md Specification](#skillmd-specification)
- [Progressive Disclosure Architecture](#progressive-disclosure-architecture)
- [Integration Approaches](#integration-approaches)
- [LangChain/LangGraph Skills Integration](#langchainlanggraph-skills-integration)
- [ADK Skills Integration](#adk-skills-integration)
- [Security Considerations](#security-considerations)
- [Reference Library (skills-ref)](#reference-library-skills-ref)
- [Best Practices](#best-practices)
- [Example Skills](#example-skills)
- [Resources](#resources)

---

## Overview

**Agent Skills** are organized folders of instructions, scripts, and resources that agents can discover and load dynamically to perform better at specific tasks. They represent a simple, open format for giving agents new capabilities and expertise.

### What Problems Do Skills Solve?

1. **Limited Context**: Agents don't inherently have access to specialized knowledge
2. **Consistency**: Enable repeatable, auditable workflows
3. **Reusability**: Build once, deploy across multiple agent products
4. **Portability**: Same skill works across different compatible agent tools

### Key Benefits

| Stakeholder | Benefit |
|-------------|---------|
| **Skill Authors** | Build capabilities once, deploy across multiple agent products |
| **Compatible Agents** | Users can give agents new capabilities out of the box |
| **Teams & Enterprises** | Capture organizational knowledge in portable, version-controlled packages |

### Governance

- **Originally developed by**: Anthropic
- **Release status**: Open standard
- **Development model**: Open to ecosystem contributions
- **Repository**: https://github.com/agentskills/agentskills
- **Example Skills**: https://github.com/anthropics/skills

---

## Core Concepts

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
| `name` | **Yes** | Max 64 characters. Lowercase letters, numbers, and hyphens only. Must not start/end with hyphen or contain consecutive hyphens. Must match parent directory name. | Short identifier for the skill |
| `description` | **Yes** | Max 1024 characters. Non-empty. | Describes what the skill does and when to use it (used for discovery) |
| `license` | No | - | License name or reference to bundled license file |
| `compatibility` | No | Max 500 characters | Environment requirements (product, system packages, network access, etc.) |
| `metadata` | No | Arbitrary key-value mapping | Additional metadata (author, version, etc.) |
| `allowed-tools` | No | Space-delimited list | Pre-approved tools (Experimental) |

### Name Field Validation

**Valid examples:**
```yaml
name: pdf-processing
name: data-analysis
name: code-review
name: mcp-builder
```

**Invalid examples:**
```yaml
name: PDF-Processing    # uppercase not allowed
name: -pdf              # cannot start with hyphen
name: pdf--processing   # consecutive hyphens not allowed
name: pdf_processing    # underscores not allowed
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

### Complete Frontmatter Example

```yaml
---
name: pdf-processing
description: Extract text and tables from PDF files, fill forms, merge documents. Use when the user needs to work with PDF files.
license: Apache-2.0
compatibility: Requires Python 3.8+, pdfplumber, and PyPDF2 packages
metadata:
  author: example-org
  version: "1.0"
  category: documents
allowed-tools: Bash(python:*) Read Write
---
```

### Markdown Body Guidelines

- **No structural restrictions** on content format
- Can include text instructions, code examples, workflows, and references
- Self-documenting format allows easy auditing and improvement
- **Recommended maximum**: Keep main `SKILL.md` under 500 lines
- Move detailed reference material to separate files in `references/`

### Recommended Body Sections

```markdown
# Skill Name

## When to use this skill
Clear criteria for when this skill should be activated.

## Prerequisites
Any required tools, packages, or access needed.

## Instructions
Step-by-step guide for performing the task.

## Examples
Concrete examples of inputs and expected outputs.

## Common Edge Cases
Known limitations and how to handle them.

## File References
Links to additional resources in the skill directory.
```

### File References

Use relative paths from skill root:

```markdown
See [the reference guide](references/REFERENCE.md) for details.

Run the extraction script:
`scripts/extract.py`
```

**Recommendation**: Keep references one level deep; avoid deeply nested chains.

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

---

## Integration Approaches

To integrate Agent Skills support into your AI agent, implement five core steps:

1. **Discover** skills in configured directories
2. **Load metadata** (name and description) at startup
3. **Match** user tasks to relevant skills
4. **Activate** skills by loading full instructions
5. **Execute** scripts and access resources as needed

### Filesystem-Based Agents

- Operate within a computer environment (bash/unix)
- Most capable option
- Skills activated when models issue shell commands like `cat /path/to/my-skill/SKILL.md`
- Bundled resources accessed through shell commands

### Tool-Based Agents

- Function without a dedicated computer environment
- Implement tools allowing models to trigger skills and access bundled assets
- Specific tool implementation is up to the developer

### Implementation Steps

#### 1. Skill Discovery

Scan configured directories for folders containing a `SKILL.md` file:

```python
def discover_skills(skill_dirs):
    skills = []
    for dir in skill_dirs:
        for folder in os.listdir(dir):
            skill_path = os.path.join(dir, folder, "SKILL.md")
            if os.path.exists(skill_path):
                skills.append(skill_path)
    return skills
```

#### 2. Parse Metadata

At startup, parse only the frontmatter to keep initial context usage low:

```python
def parse_metadata(skill_path):
    content = read_file(skill_path)
    frontmatter = extract_yaml_frontmatter(content)

    return {
        "name": frontmatter["name"],
        "description": frontmatter["description"],
        "path": skill_path
    }
```

#### 3. Inject Metadata into System Prompt

Use XML format for the system prompt:

```xml
<available_skills>
  <skill>
    <name>pdf-processing</name>
    <description>Extracts text and tables from PDF files, fills forms, merges documents.</description>
    <location>/path/to/skills/pdf-processing/SKILL.md</location>
  </skill>
  <skill>
    <name>data-analysis</name>
    <description>Analyzes datasets, generates charts, and creates summary reports.</description>
    <location>/path/to/skills/data-analysis/SKILL.md</location>
  </skill>
</available_skills>
```

**Guidelines:**
- For filesystem-based agents: include the `location` field with absolute path
- For tool-based agents: omit the location field
- Keep metadata concise (~50-100 tokens per skill)

---

## LangChain/LangGraph Skills Integration

LangChain and LangGraph implement skills as a multi-agent pattern where specialized capabilities are packaged as invokable components that augment an agent's behavior. This section covers how skills work within the LangChain ecosystem.

### Overview

In LangChain/LangGraph, skills operate primarily through **prompt-driven specialization** rather than requiring full sub-agent implementations. A single agent loads specialized prompts and context on-demand while staying in control.

**Key Design Principles:**

1. **Prompt-Driven Specialization**: Skills are fundamentally defined by specialized prompts rather than complex implementations
2. **Progressive Disclosure**: Skills become available contextually based on user needs or agent reasoning
3. **Team Distribution**: Different teams can independently develop and maintain skills without tight coupling

### When to Use the Skills Pattern

The skills pattern is ideal for scenarios requiring:

- A single agent with numerous possible specializations
- No strict enforcement of constraints between capabilities
- Independent team development of domain-specific features

**Example Use Cases:**
- Coding assistants with language-specific skills (Python, JavaScript, Rust)
- Knowledge bases with domain skills (legal, medical, financial)
- Creative tools with format-specific skills (writing, design, music)

### Skills vs Other Multi-Agent Patterns

LangChain identifies five core patterns for multi-agent systems. Here's how skills compare:

| Pattern | Description | Performance | Best For |
|---------|-------------|-------------|----------|
| **Skills** | Single agent loads specialized prompts on-demand | 3 calls (one-shot), 5 calls (repeat) | Direct user interaction, moderate parallelization |
| **Subagents** | Main agent coordinates specialized subagents as tools | 4 calls (one-shot), 8 calls (repeat) | Distributed development, parallelization |
| **Handoffs** | Agent behavior changes dynamically based on state | 3 calls (one-shot), 5 calls (repeat) | Sequential multi-hop workflows |
| **Router** | Routing step classifies input and directs to specialized agents | 3 calls (one-shot), 6 calls (repeat) | Parallel execution with explicit routing |
| **Custom Workflow** | Bespoke execution flows mixing patterns | Varies | Complex hybrid requirements |

**Key Insight**: Skills, Handoffs, and Router patterns are most efficient for single tasks (3 calls each). Subagents adds one extra call because results flow back through the main agent.

### Skills Pattern Characteristics

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Parallelization** | ⭐⭐⭐ | Moderate - can load multiple skills but executes sequentially |
| **Direct User Interaction** | ⭐⭐⭐⭐⭐ | Excellent - single agent maintains conversation context |
| **Distributed Development** | ⭐⭐⭐⭐ | Good - teams can develop skills independently |
| **Context Accumulation** | Higher | Accumulates context over time (~15K tokens in multi-domain scenarios) |

### Basic Implementation

Skills in LangChain are implemented using a tool decorator pattern:

```python
from langchain_core.tools import tool

@tool
def load_skill(skill_name: str) -> str:
    """Load specialized skill prompt.

    Available skills:
    - write_sql: SQL query writing expertise
    - review_legal_doc: Legal document review
    - analyze_data: Data analysis and visualization
    """
    skills = {
        "write_sql": load_skill_content("sql_expert.md"),
        "review_legal_doc": load_skill_content("legal_review.md"),
        "analyze_data": load_skill_content("data_analysis.md"),
    }
    return skills.get(skill_name, f"Unknown skill: {skill_name}")

def load_skill_content(filename: str) -> str:
    """Load skill content from storage."""
    with open(f"skills/{filename}", "r") as f:
        return f.read()
```

The agent receives a system prompt indicating available skills and uses `load_skill` to access them on-demand.

### LangGraph Implementation

In LangGraph, skills can be implemented as nodes in the graph:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    active_skill: str
    skill_context: str

def skill_loader(state: AgentState) -> AgentState:
    """Load skill context based on detected need."""
    skill_name = state.get("active_skill")
    if skill_name:
        skill_content = load_skill_content(skill_name)
        return {"skill_context": skill_content}
    return {}

def agent_node(state: AgentState) -> AgentState:
    """Main agent with skill context."""
    skill_context = state.get("skill_context", "")
    # Agent uses skill_context to enhance its response
    response = llm.invoke(
        system_prompt + skill_context,
        state["messages"]
    )
    return {"messages": [response]}

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("skill_loader", skill_loader)
graph.add_node("agent", agent_node)
graph.add_edge("skill_loader", "agent")
```

### Extension Patterns

#### Dynamic Tool Registration

Loading a skill can simultaneously register new tools and update agent state:

```python
@tool
def load_skill_with_tools(skill_name: str) -> str:
    """Load skill and register associated tools."""
    skill_config = get_skill_config(skill_name)

    # Register skill-specific tools
    for tool_def in skill_config.get("tools", []):
        register_tool(tool_def)

    # Return skill instructions
    return skill_config["instructions"]
```

This enables progressive capability expansion as skills load.

#### Hierarchical Skills

Skills can define sub-skills in tree structures for fine-grained discovery:

```python
SKILL_HIERARCHY = {
    "data_science": {
        "description": "Data science and analytics capabilities",
        "sub_skills": {
            "pandas_expert": "DataFrame manipulation and analysis",
            "visualization": "Charts, plots, and data visualization",
            "statistical_analysis": "Statistical methods and hypothesis testing"
        }
    },
    "web_development": {
        "description": "Web application development",
        "sub_skills": {
            "frontend": "React, Vue, HTML/CSS",
            "backend": "APIs, databases, server logic",
            "devops": "Deployment, CI/CD, infrastructure"
        }
    }
}

@tool
def load_skill(skill_path: str) -> str:
    """Load skill by path (e.g., 'data_science/pandas_expert')."""
    parts = skill_path.split("/")
    # Navigate hierarchy and load appropriate skill
    return get_nested_skill(SKILL_HIERARCHY, parts)
```

### Integration with Agent Skills Standard

LangChain's skills pattern can integrate with the Agent Skills standard (SKILL.md format):

```python
import yaml
import os

def discover_agent_skills(skills_dir: str) -> dict:
    """Discover Agent Skills format skills."""
    skills = {}
    for folder in os.listdir(skills_dir):
        skill_md_path = os.path.join(skills_dir, folder, "SKILL.md")
        if os.path.exists(skill_md_path):
            with open(skill_md_path, "r") as f:
                content = f.read()

            # Parse YAML frontmatter
            if content.startswith("---"):
                _, frontmatter, body = content.split("---", 2)
                metadata = yaml.safe_load(frontmatter)
                skills[metadata["name"]] = {
                    "description": metadata["description"],
                    "content": body.strip(),
                    "path": skill_md_path
                }
    return skills

def create_langchain_skill_tool(skills: dict):
    """Create LangChain tool from Agent Skills."""
    skill_descriptions = "\n".join(
        f"- {name}: {info['description']}"
        for name, info in skills.items()
    )

    @tool
    def load_skill(skill_name: str) -> str:
        f"""Load specialized skill.

        Available skills:
        {skill_descriptions}
        """
        if skill_name in skills:
            return skills[skill_name]["content"]
        return f"Unknown skill: {skill_name}"

    return load_skill
```

### Context Engineering Considerations

At the center of multi-agent design is **context engineering** - deciding what information each agent sees.

**Skills Pattern Trade-offs:**

| Scenario | Token Usage | Model Calls |
|----------|-------------|-------------|
| Single domain task | ~5K tokens | 3 calls |
| Multi-domain task | ~15K tokens | 7+ calls |
| Repeat requests | Accumulates | 5 calls per request |

**Optimization Strategies:**

1. **Lazy Loading**: Only load skill content when explicitly needed
2. **Context Pruning**: Remove skill context after task completion
3. **Skill Summarization**: Use condensed skill versions for initial matching
4. **Caching**: Cache frequently-used skill content

### Comparison: LangChain Skills vs Agent Skills Standard

| Aspect | LangChain Skills | Agent Skills Standard |
|--------|------------------|----------------------|
| **Format** | Python code/prompts | SKILL.md files |
| **Discovery** | Docstring/config | YAML frontmatter |
| **Portability** | LangChain ecosystem | Cross-platform |
| **Execution** | Tool invocation | File system access |
| **Resources** | Python modules | scripts/, references/, assets/ |
| **Validation** | Custom | skills-ref library |

### Resources

**LangChain Documentation:**
- [Multi-Agent Patterns](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [Skills Pattern](https://docs.langchain.com/oss/python/langchain/multi-agent/skills)
- [LangGraph Workflows](https://docs.langchain.com/oss/python/langgraph/workflows-agents)

**LangGraph Resources:**
- [LangGraph Official Site](https://www.langchain.com/langgraph)
- [Multi-Agent Workflows Blog](https://www.blog.langchain.com/langgraph-multi-agent-workflows/)

---

## ADK Skills Integration

This section describes how the Google Agent Development Kit (ADK) integrates with the Agent Skills standard, enabling skills built using the SKILL.md format to be used directly as ADK Skills with full support for progressive disclosure, scripts, and assets.

### Design Goals

1. **Full Agent Skills Standard Support**: Load and execute skills defined using SKILL.md format
2. **Progressive Disclosure**: Three-stage loading (metadata → instructions → resources)
3. **Script & Asset Support**: Execute bundled scripts and access assets
4. **Bidirectional Compatibility**: ADK BaseSkill classes and SKILL.md files work interchangeably
5. **Programmatic Tool Calling (PTC)**: Enable efficient code-based tool orchestration
6. **Security**: Sandboxed execution with defense-in-depth

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

#### 1. MarkdownSkill Class

A concrete `BaseSkill` implementation that loads from SKILL.md files:

```python
# src/google/adk/skills/markdown_skill.py

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base_skill import BaseSkill, SkillConfig


class MarkdownSkillMetadata(BaseModel):
    """Metadata extracted from SKILL.md frontmatter."""

    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    allowed_tools: Optional[str] = None


class MarkdownSkill(BaseSkill):
    """Skill loaded from Agent Skills standard SKILL.md format.

    Supports progressive disclosure with three loading stages:
    - Stage 1 (Discovery): Only name and description loaded
    - Stage 2 (Activation): Full SKILL.md content loaded
    - Stage 3 (Execution): Scripts and references loaded on-demand

    Example:
        ```python
        skill = MarkdownSkill.from_directory("/path/to/pdf-processing")

        # Stage 1: Metadata only
        print(skill.name)         # "pdf-processing"
        print(skill.description)  # "Extract text from PDFs..."

        # Stage 2: Full instructions
        instructions = skill.get_instructions()

        # Stage 3: Access scripts/references
        script = skill.get_script("extract_text.py")
        reference = skill.get_reference("FORMS.md")
        ```
    """

    # Path to the skill directory
    skill_path: Path

    # Parsed frontmatter metadata
    skill_metadata: MarkdownSkillMetadata

    # Cached content (loaded on demand - Stage 2)
    _instructions_cache: Optional[str] = None

    # Scripts directory contents (loaded on demand - Stage 3)
    _scripts_cache: Dict[str, str] = Field(default_factory=dict)

    # References directory contents (loaded on demand - Stage 3)
    _references_cache: Dict[str, str] = Field(default_factory=dict)

    # Loading stage tracking
    _current_stage: int = 1  # 1=Discovery, 2=Activation, 3=Execution

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_directory(cls, skill_dir: str | Path) -> "MarkdownSkill":
        """Load a skill from a directory containing SKILL.md.

        Args:
            skill_dir: Path to the skill directory.

        Returns:
            MarkdownSkill instance with Stage 1 (metadata) loaded.

        Raises:
            FileNotFoundError: If SKILL.md doesn't exist.
            ValueError: If frontmatter is invalid.
        """
        skill_path = Path(skill_dir)
        skill_md_path = skill_path / "SKILL.md"

        if not skill_md_path.exists():
            raise FileNotFoundError(
                f"SKILL.md not found in {skill_dir}"
            )

        # Parse only frontmatter for Stage 1
        content = skill_md_path.read_text(encoding="utf-8")
        metadata = cls._parse_frontmatter(content)

        # Validate name matches directory
        if metadata.name != skill_path.name:
            raise ValueError(
                f"Skill name '{metadata.name}' must match "
                f"directory name '{skill_path.name}'"
            )

        return cls(
            name=metadata.name,
            description=metadata.description,
            skill_path=skill_path,
            skill_metadata=metadata,
            config=cls._build_config(metadata),
        )

    @staticmethod
    def _parse_frontmatter(content: str) -> MarkdownSkillMetadata:
        """Parse YAML frontmatter from SKILL.md content."""
        import yaml

        if not content.startswith("---"):
            raise ValueError("SKILL.md must start with YAML frontmatter")

        # Split frontmatter from body
        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError("Invalid frontmatter format")

        frontmatter_yaml = parts[1].strip()
        frontmatter = yaml.safe_load(frontmatter_yaml)

        return MarkdownSkillMetadata(**frontmatter)

    @staticmethod
    def _build_config(metadata: MarkdownSkillMetadata) -> SkillConfig:
        """Build SkillConfig from metadata."""
        config = SkillConfig()

        # Parse compatibility for network requirements
        if metadata.compatibility:
            if "network" in metadata.compatibility.lower():
                config.allow_network = True

        return config

    # =========================================================================
    # Progressive Disclosure Implementation
    # =========================================================================

    def get_instructions(self) -> str:
        """Get full SKILL.md instructions (Stage 2).

        Loads and caches the markdown body on first access.
        """
        if self._instructions_cache is None:
            skill_md_path = self.skill_path / "SKILL.md"
            content = skill_md_path.read_text(encoding="utf-8")

            # Extract body after frontmatter
            parts = content.split("---", 2)
            self._instructions_cache = parts[2].strip() if len(parts) > 2 else ""
            self._current_stage = max(self._current_stage, 2)

        return self._instructions_cache

    def get_script(self, script_name: str) -> Optional[str]:
        """Get script content from scripts/ directory (Stage 3).

        Args:
            script_name: Name of the script file.

        Returns:
            Script content or None if not found.
        """
        if script_name not in self._scripts_cache:
            script_path = self.skill_path / "scripts" / script_name
            if script_path.exists():
                self._scripts_cache[script_name] = script_path.read_text(
                    encoding="utf-8"
                )
                self._current_stage = 3
            else:
                return None

        return self._scripts_cache.get(script_name)

    def get_reference(self, ref_name: str) -> Optional[str]:
        """Get reference content from references/ directory (Stage 3).

        Args:
            ref_name: Name of the reference file.

        Returns:
            Reference content or None if not found.
        """
        if ref_name not in self._references_cache:
            ref_path = self.skill_path / "references" / ref_name
            if ref_path.exists():
                self._references_cache[ref_name] = ref_path.read_text(
                    encoding="utf-8"
                )
                self._current_stage = 3
            else:
                return None

        return self._references_cache.get(ref_name)

    def get_asset_path(self, asset_name: str) -> Optional[Path]:
        """Get absolute path to an asset file (Stage 3).

        Args:
            asset_name: Relative path within assets/ directory.

        Returns:
            Absolute Path or None if not found.
        """
        asset_path = self.skill_path / "assets" / asset_name
        if asset_path.exists():
            self._current_stage = 3
            return asset_path
        return None

    def list_scripts(self) -> List[str]:
        """List available scripts in the skill."""
        scripts_dir = self.skill_path / "scripts"
        if scripts_dir.exists():
            return [f.name for f in scripts_dir.iterdir() if f.is_file()]
        return []

    def list_references(self) -> List[str]:
        """List available references in the skill."""
        refs_dir = self.skill_path / "references"
        if refs_dir.exists():
            return [f.name for f in refs_dir.iterdir() if f.is_file()]
        return []

    def list_assets(self) -> List[str]:
        """List available assets in the skill."""
        assets_dir = self.skill_path / "assets"
        if assets_dir.exists():
            return [
                str(f.relative_to(assets_dir))
                for f in assets_dir.rglob("*")
                if f.is_file()
            ]
        return []

    # =========================================================================
    # BaseSkill Abstract Method Implementations
    # =========================================================================

    def get_tool_declarations(self) -> List[dict[str, Any]]:
        """Return tool declarations extracted from SKILL.md.

        Parses the instructions to find tool references and
        generates declarations for script-based tools.
        """
        declarations = []

        # Add script-based tools
        for script_name in self.list_scripts():
            script_path = self.skill_path / "scripts" / script_name
            docstring = self._extract_script_docstring(script_path)

            declarations.append({
                "name": f"run_{script_name.replace('.', '_')}",
                "description": docstring or f"Execute {script_name}",
                "parameters": {
                    "args": "Command-line arguments for the script"
                }
            })

        # Add reference loading tools
        for ref_name in self.list_references():
            declarations.append({
                "name": f"load_reference_{ref_name.replace('.', '_')}",
                "description": f"Load reference document: {ref_name}",
            })

        return declarations

    def get_orchestration_template(self) -> str:
        """Return example orchestration code for this skill.

        Generates a template based on available scripts and tools.
        """
        scripts = self.list_scripts()

        if not scripts:
            return f'''
async def use_{self.name.replace("-", "_")}(tools):
    """Example orchestration for {self.name} skill."""
    # This skill provides instructions but no bundled scripts.
    # Follow the instructions in the SKILL.md file.
    return {{"status": "ready", "skill": "{self.name}"}}
'''

        script_calls = "\n    ".join(
            f'result_{i} = await tools.run_{s.replace(".", "_")}(args="")'
            for i, s in enumerate(scripts[:3])
        )

        return f'''
async def use_{self.name.replace("-", "_")}(tools):
    """Example orchestration for {self.name} skill."""
    {script_calls}
    return {{"results": [result_0]}}
'''

    def get_skill_prompt(self) -> str:
        """Generate LLM-friendly skill description with progressive detail."""
        base_prompt = super().get_skill_prompt()

        # Add available resources
        scripts = self.list_scripts()
        refs = self.list_references()

        resources = []
        if scripts:
            resources.append(f"Scripts: {', '.join(scripts)}")
        if refs:
            resources.append(f"References: {', '.join(refs)}")

        if resources:
            base_prompt += f"\n\nAvailable resources:\n" + "\n".join(
                f"  - {r}" for r in resources
            )

        return base_prompt

    @staticmethod
    def _extract_script_docstring(script_path: Path) -> Optional[str]:
        """Extract docstring from a Python script."""
        if not script_path.suffix == ".py":
            return None

        try:
            content = script_path.read_text(encoding="utf-8")
            # Simple regex to extract module docstring
            match = re.match(r'^"""(.+?)"""', content, re.DOTALL)
            if match:
                return match.group(1).strip().split("\n")[0]
        except Exception:
            pass

        return None
```

#### 2. AgentSkillLoader Class

Discovers and loads skills from directories:

```python
# src/google/adk/skills/agent_skill_loader.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base_skill import BaseSkill
from .markdown_skill import MarkdownSkill
from .skill_manager import SkillsManager

logger = logging.getLogger("google_adk.skills")


class AgentSkillLoader:
    """Discovers and loads Agent Skills standard skills.

    Implements progressive disclosure by loading only metadata initially,
    with full content loaded on-demand when skills are activated.

    Example:
        ```python
        loader = AgentSkillLoader()

        # Discover skills from multiple directories
        loader.add_skill_directory("/path/to/skills")
        loader.add_skill_directory("/path/to/custom-skills")

        # Get all discovered skills (Stage 1 - metadata only)
        skills = loader.get_all_skills()

        # Register with SkillsManager
        manager = SkillsManager()
        loader.register_all(manager)

        # Generate discovery prompt for LLM
        prompt = loader.generate_discovery_prompt()
        ```
    """

    def __init__(self):
        self._skill_directories: List[Path] = []
        self._discovered_skills: Dict[str, MarkdownSkill] = {}
        self._load_errors: Dict[str, str] = {}

    def add_skill_directory(self, path: Union[str, Path]) -> int:
        """Add a directory to scan for skills.

        Args:
            path: Directory containing skill folders.

        Returns:
            Number of skills discovered in this directory.

        Raises:
            FileNotFoundError: If directory doesn't exist.
        """
        dir_path = Path(path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Skill directory not found: {path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        self._skill_directories.append(dir_path)
        return self._discover_skills_in_directory(dir_path)

    def _discover_skills_in_directory(self, dir_path: Path) -> int:
        """Discover all skills in a directory."""
        count = 0

        for item in dir_path.iterdir():
            if not item.is_dir():
                continue

            skill_md = item / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                skill = MarkdownSkill.from_directory(item)
                self._discovered_skills[skill.name] = skill
                count += 1
                logger.info(f"Discovered skill: {skill.name}")
            except Exception as e:
                self._load_errors[str(item)] = str(e)
                logger.warning(f"Failed to load skill from {item}: {e}")

        return count

    def get_skill(self, name: str) -> Optional[MarkdownSkill]:
        """Get a discovered skill by name."""
        return self._discovered_skills.get(name)

    def get_all_skills(self) -> List[MarkdownSkill]:
        """Get all discovered skills."""
        return list(self._discovered_skills.values())

    def get_skill_names(self) -> List[str]:
        """Get names of all discovered skills."""
        return list(self._discovered_skills.keys())

    def get_load_errors(self) -> Dict[str, str]:
        """Get any errors encountered during discovery."""
        return self._load_errors.copy()

    def register_all(self, manager: SkillsManager) -> int:
        """Register all discovered skills with a SkillsManager.

        Args:
            manager: The SkillsManager to register skills with.

        Returns:
            Number of skills registered.
        """
        count = 0
        for skill in self._discovered_skills.values():
            try:
                manager.register_skill(skill)
                count += 1
            except ValueError as e:
                logger.warning(f"Failed to register skill {skill.name}: {e}")

        return count

    def generate_discovery_prompt(self) -> str:
        """Generate XML prompt with skill metadata for LLM discovery.

        This implements Stage 1 of progressive disclosure - only
        name and description are included, keeping context minimal.

        Returns:
            XML-formatted string with available skills.
        """
        if not self._discovered_skills:
            return "<available_skills></available_skills>"

        skills_xml = []
        for skill in self._discovered_skills.values():
            skill_xml = f"""  <skill>
    <name>{skill.name}</name>
    <description>{skill.description}</description>
    <has_scripts>{len(skill.list_scripts()) > 0}</has_scripts>
    <has_references>{len(skill.list_references()) > 0}</has_references>
  </skill>"""
            skills_xml.append(skill_xml)

        return f"""<available_skills>
{chr(10).join(skills_xml)}
</available_skills>"""

    def generate_activation_prompt(self, skill_name: str) -> Optional[str]:
        """Generate full skill prompt for activation (Stage 2).

        Args:
            skill_name: Name of the skill to activate.

        Returns:
            Full skill instructions or None if skill not found.
        """
        skill = self._discovered_skills.get(skill_name)
        if not skill:
            return None

        instructions = skill.get_instructions()
        resources = []

        scripts = skill.list_scripts()
        if scripts:
            resources.append(f"Available scripts: {', '.join(scripts)}")

        refs = skill.list_references()
        if refs:
            resources.append(f"Available references: {', '.join(refs)}")

        resource_section = ""
        if resources:
            resource_section = "\n\n## Available Resources\n" + "\n".join(
                f"- {r}" for r in resources
            )

        return f"""# Skill: {skill.name}

{instructions}
{resource_section}
"""
```

#### 3. ScriptExecutor Class

Safely executes bundled scripts:

```python
# src/google/adk/skills/script_executor.py

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.feature_decorator import experimental


class ScriptExecutionResult(BaseModel):
    """Result from script execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time_ms: float = 0.0


@experimental
class ScriptExecutor(BaseModel):
    """Executes scripts from Agent Skills bundles.

    Provides sandboxed execution of Python, Bash, and JavaScript
    scripts bundled with skills.

    Security features:
    - Execution timeout
    - Working directory isolation
    - Environment variable filtering
    - Optional container sandboxing

    Example:
        ```python
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
    """

    timeout_seconds: float = Field(
        default=60.0,
        description="Maximum execution time in seconds.",
    )
    allow_network: bool = Field(
        default=False,
        description="Whether to allow network access.",
    )
    memory_limit_mb: int = Field(
        default=256,
        description="Memory limit in megabytes.",
    )
    use_container: bool = Field(
        default=False,
        description="Use container isolation (requires Docker).",
    )
    allowed_env_vars: List[str] = Field(
        default_factory=lambda: ["PATH", "HOME", "LANG", "LC_ALL"],
        description="Environment variables to pass through.",
    )

    model_config = ConfigDict(extra="forbid")

    async def execute_script(
        self,
        script_path: Path,
        args: List[str] = None,
        working_dir: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ScriptExecutionResult:
        """Execute a script file.

        Args:
            script_path: Path to the script file.
            args: Command-line arguments.
            working_dir: Working directory for execution.
            env: Additional environment variables.

        Returns:
            ScriptExecutionResult with stdout, stderr, and status.
        """
        import time

        args = args or []
        start_time = time.time()

        # Determine interpreter based on file extension
        interpreter = self._get_interpreter(script_path)

        # Build command
        cmd = [interpreter, str(script_path)] + args

        # Build safe environment
        safe_env = self._build_safe_env(env)

        # Set working directory
        cwd = working_dir or script_path.parent

        try:
            if self.use_container:
                result = await self._execute_in_container(
                    cmd, cwd, safe_env
                )
            else:
                result = await self._execute_subprocess(
                    cmd, cwd, safe_env
                )

            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            return result

        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            return ScriptExecutionResult(
                success=False,
                stderr=f"Execution timed out after {self.timeout_seconds}s",
                return_code=-1,
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ScriptExecutionResult(
                success=False,
                stderr=str(e),
                return_code=-1,
                execution_time_ms=execution_time,
            )

    def _get_interpreter(self, script_path: Path) -> str:
        """Determine the interpreter for a script."""
        suffix = script_path.suffix.lower()

        interpreters = {
            ".py": "python3",
            ".sh": "bash",
            ".bash": "bash",
            ".js": "node",
            ".mjs": "node",
        }

        if suffix not in interpreters:
            raise ValueError(f"Unsupported script type: {suffix}")

        return interpreters[suffix]

    def _build_safe_env(
        self, additional_env: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Build a safe environment for script execution."""
        safe_env = {}

        # Only pass allowed environment variables
        for var in self.allowed_env_vars:
            if var in os.environ:
                safe_env[var] = os.environ[var]

        # Add additional environment variables
        if additional_env:
            safe_env.update(additional_env)

        return safe_env

    async def _execute_subprocess(
        self,
        cmd: List[str],
        cwd: Path,
        env: Dict[str, str],
    ) -> ScriptExecutionResult:
        """Execute script using subprocess."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds,
            )

            return ScriptExecutionResult(
                success=proc.returncode == 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                return_code=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise

    async def _execute_in_container(
        self,
        cmd: List[str],
        cwd: Path,
        env: Dict[str, str],
    ) -> ScriptExecutionResult:
        """Execute script in a Docker container."""
        # Build Docker command
        docker_cmd = [
            "docker", "run", "--rm",
            f"--memory={self.memory_limit_mb}m",
            f"--cpus=1",
            f"-v", f"{cwd}:/workspace:ro",
            "-w", "/workspace",
        ]

        # Add network isolation if required
        if not self.allow_network:
            docker_cmd.extend(["--network", "none"])

        # Add environment variables
        for key, value in env.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        # Use appropriate base image
        docker_cmd.extend(["python:3.11-slim"])
        docker_cmd.extend(cmd)

        return await self._execute_subprocess(
            docker_cmd, cwd, os.environ.copy()
        )
```

#### 4. SkillTool Wrapper

Exposes skills as ADK tools for LLM invocation:

```python
# src/google/adk/skills/skill_tool.py

from __future__ import annotations

from typing import Any, Optional

from google.genai import types

from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from .base_skill import BaseSkill
from .markdown_skill import MarkdownSkill
from .script_executor import ScriptExecutor


class SkillTool(BaseTool):
    """Wraps a Skill as a BaseTool for LLM invocation.

    Provides three action types:
    - "activate": Load full skill instructions (Stage 2)
    - "run_script": Execute a bundled script (Stage 3)
    - "load_reference": Load a reference document (Stage 3)

    Example:
        ```python
        skill = MarkdownSkill.from_directory("/path/to/pdf-processing")
        tool = SkillTool(skill)

        # LLM can invoke:
        # - {"action": "activate"} → Returns full instructions
        # - {"action": "run_script", "script": "extract.py", "args": [...]}
        # - {"action": "load_reference", "reference": "FORMS.md"}
        ```
    """

    def __init__(
        self,
        skill: BaseSkill,
        script_executor: Optional[ScriptExecutor] = None,
    ):
        super().__init__(
            name=f"skill_{skill.name.replace('-', '_')}",
            description=self._build_description(skill),
        )
        self._skill = skill
        self._script_executor = script_executor or ScriptExecutor()

    def _build_description(self, skill: BaseSkill) -> str:
        """Build tool description from skill metadata."""
        desc = f"{skill.description}\n\n"
        desc += "Actions:\n"
        desc += "- activate: Load full skill instructions\n"

        if isinstance(skill, MarkdownSkill):
            scripts = skill.list_scripts()
            if scripts:
                desc += f"- run_script: Execute scripts ({', '.join(scripts)})\n"

            refs = skill.list_references()
            if refs:
                desc += f"- load_reference: Load references ({', '.join(refs)})\n"

        return desc

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        """Get function declaration for LLM."""
        properties = {
            "action": types.Schema(
                type="STRING",
                description="Action: 'activate', 'run_script', or 'load_reference'",
                enum=["activate", "run_script", "load_reference"],
            ),
            "script": types.Schema(
                type="STRING",
                description="Script name (for run_script action)",
            ),
            "args": types.Schema(
                type="ARRAY",
                items=types.Schema(type="STRING"),
                description="Arguments for script execution",
            ),
            "reference": types.Schema(
                type="STRING",
                description="Reference file name (for load_reference action)",
            ),
        }

        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type="OBJECT",
                properties=properties,
                required=["action"],
            ),
        )

    async def run_async(
        self,
        *,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Any:
        """Execute the skill action."""
        action = args.get("action", "activate")

        if action == "activate":
            return self._handle_activate()

        elif action == "run_script":
            return await self._handle_run_script(args, tool_context)

        elif action == "load_reference":
            return self._handle_load_reference(args)

        else:
            return {"error": f"Unknown action: {action}"}

    def _handle_activate(self) -> dict:
        """Handle skill activation (Stage 2)."""
        if isinstance(self._skill, MarkdownSkill):
            instructions = self._skill.get_instructions()
            return {
                "status": "activated",
                "skill": self._skill.name,
                "instructions": instructions,
                "available_scripts": self._skill.list_scripts(),
                "available_references": self._skill.list_references(),
            }
        else:
            return {
                "status": "activated",
                "skill": self._skill.name,
                "prompt": self._skill.get_skill_prompt(),
            }

    async def _handle_run_script(
        self, args: dict, tool_context: ToolContext
    ) -> dict:
        """Handle script execution (Stage 3)."""
        if not isinstance(self._skill, MarkdownSkill):
            return {"error": "Skill does not support scripts"}

        script_name = args.get("script")
        if not script_name:
            return {"error": "Script name required"}

        script_args = args.get("args", [])

        # Get script path
        script_path = self._skill.skill_path / "scripts" / script_name
        if not script_path.exists():
            available = self._skill.list_scripts()
            return {
                "error": f"Script not found: {script_name}",
                "available_scripts": available,
            }

        # Execute script
        result = await self._script_executor.execute_script(
            script_path=script_path,
            args=script_args,
            working_dir=tool_context.get_working_directory(),
        )

        return {
            "script": script_name,
            "success": result.success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.return_code,
            "execution_time_ms": result.execution_time_ms,
        }

    def _handle_load_reference(self, args: dict) -> dict:
        """Handle reference loading (Stage 3)."""
        if not isinstance(self._skill, MarkdownSkill):
            return {"error": "Skill does not support references"}

        ref_name = args.get("reference")
        if not ref_name:
            return {"error": "Reference name required"}

        content = self._skill.get_reference(ref_name)
        if content is None:
            available = self._skill.list_references()
            return {
                "error": f"Reference not found: {ref_name}",
                "available_references": available,
            }

        return {
            "reference": ref_name,
            "content": content,
        }
```

### Integration with LlmAgent

Skills integrate with ADK agents through the `skills` field:

```python
from google.adk.agents import LlmAgent
from google.adk.skills import SkillsManager, AgentSkillLoader, SkillTool

# Load Agent Skills standard skills
loader = AgentSkillLoader()
loader.add_skill_directory("./skills")

# Create skills manager
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

### Extended SKILL.md Frontmatter for ADK

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

  # PTC enablement
  allowed_callers:
    - code_execution_20250825

  # Tool declarations for scripts
  tools:
    - name: extract_text
      script: scripts/extract_text.py
      description: Extract text from PDF pages
      parameters:
        input_file: Path to PDF file
        pages: Optional page range (e.g., "1-5")

    - name: fill_form
      script: scripts/fill_form.py
      description: Fill PDF form fields
      parameters:
        input_file: Path to PDF form
        field_values: JSON object with field names and values

  # Result filtering rules
  filter_rules:
    - field: raw_text
      action: truncate
      max_length: 10000
    - field: metadata.password
      action: remove
---
```

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

### Module Exports

```python
# src/google/adk/skills/__init__.py

from .base_skill import BaseSkill, SkillConfig
from .skill_manager import SkillInvocationResult, SkillsManager
from .markdown_skill import MarkdownSkill, MarkdownSkillMetadata
from .agent_skill_loader import AgentSkillLoader
from .script_executor import ScriptExecutor, ScriptExecutionResult
from .skill_tool import SkillTool

__all__ = [
    # Core abstractions
    "BaseSkill",
    "SkillConfig",
    "SkillsManager",
    "SkillInvocationResult",
    # Agent Skills standard support
    "MarkdownSkill",
    "MarkdownSkillMetadata",
    "AgentSkillLoader",
    # Execution
    "ScriptExecutor",
    "ScriptExecutionResult",
    # Tool integration
    "SkillTool",
]
```

### Comparison: ADK Skills vs Agent Skills Standard

| Feature | Agent Skills Standard | ADK Skills (Current) | ADK Skills (Enhanced) |
|---------|----------------------|---------------------|----------------------|
| **Format** | SKILL.md files | Python classes | Both supported |
| **Discovery** | YAML frontmatter | Class attributes | Unified loader |
| **Progressive Disclosure** | 3 stages | Partial | Full 3-stage support |
| **Scripts** | scripts/ directory | Not supported | Full support |
| **References** | references/ directory | Not supported | Full support |
| **Assets** | assets/ directory | Not supported | Full support |
| **PTC Support** | Not specified | Yes | Yes |
| **Result Filtering** | Not specified | Yes | Yes + configurable |
| **Security** | Recommendations | 4-layer defense | 4-layer defense |
| **Tool Declarations** | Not specified | Required method | Auto-generated |
| **Portability** | Cross-platform | ADK only | Cross-platform compatible |

### Usage Example: Using Anthropic Skills in ADK

```python
# Clone the Anthropic skills repository
# git clone https://github.com/anthropics/skills ./anthropic-skills

from google.adk.agents import LlmAgent
from google.adk.skills import AgentSkillLoader, SkillTool, SkillsManager

# Load skills from Anthropic's repository
loader = AgentSkillLoader()
loader.add_skill_directory("./anthropic-skills/skills")

# Check what was loaded
print(f"Discovered {len(loader.get_skill_names())} skills:")
for name in loader.get_skill_names():
    skill = loader.get_skill(name)
    print(f"  - {name}: {skill.description[:50]}...")

# Create agent with these skills
skills_manager = SkillsManager()
loader.register_all(skills_manager)

skill_tools = [SkillTool(s) for s in skills_manager.get_all_skills()]

agent = LlmAgent(
    name="anthropic_skills_agent",
    model="gemini-2.0-flash",
    instruction=f"""You have access to skills from the Agent Skills standard.

{loader.generate_discovery_prompt()}

Use the activate action first to learn how to use each skill.
""",
    tools=skill_tools,
)
```

### File Structure for Enhanced Skills Module

```
src/google/adk/skills/
├── __init__.py                  # Module exports
├── base_skill.py                # BaseSkill abstract class (existing)
├── skill_manager.py             # SkillsManager (existing)
├── markdown_skill.py            # NEW: MarkdownSkill for SKILL.md
├── agent_skill_loader.py        # NEW: Discovery and loading
├── script_executor.py           # NEW: Script execution
├── skill_tool.py                # NEW: SkillTool wrapper
└── builtin/                     # Built-in skills
    ├── __init__.py
    └── ...
```

---

## Security Considerations

Script execution introduces security risks. Implement appropriate safeguards:

| Measure | Description |
|---------|-------------|
| **Sandboxing** | Run scripts in isolated environments |
| **Allowlisting** | Only execute scripts from trusted skills |
| **Confirmation** | Ask users before running potentially dangerous operations |
| **Logging** | Record all script executions for auditing |
| **Source Verification** | Install skills only from trusted sources |
| **Audit** | Review bundled files, dependencies, and external network connections before deployment |

---

## Reference Library (skills-ref)

The `skills-ref` library provides Python utilities and a CLI for working with Agent Skills.

### Installation

```bash
# Using pip
pip install skills-ref

# Using uv
uv sync
```

### CLI Commands

```bash
# Validate a skill directory
skills-ref validate <path>

# Extract skill metadata as JSON
skills-ref read-properties <path>

# Generate <available_skills> XML for agent prompts
skills-ref to-prompt <path>...
```

### Python API

```python
from skills_ref import validate, read_properties, to_prompt

# Validate skill directories and get error reports
errors = validate("/path/to/skill")

# Read skill configuration and metadata
metadata = read_properties("/path/to/skill")

# Create XML formatted skill descriptions for system prompts
xml_prompt = to_prompt(["/path/to/skill1", "/path/to/skill2"])
```

**Note**: This library is intended for demonstration purposes. It is not meant to be used in production.

**Repository**: https://github.com/agentskills/agentskills/tree/main/skills-ref

---

## Best Practices

### Development Workflow

1. **Start with Evaluation**: Identify capability gaps through representative task testing
2. **Structure for Scale**: Split unwieldy documentation into separate, logically organized files
3. **Think from the Agent's Perspective**: Monitor real usage patterns and iterate based on skill triggering behavior
4. **Iterate with the Agent**: Collaborate with the agent to capture successful approaches into reusable skill components

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

---

## Example Skills

### Minimal Skill

```
hello-world/
└── SKILL.md
```

```markdown
---
name: hello-world
description: Demonstrates basic skill structure. Use when learning about skills.
---

# Hello World Skill

## Instructions
1. Respond with "Hello, World!" when activated
2. Explain that this is a demonstration skill

## Example
User: "Can you demonstrate a skill?"
Agent: "Hello, World! This is a demonstration of the Agent Skills format."
```

### Document Processing Skill

```
pdf-processing/
├── SKILL.md
├── scripts/
│   ├── extract_text.py
│   └── merge_pdfs.py
├── references/
│   └── FORMS.md
└── assets/
    └── templates/
        └── invoice_template.pdf
```

```markdown
---
name: pdf-processing
description: Extract text and tables from PDF files, fill PDF forms, and merge multiple PDFs. Use when working with PDF documents.
license: Apache-2.0
compatibility: Requires Python 3.8+, pdfplumber, PyPDF2
metadata:
  author: example-org
  version: "1.0"
---

# PDF Processing

## When to use this skill
Use this skill when the user needs to:
- Extract text from PDF documents
- Extract tables from PDFs
- Fill in PDF forms
- Merge multiple PDFs into one

## Prerequisites
- Python 3.8 or higher
- pdfplumber package
- PyPDF2 package

## Text Extraction

### Using pdfplumber
```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        print(text)
```

### Using the bundled script
Run `scripts/extract_text.py <input.pdf> <output.txt>`

## Table Extraction
...

## Form Filling
See [FORMS.md](references/FORMS.md) for detailed form handling instructions.

## Merging PDFs
Run `scripts/merge_pdfs.py <output.pdf> <input1.pdf> <input2.pdf> ...`
```

### Production Skills

The following production-grade skills are available as reference implementations:

| Skill | Description | Repository |
|-------|-------------|------------|
| `docx` | Word document creation/editing | anthropics/skills |
| `pdf` | PDF manipulation | anthropics/skills |
| `pptx` | PowerPoint creation/editing | anthropics/skills |
| `xlsx` | Excel spreadsheet operations | anthropics/skills |

---

## Resources

### Official Documentation

- **Agent Skills Website**: https://agentskills.io
- **Specification**: https://agentskills.io/specification
- **What Are Skills?**: https://agentskills.io/what-are-skills
- **Integration Guide**: https://agentskills.io/integrate-skills

### GitHub Repositories

- **Agent Skills Framework**: https://github.com/agentskills/agentskills
- **Example Skills**: https://github.com/anthropics/skills
- **Reference Library**: https://github.com/agentskills/agentskills/tree/main/skills-ref

### LangChain/LangGraph Resources

- **Multi-Agent Patterns**: https://docs.langchain.com/oss/python/langchain/multi-agent
- **Skills Pattern**: https://docs.langchain.com/oss/python/langchain/multi-agent/skills
- **LangGraph Workflows**: https://docs.langchain.com/oss/python/langgraph/workflows-agents
- **LangGraph Official Site**: https://www.langchain.com/langgraph
- **Multi-Agent Workflows Blog**: https://www.blog.langchain.com/langgraph-multi-agent-workflows/

### Google ADK Resources

- **ADK Repository**: https://github.com/google/adk-python
- **ADK Skills Module**: `src/google/adk/skills/`
- **PTC Design Document**: `docs/skills_programmatic_tool_calling_design.md`
- **Key Files**:
  - `base_skill.py` - BaseSkill abstract class
  - `skill_manager.py` - SkillsManager registry
  - `markdown_skill.py` - SKILL.md file loader (proposed)
  - `agent_skill_loader.py` - Discovery and loading (proposed)
  - `script_executor.py` - Script execution (proposed)
  - `skill_tool.py` - Tool wrapper (proposed)

### Anthropic Resources

- **Engineering Blog**: https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- **Claude Support - What are Skills?**: https://support.claude.com/en/articles/12512176-what-are-skills
- **Claude Support - Creating Custom Skills**: https://support.claude.com/en/articles/12512198-creating-custom-skills
- **Claude Support - Using Skills**: https://support.claude.com/en/articles/12512180-using-skills-in-claude

### Platform Support

Skills are supported across:
- Claude.ai
- Claude Code
- Claude Agent SDK
- Claude Developer Platform
- LangChain/LangGraph (via skills pattern integration)
- Google ADK (via MarkdownSkill and AgentSkillLoader)

---

## Appendix: Quick Reference

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

### Validation Checklist

- [ ] Directory name matches `name` field in frontmatter
- [ ] Name follows conventions (lowercase, hyphens only, 64 chars max)
- [ ] Description is clear and includes usage context (1024 chars max)
- [ ] SKILL.md is under 500 lines
- [ ] Scripts are self-contained with documented dependencies
- [ ] All file references use relative paths
- [ ] Security review completed for any executable code
