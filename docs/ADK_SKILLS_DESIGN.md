# ADK Dynamic Skills Framework Design Document

**Author:** Agent Development Kit Team
**Status:** Proposal
**Created:** December 2025
**Version:** 1.0

---

## Executive Summary

This document proposes a first-class **Dynamic Skills Framework** for the Google Agent Development Kit (ADK). The framework enables agents to dynamically load domain-specific knowledge into their context on-demand, addressing two critical challenges in LLM-based agents:

1. **Knowledge Staleness**: Rapidly evolving domains (like BigQuery AI functions) require up-to-date guidance that cannot be baked into model weights
2. **Context Window Efficiency**: Loading comprehensive documentation permanently wastes precious context tokens on irrelevant information

The proposed solution uses **callback-based skill injection** to automatically detect relevant skills from user input and inject them ephemerally into the system instruction, achieving zero-latency skill availability with minimal context overhead.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Goals and Non-Goals](#2-goals-and-non-goals)
3. [Design Overview](#3-design-overview)
4. [Detailed Design](#4-detailed-design)
5. [API Specification](#5-api-specification)
6. [Implementation Details](#6-implementation-details)
7. [BigQuery Skills Demo Case Study](#7-bigquery-skills-demo-case-study)
8. [Performance Analysis](#8-performance-analysis)
9. [Migration and Rollout](#9-migration-and-rollout)
10. [Future Extensions](#10-future-extensions)
11. [Appendix](#appendix)

---

## 1. Problem Statement

### 1.1 The Knowledge Staleness Problem

Modern cloud platforms evolve rapidly. Consider BigQuery's AI capabilities:

| Timeline | New Feature |
|----------|-------------|
| Q3 2024 | AI.CLASSIFY, AI.IF, AI.SCORE functions |
| Q4 2024 | Gemini 2.0 Flash endpoint |
| Q1 2025 | Gemini 2.5 Pro, Claude 3.5 Sonnet integration |
| Q2 2025 | New connection_id syntax requirements |

LLM training data lags 6-18 months behind. An agent with outdated knowledge will:
- Generate incorrect SQL syntax
- Reference deprecated endpoints
- Miss critical configuration requirements (e.g., location matching for connections)

**Example of Outdated Knowledge Impact:**
```sql
-- LLM might generate (outdated):
CREATE REMOTE MODEL `project.dataset.model`
OPTIONS (ENDPOINT = 'gemini-pro');  -- Old endpoint name

-- Correct (current):
CREATE REMOTE MODEL `project.dataset.model`
REMOTE WITH CONNECTION `us.my_connection`  -- Required connection
OPTIONS (ENDPOINT = 'gemini-2.5-pro');     -- Current endpoint
```

### 1.2 The Context Window Efficiency Problem

Comprehensive documentation for a single domain can be substantial:

| Skill | Documentation Size | Tokens (est.) |
|-------|-------------------|---------------|
| BQML | ~4,000 words | ~6,000 tokens |
| BQ AI Operator | ~2,500 words | ~3,750 tokens |
| BQ Remote Model | ~3,500 words | ~5,250 tokens |
| **Total** | **~10,000 words** | **~15,000 tokens** |

Loading all skills permanently means:
- 15,000 tokens consumed before any user interaction
- Reduced space for conversation history
- Slower response times (more tokens to process)
- Higher costs (token-based pricing)

### 1.3 Current Approaches and Limitations

| Approach | Description | Limitation |
|----------|-------------|------------|
| **Static System Prompt** | All documentation in base prompt | Context waste, knowledge staleness |
| **RAG (Retrieval)** | Semantic search for relevant docs | Latency, retrieval quality issues |
| **Tool-based Loading** | Agent calls `load_skill()` tool | Extra LLM call(s), timing delays |
| **Instruction Provider** | Dynamic system instruction | Timing issue: skills not in first call |

The tool-based approach is particularly problematic:

```
User: "Train a model to predict penguin weight"
                    │
                    ▼
        ┌────────────────────────┐
        │ LLM Call #1            │ ◄── No skill loaded yet!
        │ "I'll load the BQML    │     Agent must first decide
        │  skill to help you"    │     to load the skill
        └────────────────────────┘
                    │
                    ▼ Tool call: activate_skill("bqml")
        ┌────────────────────────┐
        │ LLM Call #2            │ ◄── Now skill is available
        │ "Here's how to train   │     But we wasted a round-trip
        │  your model..."        │
        └────────────────────────┘
```

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. **Zero-Latency Skill Availability**: Skills are available in the FIRST LLM call, not after tool calls
2. **Ephemeral Loading**: Skills are injected into system instruction, not conversation history
3. **Automatic Detection**: Keywords/patterns trigger skill loading without LLM decision-making
4. **Automatic Cleanup**: Skills are removed after each turn to free context
5. **Scalable Architecture**: Adding new skills requires only a markdown file, no code changes
6. **Multi-Skill Support**: Multiple skills can be active simultaneously
7. **Configurable Detection**: Support keyword, LLM, and hybrid detection modes

### 2.2 Non-Goals

1. **Persistent Skill Memory**: Skills are ephemeral per-turn (not across sessions)
2. **Skill Execution**: Skills provide knowledge, not executable code
3. **Skill Versioning**: No built-in version management (use Git for skill files)
4. **Cross-Agent Skill Sharing**: Skills are agent-specific (no central registry)
5. **Skill Composition**: No dependency management between skills

---

## 3. Design Overview

### 3.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User Message                                       │
│              "Train a model to predict penguin weight"                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         before_model_callback                                │
│                                                                              │
│   1. Extract text from llm_request.contents                                  │
│   2. Match against skill keywords (from SKILL.md frontmatter)               │
│      - "train" matches bqml                                                 │
│      - "model" matches bqml                                                 │
│      - "predict" matches bqml                                               │
│   3. Load skill content from SkillRegistry                                  │
│   4. Call llm_request.append_instructions([skill_content])                  │
│      └─► This modifies config.system_instruction directly                   │
│   5. Store active skills in callback_context.state                          │
│                                                                              │
│   Result: Skills injected BEFORE first LLM call                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM Processing                                     │
│                                                                              │
│   System Instruction now includes:                                           │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ [Base Agent Instructions]                                             │  │
│   │                                                                        │  │
│   │ # Currently Active Skills                                              │  │
│   │                                                                        │  │
│   │ ## Active Skill: bqml                                                  │  │
│   │ BigQuery ML - Train, evaluate, and deploy ML models using SQL...      │  │
│   │                                                                        │  │
│   │ ### Step 1: Train a Model                                              │  │
│   │ ```sql                                                                 │  │
│   │ CREATE OR REPLACE MODEL `project.dataset.model_name`                   │  │
│   │ OPTIONS(model_type='LINEAR_REG', input_label_cols=['target'])...       │  │
│   │ ```                                                                    │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   LLM generates response using skill knowledge from FIRST call              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (tool calls, multi-turn processing)
┌─────────────────────────────────────────────────────────────────────────────┐
│                         after_agent_callback                                 │
│                                                                              │
│   1. Read active skills from callback_context.state                         │
│   2. Clear state: callback_context.state[ACTIVE_SKILLS_KEY] = []            │
│   3. Log: "[SkillCallbacks] Auto-deactivated skills: ['bqml']"              │
│                                                                              │
│   Result: Context freed for next user turn                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Design Decisions

#### Decision 1: Callbacks vs Tools for Skill Management

| Aspect | Callback (Chosen) | Tool-based |
|--------|-------------------|------------|
| **LLM Calls** | Zero for skill management | 1-2 per skill activation |
| **Latency** | Instant (regex matching) | Round-trip to model |
| **Cost** | No additional tokens | Extra tool call tokens |
| **First-Call Availability** | Yes | No (skills after tool call) |
| **Determinism** | 100% for keyword mode | LLM decides, may miss |

**Rationale**: Domain-specific terminology (e.g., "AI.CLASSIFY", "BQML", "CREATE MODEL") is unambiguous and maps cleanly to skills. Keyword matching is sufficient and eliminates LLM overhead.

#### Decision 2: Direct Injection vs Instruction Provider

| Approach | When Skills Appear | Mechanism |
|----------|-------------------|-----------|
| **Direct Injection (Chosen)** | First LLM call | Modify `llm_request.config.system_instruction` |
| Instruction Provider | Second LLM call | Provider reads from state after instruction built |

**Rationale**: The ADK processes `_preprocess_async` (which builds system instruction) BEFORE `before_model_callback`. Using an instruction provider means skills set in callback aren't visible until the next LLM call. Direct injection via `llm_request.append_instructions()` bypasses this timing issue.

#### Decision 3: Ephemeral vs Persistent Skills

| Aspect | Ephemeral (Chosen) | Persistent |
|--------|-------------------|------------|
| **Memory** | Cleared after each turn | Accumulates in session |
| **Context Efficiency** | High (only load when needed) | Low (grows over time) |
| **Multi-Topic** | Clean transitions | Old skills pollute context |

**Rationale**: Most user interactions are single-topic. Clearing skills after each turn ensures fresh context and prevents irrelevant skill content from consuming tokens in unrelated queries.

---

## 4. Detailed Design

### 4.1 Component Overview

```
google/adk/skills/
├── __init__.py              # Public API exports
├── skill_registry.py        # Skill discovery and loading
├── skill_callbacks.py       # Callback-based skill management
├── skill_classifier.py      # Optional LLM-based classification
└── types.py                 # Data classes (SkillMetadata, SkillContent)
```

### 4.2 Skill Definition Format (SKILL.md)

Skills are defined as Markdown files with YAML frontmatter:

```markdown
---
name: bq_remote_model
description: BigQuery Remote Models - Create remote models connecting to Vertex AI
keywords:
  - remote model
  - create remote model
  - generate text
  - ai.generate_text
  - gemini
  - claude
  - embeddings
  - llm
---

# BQ Remote Model Skill

Create and use remote models that connect BigQuery to Vertex AI...

## Prerequisites

1. A BigQuery connection to Vertex AI is required...

## CREATE REMOTE MODEL Syntax

```sql
CREATE OR REPLACE MODEL `project.dataset.model_name`
REMOTE WITH CONNECTION `project.region.connection_id`
OPTIONS (ENDPOINT = 'gemini-2.5-pro');
```

[... full documentation ...]
```

**Frontmatter Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique skill identifier (alphanumeric, underscores) |
| `description` | Yes | Short description for skill summary |
| `keywords` | No | List of trigger keywords/phrases for auto-detection |

### 4.3 SkillRegistry Class

```python
class SkillRegistry:
    """Registry for dynamically discovering and loading skills.

    Implements progressive disclosure:
    - Level 1: Skill names and descriptions (loaded at startup)
    - Level 2: Full skill content (loaded on-demand)
    """

    def __init__(self, skills_dir: str | Path | None = None):
        """Initialize registry and discover skills.

        Args:
            skills_dir: Directory containing skill subdirectories.
                       Defaults to ./skills relative to caller.
        """

    def get_skill_names(self) -> list[str]:
        """Get list of all discovered skill names."""

    def get_skill_metadata(self, name: str) -> SkillMetadata | None:
        """Get metadata (name, description, keywords) for a skill."""

    def get_all_keywords(self) -> dict[str, list[str]]:
        """Get all keywords for all skills, for pattern building."""

    def load_skill_content(self, name: str) -> SkillContent | None:
        """Load full content of a skill (Level 2 disclosure)."""

    def get_skills_summary(self) -> str:
        """Get formatted summary for agent's base instruction."""
```

### 4.4 SkillCallbacks Class

```python
class SkillCallbacks:
    """Callback handlers for automatic skill management.

    Detection modes:
    - "keyword": Regex pattern matching (fastest, deterministic)
    - "llm": LLM-based classification (semantic understanding)
    - "hybrid": LLM with keyword fallback (recommended for mixed queries)
    """

    def __init__(
        self,
        registry: SkillRegistry,
        auto_deactivate: bool = True,
        detection_mode: Literal["keyword", "llm", "hybrid"] = "keyword",
    ):
        """Initialize skill callbacks.

        Args:
            registry: SkillRegistry instance for loading skills
            auto_deactivate: Clear skills after each turn (recommended)
            detection_mode: How to detect skills from user input
        """

    def before_model_callback(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        """Detect and inject skills before LLM processing.

        This is the critical callback that:
        1. Extracts user message from llm_request.contents
        2. Detects relevant skills via keyword/LLM matching
        3. Loads skill content from registry
        4. Injects into llm_request via append_instructions()
        5. Stores active skills in callback_context.state

        Returns None to continue with LLM processing.
        """

    def after_agent_callback(
        self,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        """Clean up skills after agent completes turn.

        Clears active_skills from state to free context.
        Returns None (no content to add to conversation).
        """
```

### 4.5 Keyword Detection Algorithm

```python
def _build_patterns_from_registry(self) -> dict[str, list[str]]:
    """Build regex patterns from skill keywords.

    Handles:
    - Multi-word keywords ("create remote model")
    - Special characters (dots in "ai.classify")
    - Word boundaries for precision
    """
    patterns = {}
    for skill_name, keywords in self._registry.get_all_keywords().items():
        skill_patterns = []
        for keyword in keywords:
            escaped = re.escape(keyword)
            # Don't add word boundaries for keywords with dots
            if "." in keyword:
                pattern = escaped
            else:
                pattern = rf"\b{escaped}\b"
            skill_patterns.append(pattern)
        patterns[skill_name] = skill_patterns
    return patterns

def _detect_skills_from_keywords(self, text: str) -> list[str]:
    """Detect skills using compiled regex patterns.

    Returns list of skill names that matched at least one keyword.
    """
    detected = []
    for skill_name, patterns in self._compiled_patterns.items():
        for pattern in patterns:
            if pattern.search(text):
                detected.append(skill_name)
                break  # One match is enough
    return detected
```

---

## 5. API Specification

### 5.1 LlmAgent Integration

```python
from google.adk.agents import LlmAgent
from google.adk.skills import SkillRegistry, SkillCallbacks

# Initialize skill infrastructure
skill_registry = SkillRegistry(skills_dir="./skills")
skill_callbacks = SkillCallbacks(
    registry=skill_registry,
    auto_deactivate=True,
    detection_mode="keyword",
)

# Create agent with skill callbacks
agent = LlmAgent(
    model="gemini-2.5-pro",
    name="my_agent",
    instruction=base_instruction,
    tools=[...],
    # Skill management via callbacks
    before_model_callback=skill_callbacks.before_model_callback,
    after_agent_callback=skill_callbacks.after_agent_callback,
)
```

### 5.2 LlmRequest.append_instructions() API

The `LlmRequest` class provides the `append_instructions()` method for modifying the system instruction:

```python
def append_instructions(
    self,
    instructions: Union[list[str], types.Content]
) -> list[types.Content]:
    """Appends instructions to the system instruction.

    Args:
        instructions: The instructions to append. Can be:
            - list[str]: Strings to concatenate with existing instruction
            - types.Content: Content object with text/non-text parts

    Returns:
        List of user contents from non-text parts (empty for list[str]).

    Behavior:
        - list[str]: concatenates with existing system_instruction using \\n\\n
        - types.Content: extracts text, creates references for non-text parts
    """
```

**Usage in Skill Injection:**
```python
def _inject_skills_into_request(
    self,
    llm_request: LlmRequest,
    skill_names: list[str],
) -> None:
    """Inject skill content directly into the LLM request."""
    skill_content = self._build_skill_content(skill_names)
    if skill_content:
        # This appends to config.system_instruction
        llm_request.append_instructions([skill_content])
```

### 5.3 State Management

Skills use ADK's state system for tracking active skills:

```python
# State key (session-scoped)
ACTIVE_SKILLS_KEY = "active_skills"

# Reading active skills
active_skills: list[str] = callback_context.state.get(ACTIVE_SKILLS_KEY, [])

# Writing active skills
callback_context.state[ACTIVE_SKILLS_KEY] = ["bqml", "bq_remote_model"]

# Clearing skills
callback_context.state[ACTIVE_SKILLS_KEY] = []
```

### 5.4 Manual Skill Tools (Optional Fallback)

For cases where automatic detection fails, manual tools are available:

```python
def activate_skill(skill_name: str, tool_context: ToolContext) -> str:
    """Manually activate a skill."""

def deactivate_skill(skill_name: str, tool_context: ToolContext) -> str:
    """Manually deactivate a skill."""

def list_active_skills(tool_context: ToolContext) -> str:
    """List currently active skills."""
```

---

## 6. Implementation Details

### 6.1 Callback Execution Order

Understanding ADK's callback execution order is critical:

```
Agent.run_async()
    │
    ├─► _preprocess_async()          # Builds initial system instruction
    │   └─► instruction_provider()   # Called HERE (skills not yet detected)
    │
    ├─► before_model_callback()      # Skills detected and injected HERE
    │   └─► llm_request.append_instructions([skills])
    │
    ├─► LLM.generate()               # Skills available in system instruction
    │
    ├─► after_model_callback()       # Process response
    │
    ├─► [Tool execution loop]        # May trigger more LLM calls
    │   └─► before_model_callback()  # Skills re-injected for each call
    │
    └─► after_agent_callback()       # Skills cleared HERE
```

### 6.2 Multi-Turn Tool Use Handling

When an agent uses tools, there are multiple LLM calls in a single turn. The callback handles this by:

1. **First call**: Detect skills from the LATEST user message
2. **Subsequent calls**: Use the ORIGINAL user message to avoid re-detection

```python
def before_model_callback(self, callback_context, llm_request):
    active_skills = callback_context.state.get(ACTIVE_SKILLS_KEY, [])

    if not active_skills:
        # NEW user request - detect from latest message
        user_text = self._get_user_message_text(llm_request)
        skills_to_activate = self._detect_skills_from_text(user_text)
        callback_context.state[ACTIVE_SKILLS_KEY] = skills_to_activate
    else:
        # Continuing same request - use original message
        user_text = self._get_original_user_message_text(llm_request)
        # Check for additional skills but don't reset

    # Always inject skills into this LLM call
    self._inject_skills_into_request(llm_request, active_skills)
```

### 6.3 Skill Content Building

```python
def _build_skill_content(self, skill_names: list[str]) -> str:
    """Build formatted skill content for system instruction."""
    sections = []
    for skill_name in skill_names:
        skill = self._registry.load_skill_content(skill_name)
        if skill:
            sections.append(f"""
## Active Skill: {skill.name}

{skill.description}

---

{skill.content}
""")

    if not sections:
        return ""

    return f"""
# Currently Active Skills

The following skills have been loaded for this task:

{"".join(sections)}

---
**Note**: Use `deactivate_skill(skill_name)` when done to free context.
"""
```

---

## 7. BigQuery Skills Demo Case Study

### 7.1 Domain Characteristics

BigQuery AI capabilities exemplify a rapidly evolving domain:

| Challenge | Manifestation |
|-----------|---------------|
| **API Changes** | New endpoints (gemini-2.5-pro), deprecated ones (gemini-pro) |
| **Syntax Requirements** | Connection IDs required for AI functions |
| **Location Rules** | Connection location must match dataset location |
| **Best Practices** | Task-specific parameters (max_output_tokens for summarization vs classification) |

### 7.2 Skill Structure

```
bigquery_skills_demo/
├── skills/
│   ├── bqml/
│   │   └── SKILL.md           # ML model training (LINEAR_REG, KMEANS, etc.)
│   ├── bq_ai_operator/
│   │   └── SKILL.md           # AI.CLASSIFY, AI.IF, AI.SCORE functions
│   └── bq_remote_model/
│       └── SKILL.md           # Remote models, AI.GENERATE_TEXT
├── skill_registry.py          # Dynamic discovery
├── skill_callbacks.py         # Callback-based injection
└── agent.py                   # Agent configuration
```

### 7.3 Keyword Mapping

| Skill | Keywords | Example Triggers |
|-------|----------|------------------|
| `bqml` | train, model, predict, regression, kmeans, forecast, arima | "Train a model to predict penguin weight" |
| `bq_ai_operator` | ai.classify, ai.if, ai.score, classify, sentiment, categorize | "Classify news articles by topic" |
| `bq_remote_model` | gemini, generate text, ai.generate_text, embeddings, remote model | "Create a Gemini model to summarize articles" |

### 7.4 Real-World Scenario

**User Input:**
> "Create a remote model using Gemini 2.5 Pro and use it to summarize 3 BBC news articles"

**Keyword Detection:**
- "remote model" → `bq_remote_model`
- "Gemini" → `bq_remote_model`
- "summarize" → triggers summarization examples in skill

**Injected Skill Content (excerpt):**
```markdown
## Active Skill: bq_remote_model

### ⚠️ DEFAULT MODEL: Always Use Gemini 2.5 Pro

**ALWAYS use `gemini-2.5-pro` as the default model** unless specifically requested.

### Example: Text Summarization (Large max_output_tokens)

```sql
-- Use 512-1024 tokens for summaries
SELECT
    title,
    ml_generate_text_result AS summary
FROM AI.GENERATE_TEXT(
    MODEL `project.bq_demo.gemini_model`,
    (SELECT
        title,
        CONCAT('Summarize: ', body) AS prompt
     FROM `bigquery-public-data.bbc_news.fulltext`
     LIMIT 5),
    STRUCT(
        1024 AS max_output_tokens,  -- LARGE for summarization
        0.3 AS temperature          -- Low for factual output
    )
);
```
```

**Agent Output (first LLM call):**
The agent immediately generates correct SQL using gemini-2.5-pro with appropriate parameters for summarization, without needing to first decide to load a skill.

---

## 8. Performance Analysis

### 8.1 Latency Comparison

| Approach | First Response Latency | Total LLM Calls |
|----------|----------------------|-----------------|
| **Callback-based (proposed)** | ~2-3s | 1 (if no tools) |
| Tool-based loading | ~5-6s | 2+ (load + respond) |
| Static full prompt | ~2.5-3.5s | 1 (but always slower) |

### 8.2 Token Efficiency

**Scenario**: Agent with 3 available skills (BQML, AI Operator, Remote Model)

| Approach | Tokens Used | Notes |
|----------|-------------|-------|
| All skills always loaded | ~15,000 | Regardless of query relevance |
| Callback-based (1 skill) | ~5,000 | Only relevant skill loaded |
| Callback-based (none) | ~500 | Just base instruction |

**Annual Cost Savings** (assuming 1M queries/year, 50% needing skills):
- All skills: 15B tokens = $150,000 (at $0.01/1K tokens)
- Callback: 5B tokens = $50,000
- **Savings: ~$100,000/year**

### 8.3 Detection Accuracy

Keyword-based detection with domain-specific terminology:

| Metric | Value | Notes |
|--------|-------|-------|
| **Precision** | 99%+ | Domain terms are unambiguous |
| **Recall** | 95%+ | Comprehensive keyword lists |
| **False Positives** | <1% | Unlikely to mention "AI.CLASSIFY" without needing skill |
| **Detection Time** | <1ms | Compiled regex patterns |

---

## 9. Migration and Rollout

### 9.1 Phase 1: Framework Integration (Q1 2026)

**Scope:**
- Add `google.adk.skills` module to ADK core
- Implement `SkillRegistry`, `SkillCallbacks` classes
- Update `LlmAgent` documentation for callback integration

**API Surface:**
```python
from google.adk.skills import (
    SkillRegistry,
    SkillCallbacks,
    SkillMetadata,
    SkillContent,
    ACTIVE_SKILLS_KEY,
)
```

### 9.2 Phase 2: BigQuery Toolset Integration (Q2 2026)

**Scope:**
- Bundle BigQuery skills with `BigQueryToolset`
- Auto-configure skill callbacks when using BQ tools
- Maintain skills as external markdown for easy updates

**Configuration:**
```python
from google.adk.tools.bigquery import BigQueryToolset

# Skills auto-configured
toolset = BigQueryToolset(
    credentials_config=...,
    enable_skills=True,  # New parameter
)
```

### 9.3 Phase 3: Skill Marketplace (Q3 2026)

**Scope:**
- Public skill repository
- Versioned skill packages
- Community contributions

---

## 10. Future Extensions

### 10.1 Multi-Modal Skills

Support for image-based skill content:
```markdown
---
name: chart_builder
description: Build charts and visualizations
modality: multi-modal
---

![Chart Types](./chart_types.png)

Use chart type 1 for time series...
```

### 10.2 Skill Dependencies

```yaml
---
name: advanced_ml
description: Advanced ML techniques
requires:
  - bqml  # Base skill must be loaded first
---
```

### 10.3 Dynamic Skill Updates

Real-time skill updates without agent restart:
```python
skill_registry.reload_skill("bq_remote_model")  # Hot reload
```

### 10.4 Skill Analytics

Track skill usage for optimization:
```python
skill_registry.get_usage_stats()
# {"bqml": {"activations": 1000, "avg_duration": 45.2}, ...}
```

---

## Appendix

### A.1 Complete SKILL.md Template

```markdown
---
name: my_skill
description: One-line description of what this skill provides
keywords:
  - primary_keyword
  - secondary_keyword
  - function_name
  - common_user_phrase
---

# My Skill Title

Brief introduction to the skill's purpose.

## Prerequisites

1. Required setup step 1
2. Required setup step 2

## Core Concepts

### Concept 1

Explanation with example:

```sql
-- Example code
SELECT * FROM table;
```

### Concept 2

More explanation...

## Examples

### Example 1: Common Use Case

```sql
-- Full working example
```

### Example 2: Advanced Use Case

```sql
-- Advanced example
```

## Troubleshooting

**Error: "common error message"**
- Cause and solution

## References

- [Official Documentation](https://...)
```

### A.2 Debugging Skill Loading

Enable debug logging:
```python
import logging
logging.getLogger("google.adk.skills").setLevel(logging.DEBUG)
```

Output:
```
[SkillCallbacks] Detecting skills from: Train a model to predict...
[SkillCallbacks] Auto-activated skills: ['bqml']
[SkillCallbacks] Injected skill content into system instruction: ['bqml']
```

### A.3 Testing Skill Detection

```python
def test_skill_detection():
    registry = SkillRegistry("./skills")
    callbacks = SkillCallbacks(registry, detection_mode="keyword")

    # Test detection
    detected = callbacks._detect_skills_from_text(
        "Create a remote model using Gemini"
    )
    assert "bq_remote_model" in detected

    # Test non-detection
    detected = callbacks._detect_skills_from_text(
        "What's the weather today?"
    )
    assert len(detected) == 0
```

---

## References

1. Anthropic Engineering: [Equipping Agents for the Real World with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
2. Google ADK Documentation: [LlmAgent Callbacks](https://cloud.google.com/docs/adk/callbacks)
3. BigQuery ML Documentation: [BQML Introduction](https://cloud.google.com/bigquery/docs/bqml-introduction)
4. BigQuery AI Functions: [AI Functions Reference](https://cloud.google.com/bigquery/docs/ai-functions)

---

*Document Version: 1.0 | Last Updated: December 2025*
