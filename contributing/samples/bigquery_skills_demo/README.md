# BigQuery Skills Demo

This sample demonstrates Anthropic's [Agent Skills Pattern](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) for dynamic skill discovery with BigQuery ML and AI capabilities, enhanced with **callback-based automatic skill loading**.

## Overview

This demo showcases:
- **Dynamic Skill Discovery**: Skills are discovered at runtime from SKILL.md files with YAML frontmatter
- **Progressive Disclosure**: Only skill names/descriptions loaded initially; full content on-demand
- **Callback-based Auto-activation**: Skills are automatically activated based on keywords in user messages (no LLM calls needed!)
- **Ephemeral Skill Loading**: Skills are injected into the system prompt (not conversation history) and can be truly unloaded
- **Automatic Cleanup**: Skills are auto-deactivated after each turn to free up context
- **Scalable Design**: Adding new skills requires only a SKILL.md file - no code changes!

### Available Skills

1. **bqml** - BigQuery ML for training and deploying ML models in SQL
   - Model training (LINEAR_REG, LOGISTIC_REG, KMEANS, ARIMA_PLUS, XGBoost, etc.)
   - Model evaluation and prediction
   - Feature importance and model analysis

2. **bq_ai_operator** - Managed AI functions in BigQuery SQL
   - AI.CLASSIFY: Categorize text into classes
   - AI.IF: Natural language TRUE/FALSE filtering
   - AI.SCORE: Rate/rank content by criteria (0.0 to 1.0)

3. **bq_remote_model** - Remote models connecting to Vertex AI
   - CREATE REMOTE MODEL: Connect to Gemini, Claude, Llama, and custom endpoints
   - AI.GENERATE_TEXT: Text generation with LLMs
   - AI.GENERATE_EMBEDDING: Vector embeddings for semantic search

## Prerequisites

1. Google Cloud project with BigQuery and Vertex AI enabled
2. Application Default Credentials configured:
   ```bash
   gcloud auth application-default login
   ```
3. Set your project ID:
   ```bash
   export GOOGLE_CLOUD_PROJECT=your-project-id
   ```

### For AI Functions (bq_ai_operator and bq_remote_model skills)

Create a BigQuery connection to Vertex AI:
```bash
bq mk --connection \
  --location=us \
  --project_id=$GOOGLE_CLOUD_PROJECT \
  --connection_type=CLOUD_RESOURCE \
  my_ai_connection
```

Grant the connection's service account access to Vertex AI:
```bash
# Get the service account
bq show --connection $GOOGLE_CLOUD_PROJECT.us.my_ai_connection

# Grant access (replace with actual service account)
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
  --role="roles/aiplatform.user"
```

## Running the Demo

### Option 1: Run with ADK CLI

```bash
cd contributing/samples/bigquery_skills_demo
adk run .
```

### Option 2: Run the web UI

```bash
adk web contributing/samples --port 8000
# Open http://127.0.0.1:8000/dev-ui/?app=bigquery_skills_demo
```

## Example Prompts

### BQML Skill (auto-activated by: "train", "model", "predict", "regression", "kmeans")
```
Train a linear regression model to predict penguin body weight using
the public penguins dataset, then evaluate it and show feature importance.
```

### BQ AI Operator Skill (auto-activated by: "classify", "AI.CLASSIFY", "sentiment", "categorize")
```
Classify 5 BBC news articles by their topic using AI.CLASSIFY with
categories: tech, sport, business, politics, entertainment, other.
```

### BQ Remote Model Skill (auto-activated by: "generate text", "gemini", "embeddings", "llm")
```
Create a remote model using Gemini 2.0 Flash and use it to summarize
product descriptions from my table.
```

## How It Works

### Architecture: Callback-based Skill Management

This demo uses ADK callbacks instead of LLM tool calls for skill management:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Message                                 │
│              "Train a model to predict penguin weight"              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    before_model_callback                             │
│   1. Extract keywords from user message                             │
│   2. Match against skill keywords (from SKILL.md frontmatter)       │
│   3. Activate matching skills: ["bqml"]                             │
│   4. Skills injected into system prompt via instruction provider    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LLM Processing                               │
│   System prompt now includes:                                        │
│   - Base instruction                                                 │
│   - Active skill documentation (BQML syntax, examples)              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    after_agent_callback                              │
│   1. Clear active skills from state                                  │
│   2. Context freed for next interaction                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Skill Discovery** (`skill_registry.py`)
   - Scans `skills/` directory for SKILL.md files
   - Parses YAML frontmatter (name, description, keywords)
   - Provides instruction provider for ephemeral skill injection

2. **Skill Callbacks** (`skill_callbacks.py`)
   - `before_model_callback`: Detects and activates skills based on keywords
   - `after_agent_callback`: Cleans up skills after each turn
   - Supports multiple detection modes: keyword (recommended), hybrid, llm

3. **Agent Configuration** (`agent.py`)
   - Registers callbacks with LlmAgent
   - Combines base instruction with dynamic skill content
   - Manual skill tools available as fallback

### Callback vs Tool Approach Comparison

| Aspect | Callback (This Demo) | Tool-based |
|--------|---------------------|------------|
| **LLM Calls** | Zero for skill management | 1-2 per skill activation |
| **Latency** | Instant (keyword matching) | Adds round-trip time |
| **Cost** | No additional tokens | Extra tool call tokens |
| **Control** | System-level, deterministic | LLM decides when to activate |
| **Best For** | Domain-specific terms (BigQuery) | Semantic understanding needed |

### Why Keyword Detection for BigQuery?

BigQuery has **highly domain-specific terminology** that makes keyword detection ideal:
- "BQML", "ML.PREDICT", "CREATE MODEL" → bqml skill
- "AI.CLASSIFY", "AI.IF", "AI.SCORE" → bq_ai_operator skill
- "GENERATE_TEXT", "gemini", "embeddings" → bq_remote_model skill

These terms are unambiguous - you don't need an LLM to understand that "AI.CLASSIFY" relates to the AI operator skill.

## Code Structure

```
bigquery_skills_demo/
├── __init__.py           # Module init
├── agent.py              # Agent with BigQuery tools and callbacks
├── skill_registry.py     # Dynamic skill discovery + instruction provider
├── skill_callbacks.py    # Callback-based auto-activation
├── skill_classifier.py   # Optional LLM-based classification (for hybrid mode)
├── skills/
│   ├── bqml/
│   │   └── SKILL.md      # BQML skill (keywords: train, model, predict, etc.)
│   ├── bq_ai_operator/
│   │   └── SKILL.md      # AI operator skill (keywords: classify, sentiment, etc.)
│   └── bq_remote_model/
│       └── SKILL.md      # Remote model skill (keywords: gemini, embeddings, etc.)
└── README.md             # This file
```

## Adding New Skills

Adding a new skill requires **only a SKILL.md file** - no code changes needed!

1. Create a directory under `skills/` (e.g., `skills/my_skill/`)
2. Add a `SKILL.md` file with YAML frontmatter:
   ```markdown
   ---
   name: my_skill
   description: Short description of what this skill does
   keywords:
     - keyword1
     - keyword2
     - specific_function_name
   ---

   # My Skill Documentation

   Detailed instructions, examples, and usage patterns...
   ```
3. The skill will be automatically discovered and keyword patterns built from frontmatter

### Keyword Guidelines

- Use domain-specific terms that clearly indicate the skill is needed
- Include function names (e.g., "ML.PREDICT", "AI.CLASSIFY")
- Include common user phrases (e.g., "train", "classify", "embeddings")
- Multiple keywords increase detection coverage

## Detection Modes

The `SkillCallbacks` class supports three detection modes:

```python
# In agent.py
skill_callbacks = SkillCallbacks(
    skill_registry,
    auto_deactivate=True,
    detection_mode="keyword",  # "keyword" | "hybrid" | "llm"
)
```

| Mode | Description | Best For |
|------|-------------|----------|
| `keyword` | Regex pattern matching from SKILL.md keywords | Domain-specific terms (recommended) |
| `hybrid` | LLM classification with keyword fallback | Mixed semantic/specific queries |
| `llm` | Pure LLM-based semantic classification | Paraphrased/ambiguous requests |

## References

- [Anthropic: Equipping Agents with Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [BigQuery ML Documentation](https://cloud.google.com/bigquery/docs/bqml-introduction)
- [BigQuery AI Functions](https://cloud.google.com/bigquery/docs/ai-functions)
- [BigQuery Remote Models](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create-remote-model)
