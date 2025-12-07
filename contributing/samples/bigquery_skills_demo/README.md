# BigQuery Skills Demo

This sample demonstrates Anthropic's [Agent Skills Pattern](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) for dynamic skill discovery with BigQuery ML and AI capabilities, enhanced with **Claude Code-style ephemeral skill loading**.

## Overview

This demo showcases:
- **Dynamic Skill Discovery**: Skills are discovered at runtime from SKILL.md files
- **Progressive Disclosure**: Only skill names/descriptions loaded initially; full content on-demand
- **Ephemeral Skill Loading**: Skills are injected into the system prompt (not conversation history) and can be truly unloaded when no longer needed
- **Context Management**: Agent can activate/deactivate skills to manage context efficiently

### Available Skills

1. **bqml** - BigQuery ML for training and deploying ML models in SQL
   - Model training (LINEAR_REG, LOGISTIC_REG, KMEANS, ARIMA_PLUS, XGBoost, etc.)
   - Model evaluation and prediction
   - Feature importance and model analysis

2. **bq_ai_operator** - Managed AI functions in BigQuery SQL
   - AI.CLASSIFY: Categorize text into classes
   - AI.IF: Natural language TRUE/FALSE filtering
   - AI.SCORE: Rate/rank content by criteria (0.0 to 1.0)

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

### For AI Functions (bq_ai_operator skill)

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

### BQML Skill
```
Train a linear regression model to predict penguin body weight using
the public penguins dataset, then evaluate it and show feature importance.
```

### BQ AI Operator Skill
```
Classify 5 BBC news articles by their topic using AI.CLASSIFY with
categories: tech, sport, business, politics, entertainment, other.
```

## How It Works

1. **Skill Discovery**: The `SkillRegistry` scans the `skills/` directory for SKILL.md files
2. **YAML Frontmatter**: Each SKILL.md has metadata (name, description) in YAML frontmatter
3. **Progressive Loading**:
   - Level 1: Agent sees skill names and descriptions in its system prompt
   - Level 2: Agent calls `activate_skill(skill_name)` to load full documentation
4. **Ephemeral Loading (Claude Code-style)**:
   - Active skills are injected into the **system prompt**, not conversation history
   - Skills can be deactivated with `deactivate_skill(skill_name)` to free up context
   - The system prompt is rebuilt fresh each LLM call, so deactivated skills truly disappear
   - This prevents context accumulation unlike traditional tool responses

### Key Difference from Traditional Approaches

Traditional skill loading returns skill content as a tool response, which persists in conversation history forever. This demo uses ADK's **InstructionProvider** pattern:

```python
# Traditional (persistent) - skill content stays in history
def load_skill(skill_name: str) -> str:
    return skill_content  # This persists in conversation history

# Ephemeral (this demo) - skill content injected into system prompt
def instruction_provider(ctx: ReadonlyContext) -> str:
    active_skills = ctx.state.get("active_skills", [])
    return build_system_prompt_with_skills(active_skills)
```

Benefits:
- **True unloading**: Deactivated skills are removed from context
- **Better context management**: Agent can activate skills when needed, deactivate when done
- **Mirrors Claude Code**: Similar to how Claude Code loads skills from filesystem on-demand

## Code Structure

```
bigquery_skills_demo/
├── __init__.py           # Module init
├── agent.py              # Agent with BigQuery tools and load_skill
├── skill_registry.py     # Dynamic skill discovery (Anthropic pattern)
├── skills/
│   ├── bqml/
│   │   └── SKILL.md      # BQML skill documentation
│   └── bq_ai_operator/
│       └── SKILL.md      # AI operator skill documentation
└── README.md             # This file
```

## Adding New Skills

1. Create a directory under `skills/` (e.g., `skills/my_skill/`)
2. Add a `SKILL.md` file with YAML frontmatter:
   ```markdown
   ---
   name: my_skill
   description: Short description of what this skill does
   ---

   # My Skill Documentation

   Detailed instructions, examples, and usage patterns...
   ```
3. The skill will be automatically discovered on agent startup

## References

- [Anthropic: Equipping Agents with Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [BigQuery ML Documentation](https://cloud.google.com/bigquery/docs/bqml-introduction)
- [BigQuery AI Functions](https://cloud.google.com/bigquery/docs/ai-functions)
