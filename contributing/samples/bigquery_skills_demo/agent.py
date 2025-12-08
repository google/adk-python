# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BigQuery Skills Demo Agent with Dynamic Skill Discovery.

This agent demonstrates the Anthropic Skills Pattern for dynamic capability
discovery, as described in:
https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills

Key Features:
1. **Progressive Disclosure**: Skills are discovered at startup with only
   names and descriptions loaded. Full content is loaded on-demand.

2. **Dynamic Discovery**: Skills are stored as SKILL.md files and discovered
   automatically from the skills/ directory.

3. **load_skill Tool**: The agent can load full skill documentation when
   it determines a skill is relevant to the current task.

Available Skills:
- bqml: BigQuery ML for training/deploying ML models in SQL
- bq_ai_operator: Generative AI functions in SQL

To run this demo:
    cd contributing/samples/bigquery_skills_demo
    adk run .

Or via web UI:
    adk web contributing/samples --port 8000
    # Then open http://127.0.0.1:8000/dev-ui/?app=bigquery_skills_demo
"""

import os

# Set environment variables for Vertex AI (uses ADC for authentication)
# Users should set GOOGLE_CLOUD_PROJECT to their own project ID
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.bigquery import BigQueryCredentialsConfig
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.bigquery.config import WriteMode
import google.auth

# Import the dynamic skill registry with ephemeral skill loading
from .skill_registry import (
    SkillRegistry,
    create_skill_instruction_provider,
    activate_skill,
    deactivate_skill,
    list_active_skills,
)

# Import callback-based skill management (saves LLM calls)
from .skill_callbacks import SkillCallbacks

# Agent name
AGENT_NAME = "bigquery_skills_demo_agent"

# Project configuration - must be set via environment variable
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError(
        "GOOGLE_CLOUD_PROJECT environment variable must be set. "
        "Set it to your GCP project ID before running this demo."
    )

# Initialize BigQuery tool config
# Using ALLOWED write mode to enable CREATE MODEL operations
tool_config = BigQueryToolConfig(
    write_mode=WriteMode.ALLOWED,
    application_name=AGENT_NAME,
)

# Use application default credentials
application_default_credentials, _ = google.auth.default()
credentials_config = BigQueryCredentialsConfig(
    credentials=application_default_credentials
)

# Initialize BigQuery toolset
bigquery_toolset = BigQueryToolset(
    credentials_config=credentials_config,
    bigquery_tool_config=tool_config,
)

# Initialize dynamic skill registry
skill_registry = SkillRegistry()
SKILLS_SUMMARY = skill_registry.get_skills_summary()

# Create instruction provider for ephemeral skill loading (Claude Code-style)
# This injects active skills into the system prompt, not conversation history
skill_instruction_provider = create_skill_instruction_provider(skill_registry)

# Create skill management tools (kept for manual override if needed)
activate_skill_tool = FunctionTool(activate_skill)
deactivate_skill_tool = FunctionTool(deactivate_skill)
list_active_skills_tool = FunctionTool(list_active_skills)

# Create callback-based skill management with keyword detection
# Modes: "keyword" (recommended for BigQuery), "hybrid", "llm"
# Keyword mode is fast, no API calls, and BigQuery terms are domain-specific enough
skill_callbacks = SkillCallbacks(
    skill_registry,
    auto_deactivate=True,
    detection_mode="keyword",  # Fast keyword detection - ideal for domain-specific terms
)

# Base instruction for the agent (static part)
BASE_INSTRUCTION = f"""\
You are a data science agent with BigQuery capabilities and automatic skill loading.

## How Skills Work (Keyword-based Auto-activation)

You have access to specialized skills that provide detailed guidance for complex tasks.
**Skills are automatically activated** based on keywords in the user's message - no API calls needed!

Skills are:
- **Auto-activated by keywords**: Domain-specific terms trigger skill loading instantly
  - "train", "model", "predict", "regression", "kmeans" → bqml
  - "classify", "AI.CLASSIFY", "sentiment", "categorize" → bq_ai_operator
  - "generate text", "gemini", "embeddings", "remote model" → bq_remote_model
- **Auto-deactivated**: After you complete your response, skills are cleared to free context
- **Multi-skill support**: Multiple skills can be loaded simultaneously if needed

**This is efficient**: Keyword detection is instant with zero API calls!

**Current Available Skills:**

{SKILLS_SUMMARY}

**Manual Skill Management (optional, for override):**
- `activate_skill(skill_name)`: Manually load a skill if auto-detection missed it
- `deactivate_skill(skill_name)`: Manually unload a skill (rarely needed)
- `list_active_skills()`: See which skills are currently loaded

**Note**: In most cases, just focus on the task - skills will be loaded automatically!

## Available BigQuery Tools

- `execute_sql`: Run any BigQuery SQL (queries, DDL, BQML, AI functions)
- `get_table_info`: Get schema information for a table
- `list_dataset_ids`: List datasets in a project
- `list_table_ids`: List tables in a dataset
- `list_connections`: **ALWAYS use this first** to discover existing BigQuery connections
- `create_connection`: Create a new BigQuery connection (auto-grants Vertex AI User role)

## Connection Management (IMPORTANT)

For AI functions and remote models, you need a BigQuery connection to Vertex AI.

**ALWAYS follow this workflow:**
1. **First**: Call `list_connections(project_id, location)` to discover existing connections
2. **If connections exist**: Use one with `connection_type: "CLOUD_RESOURCE"`
3. **Only if no connections**: Call `create_connection(project_id, location, connection_id)`
   - This automatically grants the Vertex AI User IAM role to the service account

## Project Configuration

- Project ID: {PROJECT_ID}
- Available public datasets: `bigquery-public-data.ml_datasets` (penguins, census, etc.)

## Workflow Example

1. User asks: "Train a model to predict penguin weight"
2. **[AUTOMATIC]** The bqml skill is loaded based on keywords "train", "model", "predict"
3. You follow the skill's examples to CREATE MODEL, EVALUATE, and PREDICT
4. You explain results to the user
5. **[AUTOMATIC]** Skills are cleared after your response

## Guidelines

1. **Focus on the task**: Skills load automatically - no need to call activate_skill()
2. **Explore data first**: Use `get_table_info` or `SELECT * LIMIT 5` before complex queries
3. **Use LIMIT**: Prevent large result sets with `LIMIT 10-100`
4. **Explain your steps**: Describe what each query does and interpret results

## Quick Reference (without loading skills)

**BQML Quick Start:**
```sql
-- Train: CREATE OR REPLACE MODEL `project.dataset.model` OPTIONS(model_type='LINEAR_REG', ...)
-- Evaluate: SELECT * FROM ML.EVALUATE(MODEL `project.dataset.model`)
-- Predict: SELECT * FROM ML.PREDICT(MODEL `project.dataset.model`, ...)
```

**AI Operator Quick Start:**
```sql
-- Classify: AI.CLASSIFY(text, categories => [...], connection_id => 'loc.conn')
-- Filter: AI.IF(text, 'condition', connection_id => 'loc.conn')
-- Score: AI.SCORE(text, 'criteria', connection_id => 'loc.conn')
```

**Remote Model Quick Start:**
```sql
-- Create: CREATE REMOTE MODEL `project.dataset.model` REMOTE WITH CONNECTION `loc.conn` OPTIONS(ENDPOINT='gemini-2.0-flash')
-- Generate: SELECT * FROM AI.GENERATE_TEXT(MODEL `project.dataset.model`, (SELECT 'prompt' AS prompt), STRUCT(...))
-- Embed: SELECT * FROM AI.GENERATE_EMBEDDING(MODEL `project.dataset.model`, (SELECT content FROM table), STRUCT(...))
```

For detailed syntax and examples, skills are loaded automatically based on your question!
"""


def create_combined_instruction_provider(base_instruction: str, skill_provider):
    """Create an instruction provider that combines base instruction with active skills.

    This follows the Claude Code pattern where skills are injected into the system prompt
    (ephemeral) rather than conversation history (persistent).
    """
    from google.adk.agents.readonly_context import ReadonlyContext

    def combined_provider(ctx: ReadonlyContext) -> str:
        # Get dynamic skill content
        skill_content = skill_provider(ctx)

        if skill_content:
            return f"{base_instruction}\n\n{skill_content}"
        return base_instruction

    return combined_provider


# Create the combined instruction provider
instruction_provider = create_combined_instruction_provider(
    BASE_INSTRUCTION, skill_instruction_provider
)

# Create the root agent with BigQuery tools, ephemeral skill loading, and callbacks
root_agent = LlmAgent(
    model="gemini-2.5-pro",
    name=AGENT_NAME,
    description=(
        "Data science agent with BigQuery ML and AI capabilities. "
        "Uses callback-based automatic skill activation - skills are loaded based on "
        "user input keywords and unloaded after each turn to manage context efficiently."
    ),
    instruction=instruction_provider,
    tools=[
        bigquery_toolset,
        # Keep manual tools as fallback (agent can still explicitly manage skills)
        activate_skill_tool,
        deactivate_skill_tool,
        list_active_skills_tool,
    ],
    # Callback-based skill management (saves LLM calls)
    before_model_callback=skill_callbacks.before_model_callback,
    after_agent_callback=skill_callbacks.after_agent_callback,
)
