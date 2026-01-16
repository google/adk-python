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

"""Agent Skills Demo - demonstrates Agent Skills standard integration.

This demo shows how to:
1. Load skills from directories using AgentSkillLoader
2. Convert skills to ADK tools using SkillTool
3. Generate discovery prompts for the LLM
4. Use skills with progressive disclosure
5. Integrate with BigQuery toolset for data operations

Run with: adk web contributing/samples
Then select 'agent_skills_demo' from the list.
"""

from __future__ import annotations

import os
from pathlib import Path

import google.auth

from google.adk.agents.llm_agent import LlmAgent
from google.adk.skills import AgentSkillLoader
from google.adk.skills import SkillTool
from google.adk.tools.bigquery.bigquery_credentials import BigQueryCredentialsConfig
from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.bigquery.config import WriteMode

# Default Vertex AI settings
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", os.getenv("VERTEXAI_PROJECT", ""))
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", os.getenv("VERTEXAI_LOCATION", "us-central1"))

# Path to skills directory (now in main skills module)
SKILLS_DIR = Path(__file__).parent.parent.parent.parent / "src" / "google" / "adk" / "skills"

# Initialize the skill loader
skill_loader = AgentSkillLoader(validate_names=False)

# Load skills from the BigQuery skills directory
if SKILLS_DIR.exists():
    skill_count = skill_loader.add_skill_directory(SKILLS_DIR)
    print(f"Loaded {skill_count} skills from {SKILLS_DIR}")
else:
    print(f"Warning: Skills directory not found: {SKILLS_DIR}")

# Create skill tools for the agent
skill_tools = [SkillTool(skill) for skill in skill_loader.get_all_skills()]

# Generate discovery prompt for the system instruction
discovery_prompt = skill_loader.generate_discovery_prompt(include_resources=True)

# Setup BigQuery toolset with Application Default Credentials
try:
    application_default_credentials, _ = google.auth.default()
    credentials_config = BigQueryCredentialsConfig(
        credentials=application_default_credentials
    )

    # Configure BigQuery tools - allow writes for ML model creation
    tool_config = BigQueryToolConfig(
        write_mode=WriteMode.ALLOWED,
        application_name="agent_skills_demo"
    )

    bigquery_toolset = BigQueryToolset(
        credentials_config=credentials_config,
        bigquery_tool_config=tool_config
    )
    print("BigQuery toolset initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize BigQuery toolset: {e}")
    bigquery_toolset = None

# Build the system instruction
SYSTEM_INSTRUCTION = f"""You are a data science assistant with access to BigQuery ML skills and BigQuery tools.

## Available Skills

{discovery_prompt}

## How to Use Skills

1. **Discover**: When asked about ML or AI capabilities, review the available skills above
2. **Activate**: Use the skill tool with action="activate" to get detailed instructions
3. **Execute**: Use action="run_script" to run helper scripts or action="load_reference" to load detailed documentation

## BigQuery Tools

You also have access to BigQuery tools for:
- Executing SQL queries
- Exploring datasets and tables
- Getting table schemas and metadata

## Guidelines

- Always activate a skill before using its detailed features
- Load references when you need detailed documentation
- Run scripts for validation and utility tasks
- Use BigQuery tools to explore data and run queries
- Explain what you're doing and why

## Example Workflow

User: "How do I create a classification model in BigQuery?"

1. Activate the bqml skill to get instructions
2. Load MODEL_TYPES.md reference for model type details
3. Use BigQuery tools to explore available data
4. Provide guidance based on the skill's documentation
"""

# Combine all tools
all_tools = skill_tools.copy()
if bigquery_toolset:
    all_tools.append(bigquery_toolset)

# Create the agent with Gemini 2.5 Pro
root_agent = LlmAgent(
    model="gemini-2.5-pro",
    name="agent_skills_demo",
    description="A demo agent showcasing Agent Skills standard integration with ADK and BigQuery tools.",
    instruction=SYSTEM_INSTRUCTION,
    tools=all_tools,
)

# Print summary on load
if __name__ == "__main__" or True:
    print("\n" + "=" * 60)
    print("Agent Skills Demo")
    print("=" * 60)
    print(f"\nModel: gemini-2.5-pro")
    print(f"Loaded skills: {skill_loader.get_skill_names()}")
    print(f"Skill tools: {[t.name for t in skill_tools]}")
    print(f"BigQuery toolset: {'enabled' if bigquery_toolset else 'disabled'}")
    print("\nRun with: adk web contributing/samples")
    print("Then select 'agent_skills_demo' from the list")
    print("=" * 60 + "\n")
