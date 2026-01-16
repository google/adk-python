# Agent Skills Demo

This demo showcases the Agent Skills standard integration with ADK. It demonstrates how to:

1. Load skills from directories using `AgentSkillLoader`
2. Convert skills to ADK tools using `SkillTool`
3. Generate discovery prompts for the LLM
4. Use skills with progressive disclosure

## Running the Demo

```bash
# From the adk-python root directory
adk web contributing/samples/agent_skills_demo
```

## What's Included

### Skills Loaded

The demo loads BigQuery ML skills from `src/google/adk/tools/bigquery/skills/`:

- **bqml** - Machine learning in BigQuery
- **bq-ai-operator** - AI operations (text generation, embeddings)

### Skill Structure (Agent Skills Standard)

Each skill follows the Agent Skills standard format:

```
skill-name/
├── SKILL.md           # Short description and instructions
├── references/        # Detailed documentation
│   ├── MODEL_TYPES.md
│   └── BEST_PRACTICES.md
├── scripts/          # Helper scripts
│   └── validate_model.py
└── assets/           # Templates, configs
```

### Progressive Disclosure

Skills support three stages:

1. **Discovery** (Stage 1): Minimal metadata shown in discovery prompt
2. **Activation** (Stage 2): Full instructions loaded on demand
3. **Execution** (Stage 3): Scripts and references loaded as needed

## Example Interactions

### Discover Available Skills

```
User: What ML skills do you have?
Agent: [Reviews discovery prompt and lists available skills]
```

### Activate a Skill

```
User: Tell me about BQML
Agent: [Activates bqml skill, provides detailed instructions]
```

### Load Reference Documentation

```
User: What model types are available?
Agent: [Loads MODEL_TYPES.md reference, explains options]
```

### Run a Script

```
User: Validate my model configuration
Agent: [Runs validate_model.py script]
```

## Code Walkthrough

```python
from google.adk.skills import AgentSkillLoader, SkillTool

# Load skills from directory
loader = AgentSkillLoader()
loader.add_skill_directory("./skills")

# Create tools for agent
skill_tools = [SkillTool(skill) for skill in loader.get_all_skills()]

# Generate system prompt
discovery_prompt = loader.generate_discovery_prompt()

# Create agent with skill tools
agent = LlmAgent(
    model="gemini-2.0-flash",
    instruction=f"Available skills:\n{discovery_prompt}",
    tools=skill_tools,
)
```

## Customization

### Adding Custom Skills

1. Create a skill directory following the Agent Skills standard
2. Add `SKILL.md` with YAML frontmatter
3. Add references, scripts, and assets as needed
4. Update the `SKILLS_DIR` path in `agent.py`

### Skill Configuration

Skills can include ADK-specific configuration in their frontmatter:

```yaml
adk:
  config:
    timeout_seconds: 300
    max_parallel_calls: 5
  allowed_callers:
    - my_agent
```

## Learn More

- [Agent Skills Standard](https://agentskills.io)
- [ADK Skills Documentation](../../docs/skills_programmatic_tool_calling_design.md)
