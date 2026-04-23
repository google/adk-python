# Local Environment Skill Sample

This sample demonstrates how to use the `LocalEnvironment` with the `EnvironmentToolset` to allow an agent to manually discover and load skills from the environment, rather than using the pre-configured `SkillToolset`.

## Description

The agent is configured with the `EnvironmentToolset` and is initialized with a `LocalEnvironment` pointing to the agent's directory.
Instead of having skills pre-loaded, the agent uses system instructions that guide it to search for skills in the `skills/` folder and load them by reading their `SKILL.md` files using the `ReadFile` tool.

This demonstrates a "manual skill loading" pattern where the agent can acquire new capabilities dynamically by reading instructions from the environment.

## Sample Usage

You can interact with the agent by providing prompts that require a specific skill (like weather).

### Example Prompt

> "Can you check the weather in Sunnyvale?"

### Expected Behavior

1.  **Find Skill**: The agent uses the `Execute` tool to search for all available skills by running `find skills -name SKILL.md`.
2.  **Load Skill**: The agent identifies the relevant skill and uses the `ReadFile` tool to read its `SKILL.md` file.
3.  **Execute Skill**: The agent follows the instructions in the skill file (e.g., reading references or running scripts) to answer the user's request.
