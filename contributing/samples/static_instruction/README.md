# Bingo Digital Pet Agent

This sample agent demonstrates static instruction functionality through a lovable digital pet named Bingo! The agent showcases how static instructions (personality) are placed in system_instruction for caching while dynamic instructions are added to user contents, affecting the cacheable prefix of the final model prompt.

**Prompt Construction & Caching**: The final model prompt is constructed as: `system_instruction + tools + tool_config + contents`. Static instructions are placed in system_instruction, while dynamic instructions are appended to user contents (which are part of contents along with historical chat history). This means the prefix (system_instruction + tools + tool_config) remains cacheable while only the contents portion changes between requests.

## Features

### Static Instructions (Bingo's Personality)
- **Constant personality**: Core traits and behavior patterns never change
- **Context caching**: Personality definition is cached for performance
- **Base character**: Defines Bingo as a friendly, energetic digital pet companion

### Dynamic Instructions (Hunger-Based Moods)
- **Ultra-fast hunger progression**: full (0-2s) → satisfied (2-6s) → a_little_hungry (6-12s) → hungry (12-24s) → very_hungry (24-36s) → starving (36s+)
- **Session-aware**: Mood changes based on feeding timestamp in session state
- **Realistic behavior**: Different responses based on how hungry Bingo is

### Tools
- **eat**: Allows users to feed Bingo, updating session state with timestamp

## Usage

### Setup API Credentials

Create a `.env` file in the project root with your API credentials:

```bash
# Choose Model Backend: 0 -> ML Dev, 1 -> Vertex
GOOGLE_GENAI_USE_VERTEXAI=1

# ML Dev backend config
GOOGLE_API_KEY=your_google_api_key_here

# Vertex backend config
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
```

The agent will automatically load environment variables on startup.

### Default Behavior (Hunger State Demonstration)
Run the agent to see Bingo in different hunger states:

```bash
cd contributing/samples
PYTHONPATH=../../src python -m static_instruction.main
```

This will demonstrate all hunger states by simulating different feeding times and showing how Bingo's mood changes while his core personality remains cached.

### Interactive Chat with Bingo (adk web)

For a more interactive experience, use the ADK web interface to chat with Bingo in real-time:

```bash
cd contributing/samples
PYTHONPATH=../../src adk web .
```

This will start a web interface where you can:
- **Select the agent**: Choose "static_instruction" from the dropdown in the top-left corner
- **Chat naturally** with Bingo and see his personality
- **Feed him** using commands like "feed Bingo" or "give him a treat"
- **Watch hunger progression** as Bingo gets hungrier over time
- **See mood changes** in real-time based on his hunger state
- **Experience begging** when Bingo gets very hungry and asks for food

The web interface shows how static instructions (personality) remain cached while dynamic instructions (hunger state) change based on your interactions and feeding times.

### Sample Prompts for Feeding Bingo

When chatting with Bingo, you can feed him using prompts like:

**Direct feeding commands:**
- "Feed Bingo"
- "Give Bingo some food"
- "Here's a treat for you"
- "Time to eat, Bingo!"
- "Have some kibble"

**When Bingo is begging for food:**
- Listen for Bingo saying things like "I'm so hungry", "please feed me", "I need food"
- Respond with feeding commands above
- Bingo will automatically use the eat tool when very hungry/starving

## Agent Structure

```
static_instruction/
├── __init__.py      # Package initialization
├── agent.py         # Main agent definition with static/dynamic instructions
├── main.py          # Runner script with hunger state demonstration
└── README.md        # This documentation
```
