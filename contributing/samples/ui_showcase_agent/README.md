# UI Showcase Agent

This agent demonstrates various ways information can be displayed in the Agent Development Kit (ADK) web UI.

## Features Demonstrated

*   **Thinking Process:** When you ask the agent a general question that doesn't require a specific tool, it will show its "thinking" process before providing an answer. This is achieved by using a `BuiltInPlanner` with `ThinkingConfig(include_thoughts=True)`.
*   **Function Calls:** The agent has a simple `greet_user` tool. When you ask the agent to greet someone, you will see the UI representation of the function call being made and the result returned by the tool.
*   **Text Artifacts:** The agent can create and save text files as artifacts. You can specify the content and filename. These artifacts will appear in the UI, and you should be able to view their content.
*   **Image Artifacts (Placeholders):** The agent can create and save placeholder image artifacts. This demonstrates how image artifacts might be displayed. In this example, it saves a text file with a `.png` extension containing a message about the simulated image.

## How to Use

Interact with the agent using the following types of prompts:

1.  **To see the thinking process:**
    *   "What is the capital of France?"
    *   "Explain black holes."
    *   Any general knowledge question.

2.  **To see a function call:**
    *   "Greet Mary"
    *   "Say hello to John"

3.  **To create a text artifact:**
    *   "Create a text artifact with content 'Hello world from the ADK!'"
    *   "Make a text file named 'notes.txt' with content 'Remember to buy milk.'"

4.  **To create an image artifact (placeholder):**
    *   "Create an image artifact with prompt 'a futuristic city'"
    *   "Generate a picture with prompt 'a serene landscape' and name it 'landscape.png'"

## Agent Configuration

*   **Model:** `gemini-2.0-flash-exp`
*   **Planner:** `BuiltInPlanner` with `ThinkingConfig(include_thoughts=True)`
*   **Tools:**
    *   `create_text_artifact(tool_context: ToolContext, content: str, filename: str = "showcase_text.txt")`
    *   `create_image_artifact(tool_context: ToolContext, prompt: str, filename: str = "showcase_image_placeholder.png")`
    *   `greet_user(tool_context: ToolContext, name: str)`
