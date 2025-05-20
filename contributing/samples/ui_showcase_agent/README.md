# UI Showcase Agent

This agent demonstrates various ways information can be displayed in the Agent Development Kit (ADK) web UI.

## Features Demonstrated

*   **Thinking Process:** When you ask the agent a general question that doesn't require a specific tool, it will show its "thinking" process before providing an answer. This is achieved by using a `BuiltInPlanner` with `ThinkingConfig(include_thoughts=True)`.
*   **Function Calls:** The agent has a simple `greet_user` tool. When you ask the agent to greet someone, you will see the UI representation of the function call being made and the result returned by the tool.
*   **Text Artifacts:** The agent can create and save text files as artifacts. You can specify the content and filename. These artifacts will appear in the UI, and you should be able to view their content.
*   **Image Artifacts (Placeholders):** The agent can create and save placeholder image artifacts. This demonstrates how image artifacts might be displayed. In this example, it saves a text file with a `.png` extension containing a message about the simulated image.
*   **User Choice Prompts:** The agent can use the `get_user_choice_tool` to present a list of options to the user, from which the user can select one. This demonstrates how the UI can handle agent-initiated requests for user selection.
*   **Complex Tool Input Forms (Manual Invocation):** The agent includes a tool (`process_complex_data`) that defines its input arguments using a Pydantic model with various field types (text, integer, boolean, list). When a user attempts to manually run such a tool from the ADK web interface (e.g., via a "Run tool" button or similar, after inspecting the agent's tools), the UI may automatically generate a form to help the user provide the structured input. This showcases the ADK's support for user-friendly tool invocation.

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

5.  **To see a user choice prompt:**
    *   Ask the agent: "Request user choice with options Apples, Bananas, Oranges".
    *   The agent should then invoke the `get_user_choice_tool`, and the UI should present you with these options to choose from.

6.  **To observe complex tool input forms (manual invocation):**
    *   First, you might want to know the exact name and parameters of the tool. Ask the agent: "What are your tools?" or "Describe the process_complex_data tool".
    *   Then, look for a way in the ADK web UI to manually run a tool (there might be a list of tools with a 'run' button, or you might be able to type a special command to trigger a tool manually). When you try to run `process_complex_data`, the UI should display a form with fields for 'text_input', 'integer_input', 'boolean_input', and 'list_input', corresponding to its defined Pydantic schema.

## Agent Configuration

*   **Model:** `gemini-2.0-flash-exp`
*   **Planner:** `BuiltInPlanner` with `ThinkingConfig(include_thoughts=True)`
*   **Tools:**
    *   `create_text_artifact(tool_context: ToolContext, content: str, filename: str = "showcase_text.txt")`
    *   `create_image_artifact(tool_context: ToolContext, prompt: str, filename: str = "showcase_image_placeholder.png")`
    *   `greet_user(tool_context: ToolContext, name: str)`
    *   `get_user_choice_tool(options: list[str], tool_context: ToolContext)`
    *   `process_complex_data(tool_context: ToolContext, data: ComplexDataToolInput)` (Input `data` is structured via the `ComplexDataToolInput` Pydantic model: `{text_input: str, integer_input: int, boolean_input: bool, list_input: list[str]}`)
