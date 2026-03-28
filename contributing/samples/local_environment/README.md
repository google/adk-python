# Local Environment Sample

This sample demonstrates how to use the `LocalEnvironment` with the `EnvironmentToolset` to allow an agent to interact with the local filesystem and execute commands.

## Description

The agent is configured with the `EnvironmentToolset`, which provides tools for file I/O (reading, writing) and command execution within a local environment. This allows the agent to perform tasks that involve creating files, modifying them, and running local scripts or commands.

## Sample Usage

You can interact with the agent by providing prompts that require file operations and command execution.

### Example Prompt

> "Write a Python file named `hello.py` to the working directory that prints 'Hello from ADK!'. Then read the file to verify its contents, and finally execute it using a command."

### Expected Behavior

1.  **Write File**: The agent uses a tool to write `hello.py` with the content `print("Hello from ADK!")`.
2.  **Read File**: The agent uses a tool to read `hello.py` and verify the content.
3.  **Execute Command**: The agent uses a tool to run `python3 hello.py` and returns the output.
