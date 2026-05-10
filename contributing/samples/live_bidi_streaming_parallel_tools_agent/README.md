# Simple Live (Bidi-Streaming) Agent with Parallel Tools

This project provides a basic example of a live, [bidirectional streaming](https://google.github.io/adk-docs/streaming/) agent that demonstrates parallel tool execution.

## Getting Started

Follow these steps to get the agent up and running:

1. **Start the ADK Web Server**
   Open your terminal, navigate to the root directory that contains the
   `live_bidi_streaming_parallel_tools_agent` folder, and execute the following
   command:

   ```bash
   adk web
   ```

1. **Access the ADK Web UI**
   Once the server is running, open your web browser and navigate to the URL
   provided in the terminal (it will typically be `http://localhost:8000`).

1. **Select the Agent**
   In the top-left corner of the ADK Web UI, use the dropdown menu to select
   this agent (`live_bidi_streaming_parallel_tools_agent`).

1. **Start Streaming**
   Click on the **Audio** icon located near the chat input
   box to begin the streaming session.

1. **Interact with the Agent**
   You can now begin talking to the agent, and it will respond in real-time.
   Try asking it to perform multiple actions at once, for example: "Turn on the
   lights and the TV at the same time." The agent will be able to invoke both
   `turn_on_lights` and `turn_on_tv` tools in parallel.

## Usage Notes

- You only need to click the **Audio** button once to initiate the
  stream. The current version does not support stopping and restarting the stream
  by clicking the button again during a session.
