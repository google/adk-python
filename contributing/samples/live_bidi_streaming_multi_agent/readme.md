# Simplistic Live (Bidi-Streaming) Multi-Agent
This project provides a basic example of a live, bidirectional streaming multi-agent
designed for testing and experimentation.

You can see full documentation [here](https://google.github.io/adk-docs/streaming/).

## Getting Started

Follow these steps to get the agent up and running:

1.  **Start the ADK Web Server**
    Open your terminal, navigate to the root directory that contains the
    `live_bidi_streaming_agent` folder, and execute the following command:
    ```bash
    adk web
    ```

2.  **Access the ADK Web UI**
    Once the server is running, open your web browser and navigate to the URL
    provided in the terminal (it will typically be `http://localhost:8000`).

3.  **Select the Agent**
    In the top-left corner of the ADK Web UI, use the dropdown menu to select
    this agent.

4.  **Start Streaming**
    Click on either the **Audio** or **Video** icon located near the chat input
    box to begin the streaming session.

5.  **Interact with the Agent**
    You can now begin talking to the agent, and it will respond in real-time.

## Usage Notes

* You only need to click the **Audio** or **Video** button once to initiate the
 stream. The current version does not support stopping and restarting the stream
  by clicking the button again during a session.

## Sample Queries

- Hello, what's the weather in Seattle and New York?
- Could you roll a 6-sided dice for me?
- Could you check if the number you rolled is a prime number or not?
