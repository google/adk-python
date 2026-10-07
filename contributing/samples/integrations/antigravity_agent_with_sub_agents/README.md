# AntigravityAgent with an ADK Sub-Agent

## Overview

This sample puts an
[Antigravity SDK](https://pypi.org/project/google-antigravity/) agent
(`AntigravityAgent`) at the root and gives it an ADK `LlmAgent` **sub-agent**.
The root `local_assistant` works in a scratch `workspace/` directory. The
`weather_reporter` sub-agent owns a plain function tool and answers weather
questions.

Each ADK child of an `AntigravityAgent` is bridged onto the Antigravity SDK
config as a **client-side tool** named after the child, which is the only way
the harness can reach an ADK agent. So the root consults `weather_reporter` for
a weather fact and then writes it to a file with its own built-in file tools.

A bridged child runs in isolation (its own `Runner` and session) and returns
only its final text, so every child needs a non-empty `description` -- that is
what the Antigravity model reads when deciding whether to call it. See the
[AntigravityAgent guide](../../../../docs/guides/labs/antigravity/index.md)
for the full setup, limitations, and API details.

## Prerequisites

- Install the Antigravity SDK: `pip install "google-adk[antigravity]"`
- Set a Gemini API key: `export GEMINI_API_KEY="your-api-key"`
  (required by the Antigravity SDK, which drives the root agent's model)

The root writes files into a `workspace/` directory and points the Antigravity
SDK's `save_dir` at a `trajectories/` directory. Both sit next to `agent.py` and
are created automatically on import.

## Sample Inputs

- `What is the weather in San Francisco?`

  The root calls the bridged `weather_reporter` tool and relays its answer.

- `Look up the weather in San Francisco and save a one-line summary to weather.txt.`

  The root calls `weather_reporter`, then writes `workspace/weather.txt` with its
  built-in `create_file`/`edit_file` tools and confirms the path.

## Graph

```text
              [ user ]
                  |
                  v
        [ local_assistant ]  (AntigravityAgent, ADK root)
           |                    \
           | built-in            \ client-side tool bridge
           | file tools           v
           v               [ weather_reporter ]  (ADK LlmAgent sub-agent)
       workspace/                  |
                                   v
                            [ get_weather() ]
```

## How To

Give an `AntigravityAgent` ADK `sub_agents`. Each child is bridged to the
Antigravity harness as a client-side tool:

```python
weather_reporter = Agent(
    name="weather_reporter",
    model="gemini-3.8-flash",
    description="Reports the current weather for a city...",
    instruction="Answer weather questions by calling get_weather...",
    tools=[get_weather],
)

root_agent = AntigravityAgent(
    name="local_assistant",
    description="...",
    config=_sdk_config,
    sub_agents=[weather_reporter],
)
```

The child's `name` becomes the tool name the Antigravity model calls, and its
`description` becomes the tool description. The parent session records the tool
call and a `function_response` carrying the child's final text.
