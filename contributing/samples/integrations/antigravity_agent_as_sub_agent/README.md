# AntigravityAgent as an ADK Sub-Agent

## Overview

This sample puts an ADK `LlmAgent` at the root and an
[Antigravity SDK](https://pypi.org/project/google-antigravity/) agent
(`AntigravityAgent`) underneath it as a **sub-agent**. The root `coordinator`
answers general questions itself and delegates any hands-on coding or
file-writing work to the `code_writer` sub-agent, which runs in a scratch
`workspace/` directory.

The sub-agent sets `mode='single_turn'`. That is what allows an
`AntigravityAgent` to have an ADK parent at all: the parent `LlmAgent` exposes it
as an inline tool taking a `request` string, composes each request itself, and
does not forward session history. Each delegated call is an independent
Antigravity conversation. See the
[AntigravityAgent guide](../../../../docs/guides/labs/antigravity/index.md)
for the full setup, limitations, and API details.

## Prerequisites

- Install the Antigravity SDK: `pip install "google-adk[antigravity]"`
- Set a Gemini API key: `export GEMINI_API_KEY="your-api-key"`
  (required by the Antigravity SDK, which drives the sub-agent's model)

The `code_writer` sub-agent writes generated files into a `workspace/` directory
next to `agent.py`, created automatically on import.

## Sample Inputs

- `Who wrote the Python programming language?`

  The `coordinator` answers this itself, without delegating.

- `Write a Python function that returns the nth Fibonacci number, and save it to fib.py.`

  The `coordinator` composes a self-contained coding task and delegates it to
  `code_writer`, which writes `workspace/fib.py` using the Antigravity SDK's
  built-in file tools, then reports back. The `coordinator` summarizes the result.

- `Add a docstring and a couple of doctests to fib.py.`

  The `coordinator` delegates another self-contained edit to `code_writer`.
  Because each delegated call is an independent conversation, the request must
  describe the file to edit; it is not carried over from the previous turn.

## Graph

```text
                [ user ]
                    |
                    v
            [ coordinator ]  (ADK LlmAgent, root)
                    |  single_turn delegation (inline tool)
                    v
            [ code_writer ]  (AntigravityAgent, mode='single_turn')
                    |
                    v
        [ Antigravity SDK local harness -> workspace/ ]
```

## How To

Give an `AntigravityAgent` an ADK parent by setting `mode='single_turn'` and
listing it in the parent's `sub_agents`:

```python
code_writer = AntigravityAgent(
    name="code_writer",
    description="Writes and edits real code files in a scratch workspace...",
    config=_sdk_config,
    mode="single_turn",
)

root_agent = Agent(
    name="coordinator",
    model="gemini-3.8-flash",
    instruction="... delegate coding tasks to code_writer ...",
    sub_agents=[code_writer],
)
```

Because `code_writer` sets `mode='single_turn'`, the root `LlmAgent` wraps it as
an inline tool (rather than an LLM-transfer target). The `description` is what
the coordinator's model reads when deciding whether to delegate, so it must be
non-empty and specific.
