# AntigravityAgent in an ADK Workflow

## Overview

This sample runs an ADK `LlmAgent` and an
[Antigravity SDK](https://pypi.org/project/google-antigravity/) agent
(`AntigravityAgent`) as **nodes in a `Workflow`**:

1. **`planner`** -- an ADK `LlmAgent` node that turns the user's request into a
   short, concrete build spec.
1. **`builder`** -- an `AntigravityAgent` node that receives that spec as its
   input and implements it by writing files into a scratch `workspace/`
   directory.

A `Workflow` wires nodes through edges (each node's output becomes the next
node's input), not through ADK `sub_agents`. So the `AntigravityAgent` node is
never adopted by an ADK parent, and does not need `mode='single_turn'` for that
reason. It sets `mode='single_turn'` anyway because a workflow node is invoked
once per run with a freshly composed input -- exactly single-turn semantics --
and it keeps the harness from minting a throwaway save_dir per run. See the
[AntigravityAgent guide](../../../../docs/guides/labs/antigravity/index.md)
for the full setup, limitations, and API details.

## Prerequisites

- Install the Antigravity SDK: `pip install "google-adk[antigravity]"`
- Set a Gemini API key: `export GEMINI_API_KEY="your-api-key"`
  (required by the Antigravity SDK, which drives the builder node's model)

The `builder` node writes files into a `workspace/` directory next to
`agent.py`, created automatically on import.

## Sample Inputs

- `A Python module with a function that checks whether a string is a palindrome.`

  `planner` writes a spec (e.g. a `palindrome.py` file with an
  `is_palindrome(s)` function that ignores case and non-alphanumeric
  characters), and `builder` implements it in `workspace/`.

- `A command-line number-guessing game in Python.`

  `planner` produces a spec, and `builder` writes the game file(s) into
  `workspace/`.

## Graph

```text
     [ START ]
         |
         v
    [ planner ]   (ADK LlmAgent node -> emits a build spec)
         |
         | output threaded in as node input
         v
    [ builder ]   (AntigravityAgent node, mode='single_turn')
         |
         v
   [ Antigravity SDK local harness -> workspace/ ]
```

## How To

Any `BaseAgent` -- including an `AntigravityAgent` -- can be a `Workflow` node.
Chain nodes in a single edge tuple; the workflow threads each node's output in
as the next node's input:

```python
planner = Agent(name="planner", model="gemini-3.8-flash", instruction="...")

builder = AntigravityAgent(
    name="builder",
    description="Implements a build spec by writing files into the workspace.",
    config=_sdk_config,
    mode="single_turn",
)

root_agent = Workflow(
    name="antigravity_workflow",
    edges=[("START", planner, builder)],
)
```
