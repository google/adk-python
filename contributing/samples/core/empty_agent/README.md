# ADK Agent Empty Sample

## Overview

This sample demonstrates how to create a minimal, empty agent using the **ADK** framework.

It defines a simple `Agent` that doesn't have any specific tools or complex instructions attached to it. This is useful as a basic template or starting point for defining more complex agentic behaviors over time, ensuring it follows the core directory requirements of `adk`.

## Sample Inputs

- `go`

## Graph

```mermaid
graph TD
    empty_agent[Agent: empty_agent]
```

## How To

1. Define a basic agent using the `Agent` class, specifying a unique name:

   ```python
   from google.adk.agents import Agent

   root_agent = Agent(
       name="empty_agent",
   )
   ```
