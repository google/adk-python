# Workflow Triage Sample

This sample demonstrates how to build a multi-agent workflow that intelligently triages incoming requests and delegates them to appropriate specialized agents.

## Overview

The workflow consists of three main components:

1. **Execution Manager Agent** (`agent.py`) - Analyzes user input and determines which execution agents are relevant
2. **Plan Execution Agent** - Sequential agent that coordinates execution and summarization
3. **Worker Execution Agents** (`execution_agent.py`) - Specialized agents that execute specific tasks in parallel

## Architecture

### Execution Manager Agent (`root_agent`)
- **Model**: gemini-2.5-flash
- **Name**: `execution_manager_agent`
- **Role**: Analyzes user requests and updates the execution plan
- **Tools**: `update_execution_plan` - Updates which execution agents should be activated
- **Sub-agents**: Delegates to `plan_execution_agent` for actual task execution
- **Clarification**: Asks for clarification if user intent is unclear before proceeding

### Plan Execution Agent
- **Type**: SequentialAgent
- **Name**: `plan_execution_agent`
- **Components**:
  - `worker_parallel_agent` (ParallelAgent) - Runs relevant agents in parallel
  - `execution_summary_agent` - Summarizes the execution results

### Worker Agents
The system includes two specialized execution agents that run in parallel:

- **Code Agent** (`code_agent`): Handles code generation tasks
  - Uses `before_agent_callback_check_relevance` to skip if not relevant
  - Output stored in `code_agent_output` state key
- **Math Agent** (`math_agent`): Performs mathematical calculations
  - Uses `before_agent_callback_check_relevance` to skip if not relevant
  - Output stored in `math_agent_output` state key

### Execution Summary Agent
- **Model**: gemini-2.5-flash
- **Name**: `execution_summary_agent`
- **Role**: Summarizes outputs from all activated agents
- **Dynamic Instructions**: Generated based on which agents were activated
- **Content Inclusion**: Set to "none" to focus on summarization

## Key Features

- **Dynamic Agent Selection**: Automatically determines which agents are needed based on user input
- **Parallel Execution**: Multiple relevant agents can work simultaneously via `ParallelAgent`
- **Relevance Filtering**: Agents skip execution if they're not relevant to the current state using callback mechanism
- **Stateful Workflow**: Maintains execution state through `ToolContext`
- **Execution Summarization**: Automatically summarizes results from all activated agents
- **Sequential Coordination**: Uses `SequentialAgent` to ensure proper execution flow

## Usage

The workflow follows this pattern:

1. User provides input to the root agent (`execution_manager_agent`)
2. Manager analyzes the request and identifies relevant agents (`code_agent`, `math_agent`)
3. If user intent is unclear, manager asks for clarification before proceeding
4. Manager updates the execution plan using `update_execution_plan`
5. Control transfers to `plan_execution_agent`
6. `worker_parallel_agent` (ParallelAgent) runs only relevant agents based on the updated plan
7. `execution_summary_agent` summarizes the results from all activated agents

### Example Queries

**Vague requests requiring clarification:**

```
> hi
> Help me do this.
```

The root agent (`execution_manager_agent`) will greet the user and ask for clarification about their specific task.

**Math-only requests:**

```
> What's 1+1?
```

Only the `math_agent` executes while `code_agent` is skipped.

**Multi-domain requests:**

```
> What's 1+11? Write a python function to verify it.
```

Both `code_agent` and `math_agent` execute in parallel, followed by summarization.

## Available Execution Agents

- `code_agent` - For code generation and programming tasks
- `math_agent` - For mathematical computations and analysis

## Implementation Details

- Uses Google ADK agents framework
- Implements callback-based relevance checking via `before_agent_callback_check_relevance`
- Maintains state through `ToolContext` and state keys
- Supports parallel agent execution with `ParallelAgent`
- Uses `SequentialAgent` for coordinated execution flow
- Dynamic instruction generation for summary agent based on activated agents
- Agent outputs stored in state with `{agent_name}_output` keys
