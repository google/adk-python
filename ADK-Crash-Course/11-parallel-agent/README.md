# Parallel Agents in ADK

This example demonstrates how to implement a Parallel Agent in the Agent Development Kit (ADK). The main agent in this example, `system_monitor_agent`, uses a Parallel Agent to gather system information concurrently and then synthesizes it into a comprehensive system health report.

## What are Parallel Agents?

Parallel Agents are workflow agents in ADK that:

1. **Execute Concurrently**: Sub-agents run simultaneously rather than sequentially
2. **Operate Independently**: Each sub-agent works independently without sharing state during execution
3. **Improve Performance**: Dramatically speed up workflows where tasks can be performed in parallel

Use Parallel Agents when you need to execute multiple independent tasks efficiently and time is a critical factor.

## System Monitoring Example

In this example, we've created a system monitoring application that uses a Parallel Agent to gather system information. The workflow consists of:

1. **Parallel System Information Gathering**: Using a `ParallelAgent` to concurrently collect data about:
   - CPU usage and statistics
   - Memory utilization
   - Disk space and usage

2. **Sequential Report Synthesis**: After parallel data collection, a synthesizer agent combines all information into a comprehensive report

### Sub-Agents

1. **CPU Info Agent**: Collects and analyzes CPU information
   - Retrieves core counts, usage statistics, and performance metrics
   - Identifies potential performance issues (high CPU usage)

2. **Memory Info Agent**: Gathers memory usage information
   - Collects total, used, and available memory
   - Analyzes memory pressure and swap usage

3. **Disk Info Agent**: Analyzes disk space and usage
   - Reports on total, used, and free disk space
   - Identifies disks that are running low on space

4. **System Report Synthesizer**: Combines all gathered information into a comprehensive system health report
   - Creates an executive summary of system health
   - Organizes component-specific information into sections
   - Provides recommendations based on system metrics

### How It Works

The architecture combines both parallel and sequential workflow patterns:

1. First, the `system_info_gatherer` Parallel Agent runs all three information agents concurrently
2. Then, the `system_report_synthesizer` uses the collected data to generate a final report

This hybrid approach demonstrates how to combine workflow agent types for optimal performance and logical flow.

## Project Structure

```
10-parallel-agent/
│
├── system_monitor_agent/          # Main System Monitor Agent package
│   ├── __init__.py                # Package initialization
│   ├── agent.py                   # Agent definitions (root_agent)
│   │
│   └── subagents/                 # Sub-agents folder
│       ├── __init__.py            # Sub-agents initialization
│       │
│       ├── cpu_info_agent/        # CPU information agent
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   └── tools.py           # CPU info collection tools
│       │
│       ├── memory_info_agent/     # Memory information agent
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   └── tools.py           # Memory info collection tools
│       │
│       ├── disk_info_agent/       # Disk information agent
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   └── tools.py           # Disk info collection tools
│       │
│       └── synthesizer_agent/     # Report synthesizing agent
│           ├── __init__.py
│           └── agent.py
│
├── .env.example                   # Environment variables example
└── README.md                      # This documentation
```

## Getting Started

### Setup

1. Activate the virtual environment from the root directory:
```bash
# macOS/Linux:
source ../.venv/bin/activate
# Windows CMD:
..\.venv\Scripts\activate.bat
# Windows PowerShell:
..\.venv\Scripts\Activate.ps1
```

2. Copy the `.env.example` file to `.env` and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### Running the Example

```bash
cd 10-parallel-agent
adk web
```

Then select "system_monitor_agent" from the dropdown menu in the web UI.

## Example Interactions

Try these example prompts:

```
Check my system health
```

```
Provide a comprehensive system report with recommendations
```

```
Is my system running out of memory or disk space?
```

## Key Concepts: Independent Execution

One key aspect of Parallel Agents is that **sub-agents run independently without sharing state during execution**. In this example:

1. Each information gathering agent operates in isolation
2. The results from each agent are collected after parallel execution completes
3. The synthesizer agent then uses these collected results to create the final report

This approach is ideal for scenarios where tasks are completely independent and don't require interaction during execution.

## How Parallel Agents Compare to Other Workflow Agents

ADK offers different types of workflow agents for different needs:

- **Sequential Agents**: For strict, ordered execution where each step depends on previous outputs
- **Loop Agents**: For repeated execution of sub-agents based on conditions
- **Parallel Agents**: For concurrent execution of independent sub-agents (like this example)

## Additional Resources

- [ADK Parallel Agents Documentation](https://google.github.io/adk-docs/agents/workflow-agents/parallel-agents/)
- [Full Example: Parallel Web Research](https://google.github.io/adk-docs/agents/workflow-agents/parallel-agents/#full-example-parallel-web-research) 
