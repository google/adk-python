# Sequential Agents in ADK

This example demonstrates how to implement a Sequential Agent in the Agent Development Kit (ADK). The main agent in this example, `lead_qualification_agent`, is a Sequential Agent that executes sub-agents in a predefined order, with each agent's output feeding into the next agent in the sequence.

## What are Sequential Agents?

Sequential Agents are workflow agents in ADK that:

1. **Execute in a Fixed Order**: Sub-agents run one after another in the exact sequence they are specified
2. **Pass Data Between Agents**: Using state management to pass information from one sub-agent to the next
3. **Create Processing Pipelines**: Perfect for scenarios where each step depends on the previous step's output

Use Sequential Agents when you need a deterministic, step-by-step workflow where the execution order matters.

## Lead Qualification Pipeline Example

In this example, we've created `lead_qualification_agent` as a Sequential Agent that implements a lead qualification pipeline for sales teams. This Sequential Agent orchestrates three specialized sub-agents:

1. **Lead Validator Agent**: Checks if the lead information is complete enough for qualification
   - Validates for required information like contact details and interest
   - Outputs a simple "valid" or "invalid" with a reason

2. **Lead Scorer Agent**: Scores valid leads on a scale of 1-10
   - Analyzes factors like urgency, decision-making authority, budget, and timeline
   - Provides a numeric score with a brief justification

3. **Action Recommender Agent**: Suggests next steps based on the validation and score
   - For invalid leads: Recommends what information to gather
   - For low-scoring leads (1-3): Suggests nurturing actions
   - For medium-scoring leads (4-7): Suggests qualifying actions
   - For high-scoring leads (8-10): Suggests sales actions

### How It Works

The `lead_qualification_agent` Sequential Agent orchestrates this process by:

1. Running the Validator first to determine if the lead is complete
2. Running the Scorer next (which can access validation results via state)
3. Running the Recommender last (which can access both validation and scoring results)

The output of each sub-agent is stored in the session state using the `output_key` parameter:
- `validation_status`
- `lead_score`
- `action_recommendation`

## Project Structure

```
9-sequential-agent/
│
├── lead_qualification_agent/       # Main Sequential Agent package
│   ├── __init__.py                 # Package initialization
│   ├── agent.py                    # Sequential Agent definition (root_agent)
│   │
│   └── subagents/                  # Sub-agents folder
│       ├── __init__.py             # Sub-agents initialization
│       │
│       ├── validator/              # Lead validation agent
│       │   ├── __init__.py
│       │   └── agent.py
│       │
│       ├── scorer/                 # Lead scoring agent
│       │   ├── __init__.py
│       │   └── agent.py
│       │
│       └── recommender/            # Action recommendation agent
│           ├── __init__.py
│           └── agent.py
│
├── .env.example                    # Environment variables example
└── README.md                       # This documentation
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
cd 9-sequential-agent
adk web
```

Then select "lead_qualification_agent" from the dropdown menu in the web UI.

## Example Interactions

Try these example interactions:

### Qualified Lead Example:
```
Lead Information:
Name: Sarah Johnson
Email: sarah.j@techinnovate.com
Phone: 555-123-4567
Company: Tech Innovate Solutions
Position: CTO
Interest: Looking for an AI solution to automate customer support
Budget: $50K-100K available for the right solution
Timeline: Hoping to implement within next quarter
Notes: Currently using a competitor's product but unhappy with performance
```

### Unqualified Lead Example:
```
Lead Information:
Name: John Doe
Email: john@gmail.com
Interest: Something with AI maybe
Notes: Met at conference, seemed interested but was vague about needs
```

## How Sequential Agents Compare to Other Workflow Agents

ADK offers different types of workflow agents for different needs:

- **Sequential Agents**: For strict, ordered execution (like this example)
- **Loop Agents**: For repeated execution of sub-agents based on conditions
- **Parallel Agents**: For concurrent execution of independent sub-agents

## Additional Resources

- [ADK Sequential Agents Documentation](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/)
- [Full Code Development Pipeline Example](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/#full-example-code-development-pipeline) 
