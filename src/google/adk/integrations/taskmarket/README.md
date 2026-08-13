# Taskmarket requester tools

`TaskMarketToolset` gives an ADK agent a requester workflow for Taskmarket:

1. `list_tasks`, `get_task`, and `list_submissions` use the public API.
2. `preview_task` returns the exact description, reward, deadline, Base chain,
   USDC contract, and conservative maximum spend.
3. `create_task` is exposed with ADK's confirmation gate and also requires
   `confirm=True` plus the unchanged preview token. It checks the wallet's
   network and available USDC before invoking the first-party CLI once.

The integration does not accept wallet keys or tokens. Install the official
CLI separately and initialize its own keystore:

```bash
npm install -g @lucid-agents/taskmarket
taskmarket init
```

Example:

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.taskmarket import TaskMarketToolset

agent = LlmAgent(
    model="gemini-2.5-flash",
    name="requester_agent",
    instruction=(
        "Use preview_task first. Show its complete output to the user and "
        "only create a task after explicit confirmation. Review submissions "
        "with a human; never accept or reject them automatically."
    ),
    tools=[TaskMarketToolset()],
)
```

The CLI is deliberately called without shell interpolation. A non-zero or
timed-out create command is reported as non-retryable and potentially
ambiguous; callers must inspect live status before taking any further action.
Run the focused tests with:

```bash
pytest tests/unittests/integrations/taskmarket/test_taskmarket_toolset.py
```
