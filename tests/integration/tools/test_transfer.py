import pytest
from google.adk.tools import transfer_to_agent

class DummyActions:
    def __init__(self):
        self.transfer_to_agent = None

class DummyToolContext:
    def __init__(self):
        self.actions = DummyActions()

def test_transfer_to_agent_sets_correct_agent():
    ctx = DummyToolContext()
    transfer_to_agent(agent_name="math_agent", tool_context=ctx)
    assert ctx.actions.transfer_to_agent == "math_agent"

def test_transfer_to_agent_ignores_single_extra_kwarg():
    ctx = DummyToolContext()
    transfer_to_agent(agent_name="history_agent", tool_context=ctx, query="When was WWII?")
    assert ctx.actions.transfer_to_agent == "history_agent"

def test_transfer_to_agent_ignores_multiple_extra_kwargs():
    ctx = DummyToolContext()
    transfer_to_agent(
        agent_name="code_agent",
        tool_context=ctx,
        query="foo",
        temperature=0.5,
        max_tokens=100
    )
    assert ctx.actions.transfer_to_agent == "code_agent"
