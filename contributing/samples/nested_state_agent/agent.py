import logging

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.utils.instructions_utils import inject_session_state


def inject_nested_state(callback_context: CallbackContext):
    callback_context.state["user"] = {
        # "name": "Jainish",
        # "profile": {"age": 24, "role": "Software Engineer"},
    }
    logging.info("State populated with nested user object.")


async def build_instruction(readonly_context: ReadonlyContext) -> str:
    print(readonly_context.state)
    template = (
        "Current user is {{user?.name?}} and {{user?.profile?.role?}}. Please greet"
        " them by name and designation."
    )
    return await inject_session_state(template, readonly_context)


agent = Agent(
    name="nested_state_agent",
    model="gemini-2.0-flash-lite",
    instruction=build_instruction,
    before_agent_callback=[inject_nested_state],
)

root_agent = agent
