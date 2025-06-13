import logging
import warnings
import os
from typing import Optional, Dict, Any, AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.events.event_actions import EventActions
from tests.unittests import testing_utils # Changed import
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
import asyncio

warnings.filterwarnings("ignore", category=UserWarning, module=".*pydantic.*")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

captured_log_after_tool_callback_response = None
captured_sub_agent_after_tool_callback_response = None
sub_agent_tool_called_with_args = None

def static_response(number: Optional[str] = None) -> dict:
    global sub_agent_tool_called_with_args
    sub_agent_tool_called_with_args = {'number': number}
    logger.info(f"static_response called with number: {number}")
    return {"status": "success", "result": f"{number}"}

def sub_agent_after_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: dict) -> Optional[dict]:
    global captured_sub_agent_after_tool_callback_response
    captured_sub_agent_after_tool_callback_response = tool_response
    logger.info(f"Sub-agent after_tool_callback called. Original tool_response: {tool_response}")
    tool_context.actions.skip_summarization = True
    logger.info("Sub-agent after_tool_callback: tool_context.actions.skip_summarization = True")
    return None

def root_agent_log_after_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: Any) -> Optional[dict]:
    global captured_log_after_tool_callback_response
    captured_log_after_tool_callback_response = tool_response
    logger.info(f"Root agent log_after_tool_callback called. Tool response received: {tool_response}")
    return None

mock_sub_agent_llm = testing_utils.MockModel.create(
    responses=[
        types.Part.from_function_call(name='static_response', args={'number': '5'}),
    ]
)

static_response_agent = LlmAgent(
    model=mock_sub_agent_llm,
    after_tool_callback=sub_agent_after_tool_callback,
    name='agent_agent',
    instruction="Sub-agent instruction: use static_response.",
    tools=[static_response],
)

mock_root_llm = testing_utils.MockModel.create(
    responses=[
        types.Part.from_function_call(name='agent_agent', args={'request': 'Call static_response with number 5'}),
        "Final response from root agent after processing tool output.",
    ]
)

root_agent = LlmAgent(
    model=mock_root_llm,
    after_tool_callback=root_agent_log_after_tool_callback,
    name='root_agent',
    instruction="Root agent instruction: use agent_agent tool.",
    tools=[AgentTool(agent=static_response_agent)],
)

async def main():
    logger.info("Starting verification script for issue 1103.")

    runner = Runner(
        app_name="issue_1103_verifier",
        agent=root_agent,
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService()
    )
    session = await runner.session_service.create_session(app_name="issue_1103_verifier", user_id="test_user_1103")

    user_input_text = "Please use the sub agent to get static response for number 5."
    input_content = types.Content(role="user", parts=[types.Part.from_text(text=user_input_text)])

    logger.info(f"Running root_agent with input: '{user_input_text}'")

    async for event in runner.run_async(user_id=session.user_id, session_id=session.id, new_message=input_content):
        logger.info(f"Event from runner: Author: {event.author}, Content: {event.content}, Actions: {event.actions}")

    logger.info("--- Verification Checks ---")

    expected_raw_output = {"status": "success", "result": "5"}

    logger.info(f"Sub-agent's static_response tool was called with: {sub_agent_tool_called_with_args}")
    assert sub_agent_tool_called_with_args == {'number': '5'}, "Sub-agent's tool not called with correct args."

    logger.info(f"Sub-agent's after_tool_callback received: {captured_sub_agent_after_tool_callback_response}")
    assert captured_sub_agent_after_tool_callback_response == expected_raw_output, \
        "Sub-agent's after_tool_callback did not receive the direct tool output."

    logger.info(f"Root agent's log_after_tool_callback received: {captured_log_after_tool_callback_response}")
    assert captured_log_after_tool_callback_response == expected_raw_output, \
        f"Root agent did not receive the raw JSON. Expected {expected_raw_output}, got {captured_log_after_tool_callback_response}"

    num_sub_agent_llm_calls = len(mock_sub_agent_llm.requests)
    logger.info(f"Number of calls to sub-agent's LLM: {num_sub_agent_llm_calls}")
    assert num_sub_agent_llm_calls == 1, \
        "Sub-agent's LLM was called more than once, indicating it might have tried to summarize."

    logger.info("Verification successful: Root agent received raw JSON from sub-agent's tool.")

if __name__ == "__main__":
    asyncio.run(main())
