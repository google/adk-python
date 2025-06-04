# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Tuple, Union
import functools
import inspect
import uuid


from ..agents.callback_context import CallbackContext
from google.genai.types import Content, Part, FunctionResponse


def request_answer_factory(
        request_type: str,
        tool: Callable,
        when_reject_callback: Callable = None,
) -> Tuple[
    Callable,
    Callable,
    Callable
]:
    """The util tool to add tool include "human in the loop" system.

    request answer to user or system
    Args:
        request_type: approve | text_content
        tool: actual tool function
        when_reject_callback: (optional)

    Returns:
        - _tool: main tool for Agent
        - _before_agent_callback: handler as before_agent_callback for Agent
        - _after_agent_callback: handler as before_agent_callback for Agent

    Examples
        requesting_tool, before_agent_callback, after_agent_callback = request_answer_factory("approve", your_tool)

        Agent(
            model="your/model-name",
            before_agent_callback=[before_agent_callback]
            after_agent_callback=[after_agent_callback]
            tools=[requesting_tool]
        )

    Note:
        The tool function **MUST** has `tool_context` argument.
         If not needed, use this.
         `_ = tool_context`

    """

    if request_type == "approve":
        return request_approve_factory(tool=tool, when_reject_callback=when_reject_callback)
    elif request_type == "text_content":
        raise ValueError("Not enable yet.")
    else:
        raise ValueError(f"Unexpected `request_type`: {request_type}.")


def request_approve_factory(
        tool: Callable,
        when_reject_callback: Callable = None,
) -> Tuple[
    Callable,
    Callable,
    Callable
]:
    """
    request approval to user.
    Args:
        tool:
        when_reject_callback: (optional)

    Returns:
        - _tool: main tool for Agent
        - _before_agent_callback: handler as before_agent_callback for Agent
        - _after_agent_callback: handler as before_agent_callback for Agent
    """
    tool_name = tool.__name__
    tool_id = f"request:{tool_name}"

    @functools.wraps(tool)
    async def _tool(tool_context: CallbackContext, **kwargs):
        current_status = tool_context.state.get(tool_id, {}).get("status", None)
        if current_status is None or current_status == "requesting":
            tool_context.state[tool_id] = {
                "status": "requesting",
                "args": kwargs
            }
            tool_context.state["temp:user_answer_request_status"] = "requesting"
            return {
                "status": "accept_required",
                "request_user_action": True,
                "message": "accept request to user."
            }
        else:
            return {
                "status": "error",
                "message": "tool running error."
            }

    async def _execute_tool(callback_context, **tool_calling_args):
        calling_id = str(uuid.uuid4())
        response = tool(**tool_calling_args, tool_context=callback_context)
        if inspect.isawaitable(response):
            response = await response

        callback_context.state[tool_id]["status"] = None
        return Content(
            role="model",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        id=calling_id,
                        name=tool_name,
                        response=response
                    )
                )
            ]
        )

    async def _reject_tool(callback_context, **tool_calling_args):
        if when_reject_callback:
            response = when_reject_callback(**tool_calling_args, tool_context=callback_context)
            if inspect.isawaitable(response):
                response = await response

            calling_id = str(uuid.uuid4())
            callback_context.state[tool_id]["status"] = None
            return Content(
                role="model",
                parts=[
                    Part(
                        function_response=FunctionResponse(
                            id=calling_id,
                            name=tool_name,
                            response=response
                        )
                    )
                ]
            )

        else:  # default reject message
            callback_context.state[tool_id]["status"] = None
            return Content(
                role="model",
                parts=[Part(text="tool_calling request rejected.")]
            )

    async def _before_agent_callback(callback_context: CallbackContext) -> Union[None, Content]:
        current_status = callback_context.state.get(tool_id, {}).get("status")
        if current_status is None:
            return

        tool_calling_args = callback_context.state.get(tool_id, {}).get("args", {})
        if current_status == "requesting":

            # accept or reject from user directory.
            last_event = callback_context.events[-1]
            if last_event.author == "user" and last_event.content.parts[0].text == "accept":
                return await _execute_tool(callback_context=callback_context, **tool_calling_args)

            elif last_event.author == "user" and last_event.content.parts[0].text == "reject":
                return await _execute_tool(callback_context=callback_context, **tool_calling_args)

            else:
                return

        # accept or reject from foreign system.
        elif current_status == "accept":
            return await _execute_tool(callback_context=callback_context, **tool_calling_args)

        elif current_status == "reject":
            return await _reject_tool(callback_context=callback_context, **tool_calling_args)
        else:
            raise ValueError(f"unexpected request stats: {tool_id}: {current_status}")

    async def _after_agent_callback(callback_context: CallbackContext) -> Union[None, Content]:
        if callback_context.state.get("temp:user_answer_request_status") == "requesting":
            callback_context.state["temp:user_answer_request_status"] = None
            return Content(
                    role="model",
                    parts=[
                        Part(
                            text=(
                                f"tool: `{tool_name}` required user approval.\n"
                                f"If you accept this request, send `accept` as text.\n"
                                f"If you reject this request, send `reject` as text."
                            )
                        )
                    ]
                )
        else:
            callback_context.state["temp:user_answer_request_status"] = None
            return

    return _tool, _before_agent_callback, _after_agent_callback
