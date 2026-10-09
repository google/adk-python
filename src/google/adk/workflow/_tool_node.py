# Copyright 2026 Google LLC
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

from __future__ import annotations

"""A node that wraps an ADK Tool."""

from collections.abc import AsyncGenerator
import json
from typing import Any

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ..agents.context import Context
from ..auth._auth_resume import find_requested_auth_configs
from ..auth.auth_tool import AuthConfig
from ..events.event import Event
from ..events.request_input import RequestInput
from ..platform import uuid as platform_uuid
from ..tools._confirmation_utils import apply_confirmation_gate
from ..tools.base_tool import BaseTool
from ..tools.tool_confirmation import ToolConfirmation
from ..utils.content_utils import extract_text_from_content
from ._base_node import BaseNode
from ._errors import WorkflowDataError
from ._retry_config import RetryConfig
from .utils._workflow_hitl_utils import create_auth_request_event
from .utils._workflow_hitl_utils import process_auth_resume

_TOOL_CONFIRMATION_INTERRUPT_PREFIX = 'wf_tool_confirmation:'
_TOOL_AUTH_INTERRUPT_PREFIX = 'wf_auth:'


def _get_tool_auth_config(tool: BaseTool) -> AuthConfig | None:
  """Returns the AuthConfig attached to a tool or its credential manager."""
  auth_config = getattr(tool, '_auth_config', None)
  if isinstance(auth_config, AuthConfig):
    return auth_config
  credentials_manager = getattr(tool, '_credentials_manager', None)
  cm_auth_config = getattr(credentials_manager, '_auth_config', None)
  if isinstance(cm_auth_config, AuthConfig):
    return cm_auth_config
  return None


def _parse_tool_confirmation(response: Any) -> ToolConfirmation:
  """Parses the user's answer to a tool confirmation request.

  Resume validation against the `ToolConfirmation` schema yields a dict in
  which the fields the user left out are None, so those are dropped to fall
  back to the model defaults.
  """
  if isinstance(response, ToolConfirmation):
    return response
  if not isinstance(response, dict):
    raise WorkflowDataError(
        'A tool confirmation response must be a ToolConfirmation payload, but'
        f' got {type(response).__name__}.'
    )
  try:
    return ToolConfirmation.from_response_dict(
        {k: v for k, v in response.items() if v is not None}
    )
  except ValueError as e:
    raise WorkflowDataError(
        f'Invalid tool confirmation response: {response!r}'
    ) from e


class _AuthPending:
  """Marks a tool call that is waiting for the user to provide credentials."""

  def __init__(self, auth_config: AuthConfig):
    self.auth_config = auth_config


class _ToolNode(BaseNode):
  """A node that wraps an ADK Tool."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  tool: BaseTool = Field(...)

  def __init__(
      self,
      *,
      tool: BaseTool,
      name: str | None = None,
      retry_config: RetryConfig | None = None,
      timeout: float | None = None,
  ):
    super().__init__(
        tool=tool,
        name=name or tool.name,
        # Tool nodes rerun on resume so that calls paused for confirmation execute.
        rerun_on_resume=True,
        retry_config=retry_config,
        timeout=timeout,
    )

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    # Run the tool with the node's own context (ToolContext is Context) so
    # state and artifact deltas recorded by the tool on ctx.actions are emitted
    # with this node.
    ctx.function_call_id = platform_uuid.new_uuid()

    args = node_input
    if isinstance(args, types.Content):
      args = extract_text_from_content(args)

    if isinstance(args, BaseModel):
      args = args.model_dump()
    elif isinstance(args, str):
      args = args.strip()
      if not args:
        args = None
      else:
        try:
          if isinstance(parsed := json.loads(args), dict):
            args = parsed
        except json.JSONDecodeError:
          pass

    declaration = getattr(self.tool, '_get_declaration', lambda: None)()
    schema = getattr(declaration, 'parameters_json_schema', None) or getattr(
        declaration, 'parameters', None
    )
    if isinstance(schema, dict):
      all_params = list(schema.get('properties') or ())
      required_params = list(schema.get('required') or ())
    elif schema is not None:
      all_params = list(getattr(schema, 'properties', None) or ())
      required_params = list(getattr(schema, 'required', None) or ())
    else:
      all_params, required_params = (), ()

    if args is None:
      args = {}
    elif isinstance(args, dict):
      args = dict(args)
    elif len(all_params) == 1:
      args = {all_params[0]: args}
    elif len(required_params) == 1:
      args = {required_params[0]: args}
    else:
      raise TypeError(
          'The input to ToolNode must be a dictionary of tool arguments or'
          f' None, but got {type(args)}.'
      )

    # Fallback to ctx.state for missing required parameters declared in tool declaration
    for param_name in required_params:
      if param_name not in args and param_name in ctx.state:
        args[param_name] = ctx.state[param_name]

    response = await self._run_tool_with_plugin_callbacks(ctx=ctx, args=args)
    if isinstance(response, RequestInput):
      yield response
      return
    if isinstance(response, _AuthPending):
      auth_interrupt_id = f'{_TOOL_AUTH_INTERRUPT_PREFIX}{ctx.node_path}'
      yield create_auth_request_event(
          response.auth_config, auth_interrupt_id, ctx.state
      )
      return

    # State and artifact deltas recorded on ctx.actions by the tool are
    # attached to emitted events by the node runner.
    if response is not None:
      yield Event(output=response)
    else:
      yield Event()

  async def _run_tool_with_plugin_callbacks(
      self, *, ctx: Context, args: dict[str, Any]
  ) -> Any:
    """Runs the tool between the plugin tool callbacks.

    Mirrors the plugin steps of the LlmAgent tool pipeline: a before-tool
    callback may answer the call instead of the tool, an on-tool-error
    callback may answer a failed call, and an after-tool callback may replace
    the result. Agent-level tool callbacks do not apply because no agent owns
    a tool node.

    Like the agent pipeline, the user's answer to a confirmation request is on
    `ctx.tool_confirmation`, and a credential the user supplied is stored in
    state, before any callback runs; the confirmation gate runs after the
    before-tool callback. A call waiting for confirmation
    returns the `RequestInput` to send the user, and a call waiting for
    authentication returns an `_AuthPending`; both skip the after-tool
    callback. A rejected call answers with an error that the after-tool
    callback still sees.
    """
    # Set before the callbacks so they see the answers, and outside the tool
    # error handling so a malformed answer is not reported as a tool failure.
    self._apply_confirmation_resume(ctx=ctx)
    await self._apply_auth_resume(ctx=ctx)
    plugin_manager = ctx.get_invocation_context().plugin_manager
    response = await plugin_manager.run_before_tool_callback(
        tool=self.tool, tool_args=args, tool_context=ctx
    )
    if response is None:
      try:
        response = await apply_confirmation_gate(self.tool, args, ctx)
        if response is None:
          response = await self.tool.run_async(args=args, tool_context=ctx)
        request = self._take_requested_confirmation(ctx=ctx, args=args)
        if request is not None:
          return request
        if (
            ctx.function_call_id
            and ctx.function_call_id in ctx.actions.requested_auth_configs
        ):
          auth_request = ctx.actions.requested_auth_configs.pop(
              ctx.function_call_id
          )
          return _AuthPending(auth_request)
      except Exception as error:
        # The failure is the call's result, so a confirmation or credential
        # the tool asked for before raising is dropped rather than left
        # pending.
        ctx.actions.requested_tool_confirmations.pop(ctx.function_call_id, None)
        ctx.actions.requested_auth_configs.pop(ctx.function_call_id, None)
        response = await plugin_manager.run_on_tool_error_callback(
            tool=self.tool, tool_args=args, tool_context=ctx, error=error
        )
        if response is None:
          raise

    altered_response = await plugin_manager.run_after_tool_callback(
        tool=self.tool, tool_args=args, tool_context=ctx, result=response
    )
    if altered_response is not None:
      response = altered_response
    return response

  def _apply_confirmation_resume(self, *, ctx: Context) -> None:
    """Stores the user's answer to this node's confirmation request.

    The answer is set on `ctx.tool_confirmation`, so the confirmation gate and
    the tool see it the same way they do inside an agent.
    """
    response = ctx.resume_inputs.get(self._confirmation_interrupt_id(ctx))
    if response is not None:
      ctx.tool_confirmation = _parse_tool_confirmation(response)

  def _take_requested_confirmation(
      self, *, ctx: Context, args: dict[str, Any]
  ) -> RequestInput | None:
    """Turns a confirmation requested for this call into a `RequestInput`.

    The request comes from the confirmation gate or from the tool calling
    `tool_context.request_confirmation()`. Once the user answers, the node is
    rerun with `ctx.tool_confirmation` set.
    """
    if not ctx.function_call_id:
      return None
    requested = ctx.actions.requested_tool_confirmations.pop(
        ctx.function_call_id, None
    )
    if requested is None:
      return None
    payload: dict[str, Any] = {'tool_name': self.tool.name, 'args': args}
    if requested.payload is not None:
      payload['confirmation_payload'] = requested.payload
    return RequestInput(
        interrupt_id=self._confirmation_interrupt_id(ctx),
        message=requested.hint
        or f'Please approve or reject the tool call {self.tool.name}().',
        payload=payload,
        response_schema=ToolConfirmation,
    )

  def _confirmation_interrupt_id(self, ctx: Context) -> str:
    """Returns the interrupt id of this node run's confirmation request."""
    return f'{_TOOL_CONFIRMATION_INTERRUPT_PREFIX}{ctx.node_path}'

  async def _apply_auth_resume(self, *, ctx: Context) -> None:
    """Processes a resumed auth credential response before running the tool."""
    interrupt_id = f'{_TOOL_AUTH_INTERRUPT_PREFIX}{ctx.node_path}'
    auth_response = ctx.resume_inputs.get(interrupt_id)
    if auth_response is None:
      return
    auth_config = _get_tool_auth_config(self.tool)
    if auth_config is None:
      requested = find_requested_auth_configs(
          ctx.session.events, [interrupt_id]
      ).get(interrupt_id)
      if requested is not None:
        auth_config = requested.auth_config
    if auth_config is None:
      raise WorkflowDataError(
          f'Cannot resume auth for tool node {ctx.node_path}: no AuthConfig'
          f' found for interrupt {interrupt_id}.'
      )
    await process_auth_resume(
        auth_response, auth_config, ctx.state, interrupt_id
    )
