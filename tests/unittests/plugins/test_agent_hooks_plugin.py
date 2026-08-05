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

"""Tests for AgentHooksPlugin, the agent-hooks governance host.

The optional ``agent_hooks`` package (with its compiled native core) is
required; the whole module is skipped when it is not importable.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

agent_hooks = pytest.importorskip("agent_hooks")

from google.adk.plugins import AgentHooksPlugin  # noqa: E402

AgentContext = agent_hooks.AgentContext
Decision = agent_hooks.Decision
Transform = agent_hooks.Transform
Verdict = agent_hooks.Verdict


# ---------------------------------------------------------------------------
# Interceptors (real agent-hooks verdicts)
# ---------------------------------------------------------------------------


class _AllowAll:

  def intercept(self, ctx: AgentContext) -> Any:
    return Verdict(decision=Decision.ALLOW)


class _DenyTool:

  def __init__(self, tool_name: str) -> None:
    self._tool_name = tool_name

  def intercept(self, ctx: AgentContext) -> Any:
    if (
        ctx["interception_point"] == "pre_tool_call"
        and ctx["tool_call"]["name"] == self._tool_name
    ):
      return Verdict.deny(reason="tool_denied", message="not allowed")
    return Verdict(decision=Decision.ALLOW)


class _TransformToolArgs:

  def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "pre_tool_call":
      new_args = dict(ctx["tool_call"]["args"])
      new_args["redacted"] = True
      return Verdict(
          decision=Decision.TRANSFORM,
          transform=Transform(path="$target", value=new_args),
      )
    return Verdict(decision=Decision.ALLOW)


class _TransformToolResult:

  def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "post_tool_call":
      return Verdict(
          decision=Decision.TRANSFORM,
          transform=Transform(path="$target", value={"clean": "value"}),
      )
    return Verdict(decision=Decision.ALLOW)


class _DenyModel:

  def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "pre_model_call":
      return Verdict.deny(reason="model_denied")
    return Verdict(decision=Decision.ALLOW)


class _TransformModel:

  def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "pre_model_call":
      return Verdict(
          decision=Decision.TRANSFORM,
          transform=Transform(
              path="$target", value=[{"role": "user", "content": "x"}]
          ),
      )
    if ctx["interception_point"] == "post_model_call":
      return Verdict(
          decision=Decision.TRANSFORM,
          transform=Transform(path="$target.content", value="SAFE"),
      )
    return Verdict(decision=Decision.ALLOW)


class _DenyInput:

  def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "input":
      return Verdict.deny(reason="input_denied")
    return Verdict(decision=Decision.ALLOW)


class _TransformInput:

  def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "input":
      return Verdict(
          decision=Decision.TRANSFORM,
          transform=Transform(path="$target.content", value="CLEANED"),
      )
    return Verdict(decision=Decision.ALLOW)


class _DenyOutput:

  def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "output":
      return Verdict.deny(reason="output_denied")
    return Verdict(decision=Decision.ALLOW)


class _Raiser:

  def intercept(self, ctx: AgentContext) -> Any:
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Context factories (the plugin only reads a few attributes)
# ---------------------------------------------------------------------------


def _invocation_context(
    *,
    invocation_id: str = "inv-1",
    session_id: str = "sess-1",
    agent_name: str = "agent",
    tools: list[Any] | None = None,
) -> Any:
  agent = Mock()
  agent.name = agent_name
  agent.tools = tools or []
  session = Mock()
  session.id = session_id
  ic = Mock()
  ic.invocation_id = invocation_id
  ic.agent = agent
  ic.session = session
  return ic


def _callback_context(
    *,
    invocation_id: str = "inv-1",
    session_id: str = "sess-1",
    agent_name: str = "agent",
) -> Any:
  session = Mock()
  session.id = session_id
  ctx = Mock()
  ctx.invocation_id = invocation_id
  ctx.session = session
  ctx.agent_name = agent_name
  return ctx


def _tool_context(
    *,
    invocation_id: str = "inv-1",
    session_id: str = "sess-1",
    agent_name: str = "agent",
    function_call_id: str | None = "fc-1",
) -> Any:
  ctx = _callback_context(
      invocation_id=invocation_id,
      session_id=session_id,
      agent_name=agent_name,
  )
  ctx.function_call_id = function_call_id
  return ctx


def _tool(name: str = "delete_account") -> Any:
  tool = Mock()
  tool.name = name
  return tool


def _final_event(text: str = "answer", author: str = "agent") -> Event:
  return Event(
      invocation_id="inv-1",
      author=author,
      content=types.Content(
          role="model", parts=[types.Part.from_text(text=text)]
      ),
  )


# ---------------------------------------------------------------------------
# Tool point
# ---------------------------------------------------------------------------


async def test_before_tool_allow_returns_none() -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  result = await plugin.before_tool_callback(
      tool=_tool("safe"),
      tool_args={"x": 1},
      tool_context=_tool_context(),
  )
  assert result is None


async def test_before_tool_deny_blocks_with_error() -> None:
  plugin = AgentHooksPlugin(interceptors=[_DenyTool("delete_account")])
  args = {"user_id": 42}
  result = await plugin.before_tool_callback(
      tool=_tool("delete_account"),
      tool_args=args,
      tool_context=_tool_context(),
  )
  assert result is not None
  assert result["agent_hooks_blocked"] is True
  assert "tool_denied" in result["reason"]
  assert "error" in result
  # The original args are untouched on a deny.
  assert args == {"user_id": 42}


async def test_before_tool_transform_mutates_args_in_place() -> None:
  plugin = AgentHooksPlugin(interceptors=[_TransformToolArgs()])
  args = {"user_id": 42}
  result = await plugin.before_tool_callback(
      tool=_tool("lookup"),
      tool_args=args,
      tool_context=_tool_context(),
  )
  # Transform proceeds (returns None) but rewrites args in place so the
  # real tool call sees the transformed values.
  assert result is None
  assert args == {"user_id": 42, "redacted": True}


async def test_after_tool_transform_replaces_result() -> None:
  plugin = AgentHooksPlugin(interceptors=[_TransformToolResult()])
  result = await plugin.after_tool_callback(
      tool=_tool("lookup"),
      tool_args={"user_id": 42},
      tool_context=_tool_context(),
      result={"secret": "xyz"},
  )
  assert result == {"clean": "value"}


async def test_before_tool_fails_closed_on_interceptor_error() -> None:
  plugin = AgentHooksPlugin(interceptors=[_Raiser()])
  result = await plugin.before_tool_callback(
      tool=_tool("lookup"),
      tool_args={"x": 1},
      tool_context=_tool_context(),
  )
  assert result is not None
  assert result["agent_hooks_blocked"] is True


async def test_before_tool_fails_closed_with_no_interceptors() -> None:
  plugin = AgentHooksPlugin(interceptors=[])
  result = await plugin.before_tool_callback(
      tool=_tool("lookup"),
      tool_args={"x": 1},
      tool_context=_tool_context(),
  )
  assert result is not None
  assert result["agent_hooks_blocked"] is True


# ---------------------------------------------------------------------------
# Model point
# ---------------------------------------------------------------------------


async def test_before_model_deny_returns_blocked_response() -> None:
  plugin = AgentHooksPlugin(interceptors=[_DenyModel()])
  response = await plugin.before_model_callback(
      callback_context=_callback_context(),
      llm_request=LlmRequest(model="gemini-2.5-flash"),
  )
  assert isinstance(response, LlmResponse)
  assert response.custom_metadata is not None
  assert response.custom_metadata["agent_hooks_blocked"] is True


async def test_before_model_transform_fails_closed() -> None:
  plugin = AgentHooksPlugin(interceptors=[_TransformModel()])
  response = await plugin.before_model_callback(
      callback_context=_callback_context(),
      llm_request=LlmRequest(model="gemini-2.5-flash"),
  )
  # A transform at pre_model_call is not round-trip safe -> fail closed.
  assert isinstance(response, LlmResponse)
  assert response.custom_metadata["agent_hooks_blocked"] is True


async def test_after_model_transform_rewrites_content() -> None:
  plugin = AgentHooksPlugin(interceptors=[_TransformModel()])
  original = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="unsafe")]
      )
  )
  response = await plugin.after_model_callback(
      callback_context=_callback_context(),
      llm_response=original,
  )
  assert isinstance(response, LlmResponse)
  assert response.content is not None
  assert response.content.parts[0].text == "SAFE"


async def test_after_model_allow_returns_none() -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  response = await plugin.after_model_callback(
      callback_context=_callback_context(),
      llm_response=LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part.from_text(text="hi")]
          )
      ),
  )
  assert response is None


# ---------------------------------------------------------------------------
# Input / output points
# ---------------------------------------------------------------------------


async def test_input_deny_replaces_message() -> None:
  plugin = AgentHooksPlugin(interceptors=[_DenyInput()])
  result = await plugin.on_user_message_callback(
      invocation_context=_invocation_context(),
      user_message=types.Content(
          role="user", parts=[types.Part.from_text(text="malicious")]
      ),
  )
  assert result is not None
  assert result.role == "user"
  assert "input blocked" in result.parts[0].text


async def test_input_transform_rewrites_message() -> None:
  plugin = AgentHooksPlugin(interceptors=[_TransformInput()])
  result = await plugin.on_user_message_callback(
      invocation_context=_invocation_context(),
      user_message=types.Content(
          role="user", parts=[types.Part.from_text(text="raw")]
      ),
  )
  assert result is not None
  assert result.parts[0].text == "CLEANED"


async def test_output_deny_replaces_event_content() -> None:
  plugin = AgentHooksPlugin(interceptors=[_DenyOutput()])
  event = await plugin.on_event_callback(
      invocation_context=_invocation_context(),
      event=_final_event("leaked secret"),
  )
  assert event is not None
  assert event.content is not None
  assert "output blocked" in event.content.parts[0].text


async def test_output_ignores_non_final_events() -> None:
  plugin = AgentHooksPlugin(interceptors=[_DenyOutput()])
  non_final = Event(
      invocation_id="inv-1",
      author="agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(function_call=types.FunctionCall(name="t", args={}))
          ],
      ),
  )
  result = await plugin.on_event_callback(
      invocation_context=_invocation_context(), event=non_final
  )
  assert result is None


# ---------------------------------------------------------------------------
# Startup / lifecycle / state hygiene
# ---------------------------------------------------------------------------


async def test_before_run_deny_halts() -> None:
  class _DenyStartup:

    def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "agent_startup":
        return Verdict.deny(reason="startup_denied")
      return Verdict(decision=Decision.ALLOW)

  plugin = AgentHooksPlugin(interceptors=[_DenyStartup()])
  result = await plugin.before_run_callback(
      invocation_context=_invocation_context()
  )
  assert result is not None
  assert "blocked by agent-hooks" in result.parts[0].text


async def test_state_is_evicted_after_run() -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  ic = _invocation_context()
  await plugin.before_run_callback(invocation_context=ic)
  assert ic.invocation_id in plugin._states
  await plugin.after_run_callback(invocation_context=ic)
  assert ic.invocation_id not in plugin._states


async def test_state_is_evicted_on_run_error() -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  ic = _invocation_context()
  await plugin.before_run_callback(invocation_context=ic)
  assert ic.invocation_id in plugin._states
  await plugin.on_run_error_callback(
      invocation_context=ic, error=RuntimeError("x")
  )
  assert ic.invocation_id not in plugin._states


async def test_close_clears_state() -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  await plugin.before_run_callback(invocation_context=_invocation_context())
  assert plugin._states
  await plugin.close()
  assert not plugin._states


async def test_evaluate_only_does_not_block() -> None:
  plugin = AgentHooksPlugin(
      interceptors=[_DenyTool("delete_account")], mode="evaluate_only"
  )
  result = await plugin.before_tool_callback(
      tool=_tool("delete_account"),
      tool_args={"user_id": 42},
      tool_context=_tool_context(),
  )
  # evaluate_only records the deny but does not enforce it.
  assert result is None


async def test_record_sink_receives_records() -> None:
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )
  await plugin.before_tool_callback(
      tool=_tool("safe"),
      tool_args={"x": 1},
      tool_context=_tool_context(),
  )
  assert len(records) == 1
  assert records[0].interception_point.value == "pre_tool_call"


# ---------------------------------------------------------------------------
# Enforcement-mode fidelity, startup enforcement, and audit correlation
# ---------------------------------------------------------------------------


async def test_evaluate_only_does_not_transform_tool_args() -> None:
  plugin = AgentHooksPlugin(
      interceptors=[_TransformToolArgs()], mode="evaluate_only"
  )
  args = {"user_id": 42}
  result = await plugin.before_tool_callback(
      tool=_tool("lookup"),
      tool_args=args,
      tool_context=_tool_context(),
  )
  # evaluate_only records the transform verdict but must not rewrite the args.
  assert result is None
  assert args == {"user_id": 42}


async def test_evaluate_only_preserves_model_response() -> None:
  plugin = AgentHooksPlugin(
      interceptors=[_TransformModel()], mode="evaluate_only"
  )
  original = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part.from_text(text="calling tool"),
              types.Part(function_call=types.FunctionCall(name="t", args={})),
          ],
      )
  )
  response = await plugin.after_model_callback(
      callback_context=_callback_context(),
      llm_response=original,
  )
  # evaluate_only must not rewrite the response or drop the tool-call part.
  assert response is None


async def test_startup_deny_blocks_model_call() -> None:
  class _DenyStartup:

    def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "agent_startup":
        return Verdict.deny(reason="startup_denied")
      return Verdict(decision=Decision.ALLOW)

  plugin = AgentHooksPlugin(interceptors=[_DenyStartup()])
  await plugin.before_run_callback(invocation_context=_invocation_context())
  # Runner paths that ignore before_run's return value must still be blocked
  # at the first model call.
  response = await plugin.before_model_callback(
      callback_context=_callback_context(),
      llm_request=LlmRequest(model="gemini-2.5-flash"),
  )
  assert isinstance(response, LlmResponse)
  assert response.custom_metadata is not None
  assert response.custom_metadata["agent_hooks_blocked"] is True
  assert "startup_denied" in response.custom_metadata["reason"]


async def test_synth_call_id_correlates_pre_and_post_after_transform() -> None:
  class _CaptureAndTransform:

    def __init__(self) -> None:
      self.ids: dict[str, str] = {}

    def intercept(self, ctx: AgentContext) -> Any:
      point = ctx["interception_point"]
      if point == "pre_tool_call":
        self.ids["pre"] = ctx["tool_call"]["id"]
        new_args = dict(ctx["tool_call"]["args"])
        new_args["redacted"] = True
        return Verdict(
            decision=Decision.TRANSFORM,
            transform=Transform(path="$target", value=new_args),
        )
      if point == "post_tool_call":
        self.ids["post"] = ctx["tool_call"]["id"]
      return Verdict(decision=Decision.ALLOW)

  interceptor = _CaptureAndTransform()
  plugin = AgentHooksPlugin(interceptors=[interceptor])
  # No function_call_id -> the plugin synthesizes the id and must correlate the
  # pre/post pair even though the pre-tool transform rewrites the args.
  tool_context = _tool_context(function_call_id=None)
  args = {"user_id": 42}
  await plugin.before_tool_callback(
      tool=_tool("lookup"), tool_args=args, tool_context=tool_context
  )
  assert args == {"user_id": 42, "redacted": True}
  await plugin.after_tool_callback(
      tool=_tool("lookup"),
      tool_args=args,
      tool_context=tool_context,
      result={"ok": True},
  )
  assert interceptor.ids["pre"] == interceptor.ids["post"]
  assert interceptor.ids["pre"].startswith("tc-")


def test_json_safe_bounds_recursion_depth() -> None:
  from google.adk.plugins._agent_hooks_plugin import _json_safe

  node: dict[str, Any] = {}
  cursor = node
  for _ in range(5000):
    child: dict[str, Any] = {}
    cursor["next"] = child
    cursor = child
  # Deeply nested untrusted input must truncate, not raise RecursionError.
  assert "<max-depth>" in str(_json_safe(node))


def test_missing_dependency_raises_actionable_error(monkeypatch) -> None:
  import google.adk.plugins._agent_hooks_plugin as mod

  def _boom(_name: str) -> Any:
    raise ImportError("no module")

  monkeypatch.setattr(mod.importlib, "import_module", _boom)
  with pytest.raises(ImportError, match="google-adk\\[agent-hooks\\]"):
    AgentHooksPlugin(interceptors=[])
