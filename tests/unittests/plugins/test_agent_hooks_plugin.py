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

import asyncio
from collections.abc import AsyncGenerator
import copy
import threading
from typing import Any
from unittest.mock import Mock

from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

agent_hooks = pytest.importorskip("agent_hooks")

from google.adk.agents.base_agent import BaseAgent  # noqa: E402
from google.adk.agents.invocation_context import InvocationContext  # noqa: E402
from google.adk.agents.llm_agent import Agent  # noqa: E402
from google.adk.apps.app import App  # noqa: E402
from google.adk.flows.llm_flows.functions import handle_function_calls_async  # noqa: E402
from google.adk.plugins import AgentHooksPlugin  # noqa: E402
from google.adk.plugins.base_plugin import BasePlugin  # noqa: E402
from google.adk.runners import InMemoryRunner as AdkInMemoryRunner  # noqa: E402
from google.adk.tools.function_tool import FunctionTool  # noqa: E402

from tests.unittests import testing_utils  # noqa: E402

AgentContext = agent_hooks.AgentContext
Decision = agent_hooks.Decision
Transform = agent_hooks.Transform
Verdict = agent_hooks.Verdict


# ---------------------------------------------------------------------------
# Interceptors (real agent-hooks verdicts)
# ---------------------------------------------------------------------------


class _AllowAll:

  async def intercept(self, ctx: AgentContext) -> Any:
    return Verdict(decision=Decision.ALLOW)


class _DenyTool:

  def __init__(self, tool_name: str) -> None:
    self._tool_name = tool_name

  async def intercept(self, ctx: AgentContext) -> Any:
    if (
        ctx["interception_point"] == "pre_tool_call"
        and ctx["tool_call"]["name"] == self._tool_name
    ):
      return Verdict.deny(reason="tool_denied", message="not allowed")
    return Verdict(decision=Decision.ALLOW)


class _TransformToolArgs:

  async def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "pre_tool_call":
      new_args = dict(ctx["tool_call"]["args"])
      new_args["redacted"] = True
      return Verdict(
          decision=Decision.TRANSFORM,
          transform=Transform(path="$target", value=new_args),
      )
    return Verdict(decision=Decision.ALLOW)


class _TransformToolResult:

  async def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "post_tool_call":
      return Verdict(
          decision=Decision.TRANSFORM,
          transform=Transform(path="$target", value={"clean": "value"}),
      )
    return Verdict(decision=Decision.ALLOW)


class _DenyModel:

  async def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "pre_model_call":
      return Verdict.deny(reason="model_denied")
    return Verdict(decision=Decision.ALLOW)


class _TransformModel:

  async def intercept(self, ctx: AgentContext) -> Any:
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


class _TransformPostModel:

  async def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "post_model_call":
      return Verdict(
          decision=Decision.TRANSFORM,
          transform=Transform(path="$target.content", value="SAFE"),
      )
    return Verdict(decision=Decision.ALLOW)


class _DenyInput:

  async def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "input":
      return Verdict.deny(reason="input_denied")
    return Verdict(decision=Decision.ALLOW)


class _TransformInput:

  async def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "input":
      return Verdict(
          decision=Decision.TRANSFORM,
          transform=Transform(path="$target.content", value="CLEANED"),
      )
    return Verdict(decision=Decision.ALLOW)


class _DenyOutput:

  async def intercept(self, ctx: AgentContext) -> Any:
    if ctx["interception_point"] == "output":
      return Verdict.deny(reason="output_denied")
    return Verdict(decision=Decision.ALLOW)


class _Raiser:

  async def intercept(self, ctx: AgentContext) -> Any:
    raise RuntimeError("boom")


class _MalformedVerdict:

  async def intercept(self, ctx: AgentContext) -> Any:
    return {"decision": "transform"}


class _NeverReturns:

  def __init__(self) -> None:
    self.cancelled = False

  async def intercept(self, ctx: AgentContext) -> Any:
    try:
      await asyncio.Event().wait()
    except asyncio.CancelledError:
      self.cancelled = True
      raise


# ---------------------------------------------------------------------------
# Context factories (the plugin only reads a few attributes)
# ---------------------------------------------------------------------------


class _TestRunToken:

  def __init__(
      self, nonce: str, owner_identity: tuple[str, str, str, str]
  ) -> None:
    self.nonce = nonce
    self.owner_identity = owner_identity
    self.closed = False

  def close(self) -> None:
    self.closed = True


_TEST_RUN_TOKENS: dict[str, _TestRunToken] = {}


def _test_run_token(
    nonce: str, owner_identity: tuple[str, str, str, str]
) -> _TestRunToken:
  token = _TEST_RUN_TOKENS.get(nonce)
  if token is None or token.closed:
    token = _TestRunToken(nonce, owner_identity)
    _TEST_RUN_TOKENS[nonce] = token
  return token


def _invocation_context(
    *,
    invocation_id: str = "inv-1",
    session_id: str = "sess-1",
    agent_name: str = "agent",
    tools: list[Any] | None = None,
    run_nonce: str | None = None,
) -> Any:
  agent = Mock()
  agent.name = agent_name
  agent.tools = tools or []
  session = Mock()
  session.id = session_id
  session.app_name = "test_app"
  session.user_id = "test_user"
  nonce = run_nonce or f"run:{session_id}:{invocation_id}"
  owner_identity = (
      session.app_name,
      session.user_id,
      session.id,
      invocation_id,
  )
  ic = Mock()
  ic.invocation_id = invocation_id
  ic.run_nonce = nonce
  ic._run_identity_token = _test_run_token(nonce, owner_identity)
  ic.app_name = session.app_name
  ic.user_id = session.user_id
  ic.agent = agent
  ic.session = session
  return ic


def _callback_context(
    *,
    invocation_id: str = "inv-1",
    session_id: str = "sess-1",
    agent_name: str = "agent",
    run_nonce: str | None = None,
) -> Any:
  session = Mock()
  session.id = session_id
  session.app_name = "test_app"
  session.user_id = "test_user"
  nonce = run_nonce or f"run:{session_id}:{invocation_id}"
  owner_identity = (
      session.app_name,
      session.user_id,
      session.id,
      invocation_id,
  )
  ctx = Mock()
  ctx.invocation_id = invocation_id
  ctx.run_nonce = nonce
  ctx._run_identity_token = _test_run_token(nonce, owner_identity)
  ctx.user_id = session.user_id
  ctx.session = session
  ctx.agent_name = agent_name
  return ctx


def _tool_context(
    *,
    invocation_id: str = "inv-1",
    session_id: str = "sess-1",
    agent_name: str = "agent",
    function_call_id: str | None = "fc-1",
    run_nonce: str | None = None,
) -> Any:
  ctx = _callback_context(
      invocation_id=invocation_id,
      session_id=session_id,
      agent_name=agent_name,
      run_nonce=run_nonce,
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
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_DenyTool("delete_account")], record_sink=records.append
  )
  args = {"user_id": 42}
  result = await plugin.before_tool_callback(
      tool=_tool("delete_account"),
      tool_args=args,
      tool_context=_tool_context(),
  )
  assert result is not None
  assert result["agent_hooks_blocked"] is True
  assert result["reason"] == "policy_denied"
  assert "error" in result
  assert records[-1].verdict.reason == "tool_denied"
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
  tool_context = _tool_context()
  tool_args = {"user_id": 42}
  assert (
      await plugin.before_tool_callback(
          tool=_tool("lookup"),
          tool_args=tool_args,
          tool_context=tool_context,
      )
      is None
  )
  result = await plugin.after_tool_callback(
      tool=_tool("lookup"),
      tool_args=tool_args,
      tool_context=tool_context,
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


async def test_before_tool_fails_closed_on_malformed_verdict() -> None:
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_MalformedVerdict()], record_sink=records.append
  )

  result = await plugin.before_tool_callback(
      tool=_tool("lookup"),
      tool_args={"x": 1},
      tool_context=_tool_context(),
  )

  assert result is not None
  assert result["agent_hooks_blocked"] is True
  assert result["reason"] == "policy_denied"
  assert records[-1].verdict.reason == "host_error:verdict_invalid"


async def test_before_tool_timeout_cancels_interceptor() -> None:
  interceptor = _NeverReturns()
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[interceptor], timeout=0.01, record_sink=records.append
  )

  result = await plugin.before_tool_callback(
      tool=_tool("lookup"),
      tool_args={"x": 1},
      tool_context=_tool_context(),
  )

  assert result is not None
  assert result["agent_hooks_blocked"] is True
  assert result["reason"] == "policy_denied"
  assert records[-1].verdict.reason == "host_error:interceptor_timeout"
  assert interceptor.cancelled


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
  plugin = AgentHooksPlugin(interceptors=[_TransformPostModel()])
  callback_context = _callback_context()
  assert (
      await plugin.before_model_callback(
          callback_context=callback_context,
          llm_request=LlmRequest(model="gemini-2.5-flash"),
      )
      is None
  )
  original = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="unsafe")]
      )
  )
  response = await plugin.after_model_callback(
      callback_context=callback_context,
      llm_response=original,
  )
  assert isinstance(response, LlmResponse)
  assert response.content is not None
  assert response.content.parts[0].text == "SAFE"


async def test_after_model_allow_returns_none() -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  callback_context = _callback_context()
  assert (
      await plugin.before_model_callback(
          callback_context=callback_context,
          llm_request=LlmRequest(model="gemini-2.5-flash"),
      )
      is None
  )
  response = await plugin.after_model_callback(
      callback_context=callback_context,
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
  assert result.parts[0].text == "[blocked by policy]"


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
  assert event.content.parts[0].text == "[blocked by policy]"


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

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "agent_startup":
        return Verdict.deny(reason="startup_denied")
      return Verdict(decision=Decision.ALLOW)

  plugin = AgentHooksPlugin(interceptors=[_DenyStartup()])
  result = await plugin.before_run_callback(
      invocation_context=_invocation_context()
  )
  assert result is not None
  assert result.parts[0].text == "[blocked by policy]"


async def test_state_is_evicted_after_run() -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  ic = _invocation_context()
  await plugin.before_run_callback(invocation_context=ic)
  assert ic.run_nonce in plugin._states
  await plugin.on_run_complete_callback(invocation_context=ic)
  assert ic.run_nonce not in plugin._states


async def test_state_is_evicted_on_run_error() -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  ic = _invocation_context()
  await plugin.before_run_callback(invocation_context=ic)
  assert ic.run_nonce in plugin._states
  await plugin.on_run_error_callback(
      invocation_context=ic, error=RuntimeError("x")
  )
  assert ic.run_nonce not in plugin._states


async def test_close_clears_state() -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  await plugin.before_run_callback(invocation_context=_invocation_context())
  assert plugin._states
  await plugin.close()
  assert not plugin._states

  with pytest.raises(RuntimeError, match="closed"):
    await plugin.before_run_callback(
        invocation_context=_invocation_context(invocation_id="after-close")
    )


async def test_active_invocation_state_is_bounded_and_released() -> None:
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], max_active_invocations=1
  )
  first = _invocation_context(invocation_id="first")
  second = _invocation_context(invocation_id="second")
  await plugin.before_run_callback(invocation_context=first)

  with pytest.raises(RuntimeError, match="capacity exhausted"):
    await plugin.before_run_callback(invocation_context=second)

  await plugin.on_run_cancelled_callback(invocation_context=first)
  assert await plugin.before_run_callback(invocation_context=second) is None


async def test_cancelled_close_drains_terminal_audit_and_rejects_work() -> None:
  class _HoldShutdown:

    def __init__(self) -> None:
      self.entered = asyncio.Event()
      self.release = asyncio.Event()
      self.shutdown_count = 0

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "agent_shutdown":
        self.shutdown_count += 1
        self.entered.set()
        await self.release.wait()
      return Verdict(decision=Decision.ALLOW)

  interceptor = _HoldShutdown()
  plugin = AgentHooksPlugin(interceptors=[interceptor])
  await plugin.before_run_callback(
      invocation_context=_invocation_context(invocation_id="active")
  )
  close_task = asyncio.create_task(plugin.close())
  await interceptor.entered.wait()

  with pytest.raises(RuntimeError, match="closed"):
    await plugin.before_run_callback(
        invocation_context=_invocation_context(invocation_id="late")
    )
  close_task.cancel()
  interceptor.release.set()

  with pytest.raises(asyncio.CancelledError):
    await close_task
  assert interceptor.shutdown_count == 1
  assert plugin._states == {}


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


async def test_record_sink_failure_is_logged_and_fails_closed(caplog) -> None:
  def failing_sink(_record: Any) -> None:
    raise RuntimeError("sensitive sink detail")

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=failing_sink
  )

  with caplog.at_level("ERROR"):
    result = await plugin.before_tool_callback(
        tool=_tool("safe"),
        tool_args={"x": 1},
        tool_context=_tool_context(),
    )

  assert result is not None
  assert result["agent_hooks_blocked"] is True
  assert "agent-hooks audit sink failed" in caplog.text
  assert "sensitive sink detail" not in caplog.text


async def test_record_sink_log_mode_preserves_policy_result(caplog) -> None:
  def failing_sink(_record: Any) -> None:
    raise RuntimeError("sink down")

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()],
      record_sink=failing_sink,
      audit_failure_mode="log",
  )

  with caplog.at_level("ERROR"):
    result = await plugin.before_tool_callback(
        tool=_tool("safe"),
        tool_args={"x": 1},
        tool_context=_tool_context(),
    )

  assert result is None
  assert "agent-hooks audit sink failed" in caplog.text


async def test_successful_audit_acknowledgments_are_not_retained() -> None:
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()],
      record_sink=records.append,
      max_records=2,
  )
  invocation_context = _invocation_context(invocation_id="many-records")
  await plugin.before_run_callback(invocation_context=invocation_context)

  for index in range(10):
    assert (
        await plugin.on_event_callback(
            invocation_context=invocation_context,
            event=_final_event(text=str(index)),
        )
        is None
    )

  state = plugin._states[invocation_context.run_nonce]
  assert state.delivered_record_sequences == set()
  assert state.late_delivery_sequences == set()
  assert len(state.emitter.results) == 2
  await plugin.on_run_complete_callback(invocation_context=invocation_context)


async def test_audit_timeout_retains_bounded_admission(caplog) -> None:
  entered = threading.Event()
  release = threading.Event()
  calls: list[int] = []

  def blocking_sink(record: Any) -> None:
    calls.append(record.sequence)
    entered.set()
    release.wait()

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()],
      record_sink=blocking_sink,
      audit_timeout=0.01,
      max_pending_audit_records=1,
      audit_workers=1,
  )
  with caplog.at_level("ERROR"):
    first = await plugin.before_tool_callback(
        tool=_tool("first"),
        tool_args={},
        tool_context=_tool_context(function_call_id="first"),
    )
    second = await plugin.before_tool_callback(
        tool=_tool("second"),
        tool_args={},
        tool_context=_tool_context(function_call_id="second"),
    )

  assert entered.is_set()
  assert first is not None and first["agent_hooks_blocked"] is True
  assert second is not None and second["agent_hooks_blocked"] is True
  assert len(calls) == 1
  assert caplog.text.count("agent-hooks audit sink failed") == 2

  release.set()
  await plugin.close()


async def test_late_audit_success_is_not_resubmitted() -> None:
  entered = threading.Event()
  release = threading.Event()
  calls: list[int] = []

  def slow_sink(record: Any) -> None:
    calls.append(record.sequence)
    entered.set()
    release.wait()

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()],
      record_sink=slow_sink,
      audit_timeout=0.01,
      max_pending_audit_records=1,
  )
  result = await plugin.before_tool_callback(
      tool=_tool("slow"),
      tool_args={},
      tool_context=_tool_context(),
  )

  assert result is not None
  assert entered.is_set()
  release.set()
  await plugin.close()

  assert calls.count(0) == 1


async def test_failed_close_retains_state_for_retry() -> None:
  failing = True

  def recovering_sink(_record: Any) -> None:
    if failing:
      raise RuntimeError("offline")

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=recovering_sink
  )
  await plugin.before_run_callback(
      invocation_context=_invocation_context(invocation_id="retry-close")
  )

  with pytest.raises(RuntimeError, match="failed to finalize"):
    await plugin.close()
  assert plugin._states
  assert plugin._audit_executor is not None

  failing = False
  await plugin.close()

  assert plugin._states == {}
  assert plugin._audit_executor is None


async def test_persistent_audit_failures_have_bounded_retention() -> None:
  failing = True

  def recovering_sink(_record: Any) -> None:
    if failing:
      raise RuntimeError("offline")

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()],
      record_sink=recovering_sink,
      audit_failure_mode="log",
      max_retained_audit_records=3,
  )
  invocation_context = _invocation_context(invocation_id="bounded-retry")

  for index in range(10):
    await plugin.on_event_callback(
        invocation_context=invocation_context,
        event=_final_event(text=str(index)),
    )

  state = plugin._states[invocation_context.run_nonce]
  assert len(state.undelivered_records) == 2
  failing = False
  await plugin.close()
  assert plugin._states == {}


async def test_terminal_audit_failure_blocks_run_and_retries_on_close() -> None:
  failing = True
  delivered: list[tuple[int, str]] = []

  def recovering_sink(record: Any) -> None:
    if failing:
      raise RuntimeError("audit unavailable")
    delivered.append((record.sequence, record.interception_point.value))

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=recovering_sink
  )
  agent = Agent(
      name="root", model=testing_utils.MockModel.create(responses=["done"])
  )
  runner = testing_utils.InMemoryRunner(
      app=App(name="test_app", root_agent=agent, plugins=[plugin])
  )

  with pytest.raises(RuntimeError, match="on_run_complete_callback"):
    await runner.run_async("hello")

  assert plugin._states
  failing = False
  await plugin.close()

  assert plugin._states == {}
  assert delivered
  assert delivered[-1][1] == "agent_shutdown"
  assert len({sequence for sequence, _ in delivered}) == len(delivered)


async def test_cancelled_post_sink_does_not_create_duplicate_post() -> None:
  post_entered = threading.Event()
  release_post = threading.Event()
  delivered: list[tuple[int, str]] = []

  def blocking_post_sink(record: Any) -> None:
    delivered.append((record.sequence, record.interception_point.value))
    if record.interception_point.value == "post_tool_call":
      post_entered.set()
      release_post.wait()

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()],
      record_sink=blocking_post_sink,
      audit_timeout=5.0,
  )
  tool_context = _tool_context()
  assert (
      await plugin.before_tool_callback(
          tool=_tool("lookup"),
          tool_args={"x": 1},
          tool_context=tool_context,
      )
      is None
  )
  post_task = asyncio.create_task(
      plugin.after_tool_callback(
          tool=_tool("lookup"),
          tool_args={"x": 1},
          tool_context=tool_context,
          result={"ok": True},
      )
  )
  await asyncio.to_thread(post_entered.wait)
  post_task.cancel()
  release_post.set()

  with pytest.raises(asyncio.CancelledError):
    await post_task
  await plugin.on_run_cancelled_callback(
      invocation_context=_invocation_context()
  )
  await plugin.close()

  post_records = [
      sequence for sequence, point in delivered if point == "post_tool_call"
  ]
  assert len(post_records) == 1
  assert len(set(post_records)) == 1


async def test_cancelled_fanout_post_sink_does_not_duplicate_model_post() -> (
    None
):
  post_entered = threading.Event()
  release_post = threading.Event()
  delivered: list[tuple[int, str]] = []

  def blocking_post_sink(record: Any) -> None:
    delivered.append((record.sequence, record.interception_point.value))
    if record.interception_point.value == "post_model_call":
      post_entered.set()
      release_post.wait()

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()],
      record_sink=blocking_post_sink,
      audit_timeout=5.0,
      max_tool_calls_per_response=1,
  )
  callback_context = _callback_context()
  assert (
      await plugin.before_model_callback(
          callback_context=callback_context,
          llm_request=LlmRequest(model="model"),
      )
      is None
  )
  response = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="one", name="one", args={}
                  )
              ),
              types.Part(
                  function_call=types.FunctionCall(
                      id="two", name="two", args={}
                  )
              ),
          ],
      )
  )
  post_task = asyncio.create_task(
      plugin.after_model_callback(
          callback_context=callback_context, llm_response=response
      )
  )
  await asyncio.to_thread(post_entered.wait)
  post_task.cancel()
  release_post.set()

  with pytest.raises(asyncio.CancelledError):
    await post_task
  await plugin.on_run_cancelled_callback(
      invocation_context=_invocation_context()
  )
  await plugin.close()

  model_posts = [
      sequence for sequence, point in delivered if point == "post_model_call"
  ]
  assert len(model_posts) == 1


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
  callback_context = _callback_context()
  assert (
      await plugin.before_model_callback(
          callback_context=callback_context,
          llm_request=LlmRequest(model="gemini-2.5-flash"),
      )
      is None
  )
  response = await plugin.after_model_callback(
      callback_context=callback_context,
      llm_response=original,
  )
  # evaluate_only must not rewrite the response or drop the tool-call part.
  assert response is None


async def test_startup_deny_blocks_model_call() -> None:
  class _DenyStartup:

    async def intercept(self, ctx: AgentContext) -> Any:
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
  assert response.custom_metadata["reason"] == "policy_denied"
  assert set(response.custom_metadata) == {"agent_hooks_blocked", "reason"}


async def test_synth_call_id_correlates_pre_and_post_after_transform() -> None:
  class _CaptureAndTransform:

    def __init__(self) -> None:
      self.ids: dict[str, str] = {}

    async def intercept(self, ctx: AgentContext) -> Any:
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
  assert tool_context.function_call_id is None
  assert tool_context._agent_hooks_call_id == interceptor.ids["pre"]


async def test_duplicate_tool_call_id_fails_closed_without_false_post() -> None:
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )
  first_context = _tool_context(function_call_id="duplicate")
  second_context = _tool_context(function_call_id="duplicate")
  first_result = await plugin.before_tool_callback(
      tool=_tool("lookup"),
      tool_args={"request": 1},
      tool_context=first_context,
  )
  second_result = await plugin.before_tool_callback(
      tool=_tool("lookup"),
      tool_args={"request": 2},
      tool_context=second_context,
  )

  assert first_result is None
  assert second_result is not None
  assert second_result["agent_hooks_blocked"] is True
  assert first_context._agent_hooks_call_id == "duplicate"
  assert second_context._agent_hooks_call_id != "duplicate"

  await plugin.after_tool_callback(
      tool=_tool("lookup"),
      tool_args={"request": 1},
      tool_context=first_context,
      result={"ok": True},
  )
  await plugin.after_tool_callback(
      tool=_tool("lookup"),
      tool_args={"request": 2},
      tool_context=second_context,
      result=second_result,
  )

  points = _points(records)
  assert points.count("pre_tool_call") == 2
  assert points.count("post_tool_call") == 1
  assert records[1].verdict.reason == "host_error:context_invalid"


async def test_tool_call_tracking_capacity_fails_closed() -> None:
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], max_tracked_tool_calls=1
  )
  first_context = _tool_context(function_call_id="first")
  second_context = _tool_context(function_call_id="second")

  assert (
      await plugin.before_tool_callback(
          tool=_tool("lookup"),
          tool_args={},
          tool_context=first_context,
      )
      is None
  )
  result = await plugin.before_tool_callback(
      tool=_tool("lookup"),
      tool_args={},
      tool_context=second_context,
  )

  assert result is not None
  assert result["agent_hooks_blocked"] is True


async def test_parallel_tool_capacity_is_reserved_before_policy_await() -> None:
  class _ObservedLock:

    def __init__(self) -> None:
      self._lock = asyncio.Lock()
      self.waiter_entered = asyncio.Event()

    async def __aenter__(self) -> _ObservedLock:
      if self._lock.locked():
        self.waiter_entered.set()
      await self._lock.acquire()
      return self

    async def __aexit__(self, *args: Any) -> None:
      self._lock.release()

  class _HoldFirstTool:

    def __init__(self) -> None:
      self.entered = asyncio.Event()
      self.release = asyncio.Event()

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "pre_tool_call":
        self.entered.set()
        await self.release.wait()
      return Verdict(decision=Decision.ALLOW)

  interceptor = _HoldFirstTool()
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[interceptor],
      max_tracked_tool_calls=1,
      record_sink=records.append,
  )
  first_context = _tool_context(function_call_id="first")
  second_context = _tool_context(function_call_id="second")
  state = plugin._state_for_context(first_context)
  observed_lock = _ObservedLock()
  state.emission_lock = observed_lock
  first_task = asyncio.create_task(
      plugin.before_tool_callback(
          tool=_tool("first"),
          tool_args={},
          tool_context=first_context,
      )
  )
  await interceptor.entered.wait()
  second_task = asyncio.create_task(
      plugin.before_tool_callback(
          tool=_tool("second"),
          tool_args={},
          tool_context=second_context,
      )
  )
  await observed_lock.waiter_entered.wait()

  assert len(state.open_tool_calls) == 1
  interceptor.release.set()
  first_result, second_result = await asyncio.gather(first_task, second_task)

  assert first_result is None
  assert second_result is not None
  assert second_result["agent_hooks_blocked"] is True
  assert records[-1].verdict.reason == "host_error:context_invalid"


async def test_deep_tool_args_fail_closed() -> None:
  node: dict[str, Any] = {}
  cursor = node
  for _ in range(5000):
    child: dict[str, Any] = {}
    cursor["next"] = child
    cursor = child

  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )
  result = await plugin.before_tool_callback(
      tool=_tool("deep"),
      tool_args=node,
      tool_context=_tool_context(),
  )

  assert result is not None
  assert result["agent_hooks_blocked"] is True
  assert records[-1].verdict.reason == "host_error:context_invalid"


@pytest.mark.parametrize(
    "unsafe_value",
    [object(), b"raw-bytes", {1: "non-string-key"}],
)
async def test_unprojectable_tool_args_record_context_invalid(
    unsafe_value: Any,
) -> None:
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )

  result = await plugin.before_tool_callback(
      tool=_tool("unsafe"),
      tool_args={"value": unsafe_value},
      tool_context=_tool_context(),
  )

  assert result is not None
  assert result["agent_hooks_blocked"] is True
  assert records[-1].verdict.reason == "host_error:context_invalid"


async def test_cyclic_tool_args_record_context_invalid() -> None:
  cyclic: dict[str, Any] = {}
  cyclic["self"] = cyclic
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )

  result = await plugin.before_tool_callback(
      tool=_tool("cyclic"),
      tool_args=cyclic,
      tool_context=_tool_context(),
  )

  assert result is not None
  assert records[-1].verdict.reason == "host_error:context_invalid"


async def test_wide_context_exceeding_node_budget_fails_closed(
    monkeypatch,
) -> None:
  import google.adk.plugins._agent_hooks_plugin as module

  monkeypatch.setattr(module, "_MAX_CONTEXT_NODES", 5)
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )

  result = await plugin.before_tool_callback(
      tool=_tool("wide"),
      tool_args={"values": [1, 2, 3, 4, 5, 6]},
      tool_context=_tool_context(),
  )

  assert result is not None
  assert records[-1].verdict.reason == "host_error:context_invalid"


async def test_text_exceeding_byte_budget_fails_closed(monkeypatch) -> None:
  import google.adk.plugins._agent_hooks_plugin as module

  monkeypatch.setattr(module, "_MAX_CONTEXT_TEXT_BYTES", 4)
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )

  result = await plugin.on_user_message_callback(
      invocation_context=_invocation_context(),
      user_message=types.Content(
          role="user", parts=[types.Part.from_text(text="12345")]
      ),
  )

  assert result is not None
  assert records[-1].verdict.reason == "host_error:context_invalid"


async def test_blob_exceeding_byte_budget_fails_closed(monkeypatch) -> None:
  import google.adk.plugins._agent_hooks_plugin as module

  monkeypatch.setattr(module, "_MAX_CONTEXT_BLOB_BYTES", 4)
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )

  result = await plugin.on_user_message_callback(
      invocation_context=_invocation_context(),
      user_message=types.Content(
          role="user",
          parts=[
              types.Part(
                  inline_data=types.Blob(
                      mime_type="application/octet-stream", data=b"12345"
                  )
              )
          ],
      ),
  )

  assert result is not None
  assert records[-1].verdict.reason == "host_error:context_invalid"


async def test_content_role_exceeding_byte_budget_fails_closed(
    monkeypatch,
) -> None:
  import google.adk.plugins._agent_hooks_plugin as module

  monkeypatch.setattr(module, "_MAX_CONTEXT_TEXT_BYTES", 4)
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )

  result = await plugin.on_user_message_callback(
      invocation_context=_invocation_context(),
      user_message=types.Content(
          role="external-role",
          parts=[types.Part.from_text(text="x")],
      ),
  )

  assert result is not None
  assert records[-1].verdict.reason == "host_error:context_invalid"


async def test_multimodal_input_is_preserved_for_policy() -> None:
  class _CaptureInput:

    def __init__(self) -> None:
      self.content: Any = None

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "input":
        self.content = ctx["input"]["content"]
      return Verdict(decision=Decision.ALLOW)

  interceptor = _CaptureInput()
  plugin = AgentHooksPlugin(interceptors=[interceptor])
  result = await plugin.on_user_message_callback(
      invocation_context=_invocation_context(),
      user_message=types.Content(
          role="user",
          parts=[
              types.Part(
                  inline_data=types.Blob(
                      mime_type="image/png", data=b"not-a-real-image"
                  )
              )
          ],
      ),
  )

  assert result is None
  assert interceptor.content["role"] == "user"
  assert interceptor.content["parts"][0]["inline_data"]["mime_type"] == (
      "image/png"
  )


async def test_structured_system_and_function_call_are_visible_to_policy() -> (
    None
):
  class _CaptureMessages:

    def __init__(self) -> None:
      self.messages: list[dict[str, Any]] = []

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "pre_model_call":
        self.messages = ctx["messages"]
      return Verdict(decision=Decision.ALLOW)

  interceptor = _CaptureMessages()
  plugin = AgentHooksPlugin(interceptors=[interceptor])
  request = LlmRequest(
      model="model",
      contents=[
          types.Content(
              role="model",
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id="call", name="lookup", args={"x": 1}
                      )
                  )
              ],
          )
      ],
      config=types.GenerateContentConfig(
          system_instruction=types.Content(
              role="system",
              parts=[types.Part.from_text(text="structured policy")],
          )
      ),
  )

  assert (
      await plugin.before_model_callback(
          callback_context=_callback_context(), llm_request=request
      )
      is None
  )

  assert interceptor.messages[0]["role"] == "system"
  assert "structured policy" in str(interceptor.messages[0]["content"])
  assert "function_call" in str(interceptor.messages[1]["content"])


async def test_expanded_function_response_messages_are_bounded(
    monkeypatch,
) -> None:
  import google.adk.plugins._agent_hooks_plugin as module

  monkeypatch.setattr(module, "_MAX_MESSAGES", 2)
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )
  request = LlmRequest(
      model="model",
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id=str(index),
                          name="tool",
                          response={"result": index},
                      )
                  )
                  for index in range(3)
              ],
          )
      ],
  )

  response = await plugin.before_model_callback(
      callback_context=_callback_context(), llm_request=request
  )

  assert isinstance(response, LlmResponse)
  assert records[-1].verdict.reason == "host_error:context_invalid"


def test_agent_callback_cannot_bypass_governance() -> None:
  async def replace_tool_result(**kwargs: Any) -> dict[str, Any]:
    return {"bypassed": True}

  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  agent = Agent(
      name="agent",
      model=testing_utils.MockModel.create(responses=["ok"]),
      before_tool_callback=replace_tool_result,
  )

  with pytest.raises(ValueError, match="before_tool_callback"):
    testing_utils.InMemoryRunner(
        app=App(name="test_app", root_agent=agent, plugins=[plugin])
    )


def test_sub_agent_callback_cannot_bypass_governance() -> None:
  async def replace_model_response(**kwargs: Any) -> LlmResponse:
    return LlmResponse()

  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  sub_agent = Agent(
      name="sub_agent",
      model=testing_utils.MockModel.create(responses=["ok"]),
      after_model_callback=replace_model_response,
  )
  root_agent = Agent(
      name="root_agent",
      model=testing_utils.MockModel.create(responses=["ok"]),
      sub_agents=[sub_agent],
  )

  with pytest.raises(ValueError, match="after_model_callback"):
    testing_utils.InMemoryRunner(
        app=App(name="test_app", root_agent=root_agent, plugins=[plugin])
    )


def test_unsafe_composition_opt_out_allows_agent_callback() -> None:
  async def replace_tool_result(**kwargs: Any) -> dict[str, Any]:
    return {"cooperative": True}

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], allow_unsafe_plugin_composition=True
  )
  agent = Agent(
      name="agent",
      model=testing_utils.MockModel.create(responses=["ok"]),
      before_tool_callback=replace_tool_result,
  )

  runner = testing_utils.InMemoryRunner(
      app=App(name="test_app", root_agent=agent, plugins=[plugin])
  )

  assert runner is not None


def test_missing_dependency_raises_actionable_error(monkeypatch) -> None:
  import google.adk.plugins._agent_hooks_plugin as mod

  def _boom(_name: str) -> Any:
    raise ImportError("no module")

  monkeypatch.setattr(mod.importlib, "import_module", _boom)
  with pytest.raises(ImportError, match="google-adk\\[agent-hooks\\]"):
    AgentHooksPlugin(interceptors=[])


def test_timeout_rejects_synchronous_interceptor() -> None:
  class _SyncAllow:

    def intercept(self, ctx: AgentContext) -> Any:
      return Verdict(decision=Decision.ALLOW)

  with pytest.raises(ValueError, match="cannot preempt synchronous"):
    AgentHooksPlugin(interceptors=[_SyncAllow()])


def test_synchronous_interceptor_requires_explicit_unbounded_mode() -> None:
  class _SyncAllow:

    def intercept(self, ctx: AgentContext) -> Any:
      return Verdict(decision=Decision.ALLOW)

  plugin = AgentHooksPlugin(interceptors=[_SyncAllow()], timeout=None)

  assert plugin is not None


def test_unbounded_records_require_explicit_unsafe_opt_out() -> None:
  with pytest.raises(ValueError, match="allow_unsafe_unbounded_records"):
    AgentHooksPlugin(interceptors=[_AllowAll()], max_records=None)

  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()],
      max_records=None,
      allow_unsafe_unbounded_records=True,
  )

  assert plugin is not None


# ---------------------------------------------------------------------------
# Fail closed on an ENGINE error (record is None), not an interceptor error.
#
# A raising interceptor is substituted with a deny by the emitter (§6.3), so it
# returns a real record and never reaches the plugin's `record is None` branch.
# Only an emitter-level failure exercises that branch; patching emit_unchecked
# to raise pins it.
# ---------------------------------------------------------------------------


async def _raise_engine_error(self: Any, ctx: Any) -> Any:
  raise RuntimeError("agent-hooks engine exploded")


def _patch_engine_error(monkeypatch: Any) -> None:
  monkeypatch.setattr(
      agent_hooks.InterceptionEmitter, "emit_unchecked", _raise_engine_error
  )


async def test_input_fails_closed_on_engine_error(monkeypatch) -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)
  _patch_engine_error(monkeypatch)
  result = await plugin.on_user_message_callback(
      invocation_context=invocation_context,
      user_message=types.Content(
          role="user", parts=[types.Part.from_text(text="hi")]
      ),
  )
  assert result is not None
  assert result.parts[0].text == "[blocked by policy]"


async def test_before_model_fails_closed_on_engine_error(monkeypatch) -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  _patch_engine_error(monkeypatch)
  response = await plugin.before_model_callback(
      callback_context=_callback_context(),
      llm_request=LlmRequest(model="gemini-2.5-flash"),
  )
  assert isinstance(response, LlmResponse)
  assert response.custom_metadata is not None
  assert response.custom_metadata["agent_hooks_blocked"] is True


async def test_before_tool_fails_closed_on_engine_error(monkeypatch) -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  _patch_engine_error(monkeypatch)
  result = await plugin.before_tool_callback(
      tool=_tool("lookup"),
      tool_args={"x": 1},
      tool_context=_tool_context(),
  )
  assert result is not None
  assert result["agent_hooks_blocked"] is True


async def test_startup_fails_closed_on_engine_error(monkeypatch) -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  _patch_engine_error(monkeypatch)

  result = await plugin.before_run_callback(
      invocation_context=_invocation_context()
  )

  assert result is not None
  assert result.parts[0].text == "[blocked by policy]"


async def test_after_model_fails_closed_on_engine_error(monkeypatch) -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  callback_context = _callback_context()
  assert (
      await plugin.before_model_callback(
          callback_context=callback_context,
          llm_request=LlmRequest(model="gemini-2.5-flash"),
      )
      is None
  )
  _patch_engine_error(monkeypatch)

  result = await plugin.after_model_callback(
      callback_context=callback_context,
      llm_response=LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part.from_text(text="unsafe")]
          )
      ),
  )

  assert isinstance(result, LlmResponse)
  assert result.custom_metadata is not None
  assert result.custom_metadata["agent_hooks_blocked"] is True


async def test_after_tool_fails_closed_on_engine_error(monkeypatch) -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  tool_context = _tool_context()
  tool_args = {"x": 1}
  assert (
      await plugin.before_tool_callback(
          tool=_tool("lookup"),
          tool_args=tool_args,
          tool_context=tool_context,
      )
      is None
  )
  _patch_engine_error(monkeypatch)

  result = await plugin.after_tool_callback(
      tool=_tool("lookup"),
      tool_args=tool_args,
      tool_context=tool_context,
      result={"secret": "value"},
  )

  assert result is not None
  assert result["agent_hooks_blocked"] is True


async def test_output_fails_closed_on_engine_error(monkeypatch) -> None:
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  _patch_engine_error(monkeypatch)

  result = await plugin.on_event_callback(
      invocation_context=_invocation_context(), event=_final_event("unsafe")
  )

  assert result is not None
  assert result.content is not None
  assert result.content.parts[0].text == "[blocked by policy]"


# ---------------------------------------------------------------------------
# Real-runner / real-flow conformance. These drive ADK's Runner and function
# flow instead of calling callbacks directly, so they exercise what ADK does
# with each return value -- the seam where the ordering/enforcement bugs live.
# ---------------------------------------------------------------------------


def _event_text(event: Event) -> str:
  if event.content is None or not event.content.parts:
    return ""
  return "".join(part.text or "" for part in event.content.parts if part.text)


def _points(records: list[Any]) -> list[str]:
  return [record.interception_point.value for record in records]


async def _run_tool_flow(
    plugin: AgentHooksPlugin,
    tool: FunctionTool,
    *,
    args: dict[str, Any] | None = None,
) -> Any:
  """Drive ADK's real tool flow (before_tool -> tool -> after_tool/error)."""
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(name="agent", model=model, tools=[tool])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content="", plugins=[plugin]
  )
  function_call = types.FunctionCall(name=tool.name, args=args or {})
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=types.Content(parts=[types.Part(function_call=function_call)]),
  )
  return await handle_function_calls_async(
      invocation_context, event, {tool.name: tool}
  )


async def test_runner_input_deny_blocks_the_turn() -> None:
  # §6: a deny at input means the turn MUST NOT begin. ADK treats the returned
  # content as a replacement message, so without the latch the model still runs.
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_DenyInput()], record_sink=records.append
  )
  model = testing_utils.MockModel.create(responses=["MODEL SHOULD NOT RUN"])
  agent = Agent(name="root", model=model, tools=[])
  app = App(name="test_app", root_agent=agent, plugins=[plugin])
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async("please do something malicious")

  # The model was never dispatched and no later point was emitted.
  assert model.response_index == -1
  assert _points(records) == ["agent_startup", "input", "agent_shutdown"]
  # The caller sees a block, not a model answer.
  assert any("blocked" in _event_text(event) for event in events)


async def test_concurrent_callbacks_wait_for_single_startup() -> None:
  class _ObservedLock:

    def __init__(self) -> None:
      self._lock = asyncio.Lock()
      self.waiter_entered = asyncio.Event()

    async def __aenter__(self) -> _ObservedLock:
      if self._lock.locked():
        self.waiter_entered.set()
      await self._lock.acquire()
      return self

    async def __aexit__(self, *args: Any) -> None:
      self._lock.release()

  class _HoldStartup:

    def __init__(self) -> None:
      self.entered = asyncio.Event()
      self.release = asyncio.Event()

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "agent_startup":
        self.entered.set()
        await self.release.wait()
      return Verdict(decision=Decision.ALLOW)

  interceptor = _HoldStartup()
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[interceptor], record_sink=records.append
  )
  invocation_context = _invocation_context()
  state = plugin._state_for_invocation(invocation_context)
  observed_lock = _ObservedLock()
  state.startup_lock = observed_lock
  input_task = asyncio.create_task(
      plugin.on_user_message_callback(
          invocation_context=invocation_context,
          user_message=types.Content(
              role="user", parts=[types.Part.from_text(text="hello")]
          ),
      )
  )
  await interceptor.entered.wait()
  run_task = asyncio.create_task(
      plugin.before_run_callback(invocation_context=invocation_context)
  )

  await observed_lock.waiter_entered.wait()
  assert _points(records) == []
  interceptor.release.set()
  await asyncio.gather(input_task, run_task)

  assert _points(records) == ["agent_startup", "input"]


async def test_tool_deny_emits_no_post_tool_call() -> None:
  # §6.2: a pre_tool_call block must not emit the paired post_tool_call, even
  # though ADK still invokes after_tool_callback with the block result.
  records: list[Any] = []

  def delete_account(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True}

  plugin = AgentHooksPlugin(
      interceptors=[_DenyTool("delete_account")], record_sink=records.append
  )
  await _run_tool_flow(plugin, FunctionTool(delete_account))

  assert "pre_tool_call" in _points(records)
  assert "post_tool_call" not in _points(records)


async def test_tool_error_emits_one_paired_post_tool_call() -> None:
  # §3.1(5): a dispatched tool that raises still owes exactly one post_tool_call.
  class _CaptureToolError:

    def __init__(self) -> None:
      self.is_error: bool | None = None

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "post_tool_call":
        self.is_error = ctx["tool_result"]["is_error"]
      return Verdict(decision=Decision.ALLOW)

  interceptor = _CaptureToolError()
  records: list[Any] = []

  def failing_tool(**kwargs: Any) -> dict[str, Any]:
    raise ValueError("kaboom")

  plugin = AgentHooksPlugin(
      interceptors=[interceptor], record_sink=records.append
  )
  # The plugin records the paired post for audit and lets the error propagate.
  with pytest.raises(ValueError, match="kaboom"):
    await _run_tool_flow(plugin, FunctionTool(failing_tool))

  points = _points(records)
  assert points.count("pre_tool_call") == 1
  assert points.count("post_tool_call") == 1
  assert interceptor.is_error is True


async def test_runner_startup_deny_suppresses_output_and_errors_shutdown() -> (
    None
):
  # §6.1a: after a startup deny, no input/model/tool/output point may follow,
  # and agent_shutdown must record summary.reason == "error".
  class _DenyStartupCapture:

    def __init__(self) -> None:
      self.shutdown_reason: Any = None

    async def intercept(self, ctx: AgentContext) -> Any:
      point = ctx["interception_point"]
      if point == "agent_shutdown":
        self.shutdown_reason = ctx.get("summary", {}).get("reason")
      if point == "agent_startup":
        return Verdict.deny(reason="startup_denied")
      return Verdict(decision=Decision.ALLOW)

  interceptor = _DenyStartupCapture()
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[interceptor], record_sink=records.append
  )
  model = testing_utils.MockModel.create(responses=["MODEL SHOULD NOT RUN"])
  agent = Agent(name="root", model=model, tools=[])
  app = App(name="test_app", root_agent=agent, plugins=[plugin])
  runner = testing_utils.InMemoryRunner(app=app)

  await runner.run_async("hi")

  assert model.response_index == -1
  assert _points(records) == ["agent_startup", "agent_shutdown"]
  assert interceptor.shutdown_reason == "error"


async def test_run_cancellation_closes_audit_state() -> None:
  class _CaptureShutdown:

    def __init__(self) -> None:
      self.shutdown_reason: str | None = None

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "agent_shutdown":
        self.shutdown_reason = ctx["summary"]["reason"]
      return Verdict(decision=Decision.ALLOW)

  interceptor = _CaptureShutdown()
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[interceptor], record_sink=records.append
  )
  invocation_context = _invocation_context()
  await plugin.on_user_message_callback(
      invocation_context=invocation_context,
      user_message=types.Content(
          role="user", parts=[types.Part.from_text(text="hello")]
      ),
  )

  await plugin.on_run_cancelled_callback(invocation_context=invocation_context)

  assert invocation_context.run_nonce not in plugin._states
  assert _points(records) == ["agent_startup", "input", "agent_shutdown"]
  assert interceptor.shutdown_reason == "cancelled"


async def test_public_invocation_id_collision_isolated_by_host_nonce() -> None:
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )
  first = _invocation_context(
      invocation_id="shared", session_id="session-a", run_nonce="run-a"
  )
  second = _invocation_context(
      invocation_id="shared", session_id="session-b", run_nonce="run-b"
  )

  await asyncio.gather(
      plugin.before_run_callback(invocation_context=first),
      plugin.before_run_callback(invocation_context=second),
  )

  assert set(plugin._states) == {"run-a", "run-b"}
  assert {record.session_id for record in records} == {"run-a", "run-b"}
  await plugin.on_run_complete_callback(invocation_context=first)
  assert set(plugin._states) == {"run-b"}
  await plugin.on_run_complete_callback(invocation_context=second)
  assert plugin._states == {}


async def test_stale_copied_context_cannot_reopen_closed_run() -> None:
  agent = Agent(
      name="agent", model=testing_utils.MockModel.create(responses=["ok"])
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  stale_copy = invocation_context.model_copy()
  deep_copied_token = copy.deepcopy(invocation_context._run_identity_token)
  plugin = AgentHooksPlugin(interceptors=[_AllowAll()])
  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.on_run_complete_callback(invocation_context=invocation_context)

  assert (
      stale_copy._run_identity_token is invocation_context._run_identity_token
  )
  assert deep_copied_token is invocation_context._run_identity_token
  assert stale_copy._run_identity_token.closed is True
  with pytest.raises(RuntimeError, match="closed"):
    await plugin.before_run_callback(invocation_context=stale_copy)
  assert plugin._states == {}


async def test_fresh_resume_has_new_run_and_stable_trusted_lineage() -> None:
  class _CaptureLineage:

    def __init__(self) -> None:
      self.lineages: list[str] = []

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "agent_startup":
        self.lineages.append(ctx["extensions"]["adk"]["resume_lineage_id"])
      return Verdict(decision=Decision.ALLOW)

  interceptor = _CaptureLineage()
  plugin = AgentHooksPlugin(interceptors=[interceptor])
  first = _invocation_context(
      invocation_id="resume-id", session_id="session", run_nonce="run-one"
  )
  second = _invocation_context(
      invocation_id="resume-id", session_id="session", run_nonce="run-two"
  )

  await plugin.before_run_callback(invocation_context=first)
  await plugin.on_run_complete_callback(invocation_context=first)
  await plugin.before_run_callback(invocation_context=second)
  await plugin.on_run_complete_callback(invocation_context=second)

  assert len(set(interceptor.lineages)) == 1
  assert interceptor.lineages[0].startswith("rl-")


async def test_model_cancellation_emits_error_post_before_shutdown() -> None:
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )
  callback_context = _callback_context()
  assert (
      await plugin.before_model_callback(
          callback_context=callback_context,
          llm_request=LlmRequest(model="ctk-model"),
      )
      is None
  )
  invocation_context = _invocation_context()

  await plugin.on_run_cancelled_callback(invocation_context=invocation_context)

  assert _points(records) == [
      "pre_model_call",
      "post_model_call",
      "agent_shutdown",
  ]
  assert records[-2].verdict.decision == Decision.ALLOW
  assert plugin._states == {}


async def test_tool_cancellation_emits_error_post_before_shutdown() -> None:
  class _CaptureCancelledTool:

    def __init__(self) -> None:
      self.is_error: bool | None = None

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "post_tool_call":
        self.is_error = ctx["tool_result"]["is_error"]
      return Verdict(decision=Decision.ALLOW)

  interceptor = _CaptureCancelledTool()
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[interceptor], record_sink=records.append
  )
  assert (
      await plugin.before_tool_callback(
          tool=_tool("lookup"),
          tool_args={"x": 1},
          tool_context=_tool_context(),
      )
      is None
  )

  await plugin.on_run_cancelled_callback(
      invocation_context=_invocation_context()
  )

  assert _points(records) == [
      "pre_tool_call",
      "post_tool_call",
      "agent_shutdown",
  ]
  assert interceptor.is_error is True
  assert plugin._states == {}


async def test_model_tool_fanout_is_blocked_before_task_creation() -> None:
  executed: list[str] = []

  def first_tool() -> dict[str, bool]:
    executed.append("first")
    return {"ok": True}

  def second_tool() -> dict[str, bool]:
    executed.append("second")
    return {"ok": True}

  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()],
      max_tool_calls_per_response=1,
      record_sink=records.append,
  )
  model = testing_utils.MockModel.create(
      responses=[
          LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id="first", name="first_tool", args={}
                          )
                      ),
                      types.Part(
                          function_call=types.FunctionCall(
                              id="second", name="second_tool", args={}
                          )
                      ),
                  ],
              )
          )
      ]
  )
  agent = Agent(
      name="root",
      model=model,
      tools=[FunctionTool(first_tool), FunctionTool(second_tool)],
  )
  runner = testing_utils.InMemoryRunner(
      app=App(name="test_app", root_agent=agent, plugins=[plugin])
  )

  await runner.run_async("run both")

  assert executed == []
  post_model = next(
      record
      for record in records
      if record.interception_point.value == "post_model_call"
  )
  assert post_model.verdict.reason == "host_error:context_invalid"


async def test_parallel_model_capacity_is_reserved_before_policy_await() -> (
    None
):
  class _ObservedLock:

    def __init__(self) -> None:
      self._lock = asyncio.Lock()
      self.waiter_entered = asyncio.Event()

    async def __aenter__(self) -> _ObservedLock:
      if self._lock.locked():
        self.waiter_entered.set()
      await self._lock.acquire()
      return self

    async def __aexit__(self, *args: Any) -> None:
      self._lock.release()

  class _HoldFirstModel:

    def __init__(self) -> None:
      self.entered = asyncio.Event()
      self.release = asyncio.Event()

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "pre_model_call":
        self.entered.set()
        await self.release.wait()
      return Verdict(decision=Decision.ALLOW)

  interceptor = _HoldFirstModel()
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[interceptor],
      max_tracked_model_calls=1,
      record_sink=records.append,
  )
  first_context = _callback_context()
  second_context = _callback_context()
  state = plugin._state_for_context(first_context)
  observed_lock = _ObservedLock()
  state.emission_lock = observed_lock
  first_task = asyncio.create_task(
      plugin.before_model_callback(
          callback_context=first_context,
          llm_request=LlmRequest(model="first"),
      )
  )
  await interceptor.entered.wait()
  second_task = asyncio.create_task(
      plugin.before_model_callback(
          callback_context=second_context,
          llm_request=LlmRequest(model="second"),
      )
  )
  await observed_lock.waiter_entered.wait()

  assert len(state.open_model_calls) == 1
  interceptor.release.set()
  first_result, second_result = await asyncio.gather(first_task, second_task)

  assert first_result is None
  assert isinstance(second_result, LlmResponse)
  assert records[-1].verdict.reason == "host_error:context_invalid"


async def test_cancelled_model_policy_rolls_back_reservation() -> None:
  class _HoldModel:

    def __init__(self) -> None:
      self.entered = asyncio.Event()

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "pre_model_call":
        self.entered.set()
        await asyncio.Event().wait()
      return Verdict(decision=Decision.ALLOW)

  interceptor = _HoldModel()
  plugin = AgentHooksPlugin(interceptors=[interceptor])
  callback_context = _callback_context()
  state = plugin._state_for_context(callback_context)
  task = asyncio.create_task(
      plugin.before_model_callback(
          callback_context=callback_context,
          llm_request=LlmRequest(model="model"),
      )
  )
  await interceptor.entered.wait()
  assert len(state.open_model_calls) == 1
  task.cancel()

  with pytest.raises(asyncio.CancelledError):
    await task
  assert state.open_model_calls == {}


async def test_runner_aclose_records_cancelled_shutdown() -> None:
  class _StreamingAgent(BaseAgent):

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      yield Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          partial=True,
          content=types.Content(
              role="model", parts=[types.Part.from_text(text="partial")]
          ),
      )
      await asyncio.Event().wait()

    async def _run_live_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      if False:
        yield Event(author=self.name, invocation_id=ctx.invocation_id)

  class _CaptureShutdown:

    def __init__(self) -> None:
      self.reason: str | None = None

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "agent_shutdown":
        self.reason = ctx["summary"]["reason"]
      return Verdict(decision=Decision.ALLOW)

  interceptor = _CaptureShutdown()
  plugin = AgentHooksPlugin(interceptors=[interceptor])
  app = App(
      name="test_app",
      root_agent=_StreamingAgent(name="streaming"),
      plugins=[plugin],
  )
  runner = AdkInMemoryRunner(app=app)
  session = await runner.session_service.create_session(
      app_name="test_app", user_id="user"
  )
  events = runner.run_async(
      user_id="user",
      session_id=session.id,
      new_message=types.Content(
          role="user", parts=[types.Part.from_text(text="start")]
      ),
  )

  event = await anext(events)
  assert event.partial is True
  await events.aclose()

  assert interceptor.reason == "cancelled"
  assert plugin._states == {}


async def test_later_after_run_failure_records_error_shutdown() -> None:
  class _CaptureShutdown:

    def __init__(self) -> None:
      self.reason: str | None = None

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "agent_shutdown":
        self.reason = ctx["summary"]["reason"]
      return Verdict(decision=Decision.ALLOW)

  class _FailAfterRun(BasePlugin):

    def __init__(self) -> None:
      super().__init__("fail_after_run")

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
      raise RuntimeError("cleanup failed")

  interceptor = _CaptureShutdown()
  plugin = AgentHooksPlugin(interceptors=[interceptor])
  agent = Agent(
      name="root", model=testing_utils.MockModel.create(responses=["done"])
  )
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=agent,
          plugins=[plugin, _FailAfterRun()],
      )
  )

  with pytest.raises(RuntimeError, match="cleanup failed"):
    await runner.run_async("hello")

  assert interceptor.reason == "error"
  assert plugin._states == {}


async def test_legacy_setup_failure_closes_open_lifecycle(monkeypatch) -> None:
  class _NoopAgent(BaseAgent):

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      if False:
        yield Event(author=self.name, invocation_id=ctx.invocation_id)

    async def _run_live_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      if False:
        yield Event(author=self.name, invocation_id=ctx.invocation_id)

  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )
  runner = AdkInMemoryRunner(
      app=App(
          name="test_app",
          root_agent=_NoopAgent(name="noop"),
          plugins=[plugin],
      )
  )
  session = await runner.session_service.create_session(
      app_name="test_app", user_id="user"
  )

  async def fail_append(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("append failed")

  monkeypatch.setattr(runner, "_append_new_message_to_session", fail_append)

  with pytest.raises(RuntimeError, match="append failed"):
    _ = [
        event
        async for event in runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="hello")]
            ),
        )
    ]

  assert _points(records) == ["agent_startup", "input", "agent_shutdown"]
  assert plugin._states == {}


async def test_completion_cancellation_keeps_committed_outcome() -> None:
  class _HoldShutdown:

    def __init__(self) -> None:
      self.entered = asyncio.Event()
      self.release = asyncio.Event()
      self.reason: str | None = None

    async def intercept(self, ctx: AgentContext) -> Any:
      if ctx["interception_point"] == "agent_shutdown":
        self.reason = ctx["summary"]["reason"]
        self.entered.set()
        await self.release.wait()
      return Verdict(decision=Decision.ALLOW)

  interceptor = _HoldShutdown()
  plugin = AgentHooksPlugin(interceptors=[interceptor])
  invocation_context = _invocation_context(invocation_id="complete")
  await plugin.before_run_callback(invocation_context=invocation_context)
  completion = asyncio.create_task(
      plugin.on_run_complete_callback(invocation_context=invocation_context)
  )
  await interceptor.entered.wait()
  completion.cancel()
  interceptor.release.set()

  with pytest.raises(asyncio.CancelledError):
    await completion
  assert interceptor.reason == "completed"
  assert plugin._states == {}


async def test_runner_model_error_emits_paired_post_model_call() -> None:
  # §3.1(4): a dispatched model call that raises still owes one post_model_call.
  records: list[Any] = []
  plugin = AgentHooksPlugin(
      interceptors=[_AllowAll()], record_sink=records.append
  )
  model = testing_utils.MockModel.create(
      responses=[], error=ValueError("model boom")
  )
  agent = Agent(name="root", model=model, tools=[])
  app = App(name="test_app", root_agent=agent, plugins=[plugin])
  runner = testing_utils.InMemoryRunner(app=app)

  with pytest.raises(ValueError, match="model boom"):
    await runner.run_async("hi")

  points = _points(records)
  assert points.count("pre_model_call") == 1
  assert points.count("post_model_call") == 1
