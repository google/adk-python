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

"""Approval-threshold guard tests for the a2a_human_in_loop sample.

Rule under test: contributing/samples/a2a/a2a_human_in_loop/approval_config.py
and the `reimburse()` tools in agent.py / remote_a2a/human_in_loop/agent.py.
The approval threshold that decides whether a reimbursement needs manager
confirmation must (a) be resolved from server-side config only -- a caller
has no tool-call argument through which to raise or lower it -- and (b) a
`reimburse()` call for an amount at or above that threshold must not be able
to bypass manager confirmation via a direct call to the tool function, i.e.
there is no code path to "status": "ok" for a large amount other than
through a confirmed `ToolConfirmation`.

Both `agent.py` (the local root agent) and `remote_a2a/human_in_loop/agent.py`
(the remote A2A approval agent) define their own copy of `reimburse()` --
mirroring the sample's own pre-existing duplication of that function -- so
this test module parametrizes over both to cover both copies.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
from unittest.mock import MagicMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.session import Session
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
import pytest

# ToolConfirmation is gated behind an experimental feature flag, which emits a
# UserWarning on use; expected and not under test here (same pattern as
# tests/unittests/tools/test_tool_confirmation.py).
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

_CONTRIBUTING_DIR = Path(__file__).parent.parent.parent / "contributing"
_SAMPLE_DIR = _CONTRIBUTING_DIR / "samples" / "a2a" / "a2a_human_in_loop"
_REMOTE_SAMPLE_DIR = _SAMPLE_DIR / "remote_a2a" / "human_in_loop"


def _mock_tool_context(*, function_call_id: str = "test-call-1") -> ToolContext:
  """Builds a minimal ToolContext, following test_function_tool.py's pattern."""
  mock_invocation_context = MagicMock(spec=InvocationContext)
  mock_invocation_context._state_schema = None
  mock_invocation_context.session = MagicMock(spec=Session)
  mock_invocation_context.session.state = MagicMock()
  return ToolContext(
      invocation_context=mock_invocation_context,
      function_call_id=function_call_id,
  )


def _import_agent_module(sample_dir: Path):
  """Imports `<sample_dir>.agent` the way tests/unittests/test_samples.py's
  `_load_root_agent` loads a sample: put the sample's *parent* directory on
  sys.path (so `agent.py`'s own `from .approval_config import ...` relative
  import resolves against the sample's `__init__.py` package), import the
  dotted module path, and evict the sample's modules from sys.modules
  afterwards so re-importing with a different env var (see the env-var test
  below) picks up fresh module state rather than a cached one.
  """
  saved_modules = set(sys.modules)
  saved_path = list(sys.path)
  sys.path.insert(0, str(sample_dir.parent))
  try:
    module_name = f"{sample_dir.name}.agent"
    sys.modules.pop(module_name, None)
    sys.modules.pop(sample_dir.name, None)
    return importlib.import_module(module_name)
  finally:
    sys.path[:] = saved_path
    prefix = sample_dir.name
    for name in set(sys.modules) - saved_modules:
      if name == prefix or name.startswith(prefix + "."):
        del sys.modules[name]


@pytest.fixture(
    params=[_SAMPLE_DIR, _REMOTE_SAMPLE_DIR],
    ids=["root_agent", "remote_approval_agent"],
)
def agent_module(request, monkeypatch):
  monkeypatch.delenv("REIMBURSEMENT_APPROVAL_THRESHOLD_USD", raising=False)
  return _import_agent_module(request.param)


def test_reimburse_signature_has_no_threshold_argument(agent_module) -> None:
  """A caller cannot name a threshold override: no such parameter exists.

  This is the first half of the config-derived-field guard: the only inputs
  `reimburse()` accepts are `purpose`, `amount`, and `tool_context` -- there
  is no `threshold`/`approval_threshold_usd`/similar parameter a caller could
  set to change what counts as "requires approval".
  """
  params = set(inspect.signature(agent_module.reimburse).parameters)
  assert params == {"purpose", "amount", "tool_context"}


def test_threshold_is_read_from_config_not_from_any_call_argument(
    agent_module,
) -> None:
  """The guard always re-derives the threshold from config, every call."""
  default_threshold = agent_module.get_approval_threshold_usd()
  assert default_threshold == 100.0  # approval_config.DEFAULT_APPROVAL_THRESHOLD_USD

  tool_context = _mock_tool_context()
  # Passing arbitrary extra state on the tool_context that *looks* like an
  # attempted override must have no effect -- there is nothing in
  # requires_manager_approval()/get_approval_threshold_usd() that reads it.
  tool_context.state["approval_threshold_usd"] = 0
  tool_context.state["threshold"] = 0

  assert agent_module.get_approval_threshold_usd() == default_threshold
  assert agent_module.requires_manager_approval(default_threshold) is True


def test_large_reimbursement_cannot_bypass_approval_via_direct_call(
    agent_module,
) -> None:
  """A direct call for an amount at/above threshold never returns "ok".

  This is the "cannot bypass the approval step via a direct invocation"
  property: `reimburse()` is the *only* entry point (there is no second,
  ungated function to call instead), and calling it directly -- exactly the
  way a model that ignored its instructions, or a compromised/prompt-injected
  caller, would -- must not execute the reimbursement.
  """
  threshold = agent_module.get_approval_threshold_usd()
  tool_context = _mock_tool_context()

  result = agent_module.reimburse(
      purpose="new laptop", amount=threshold + 1, tool_context=tool_context
  )

  assert result["status"] != "ok"
  # A confirmation must actually have been requested, not merely refused.
  assert tool_context.actions.requested_tool_confirmations


def test_rejected_confirmation_still_refuses_execution(agent_module) -> None:
  threshold = agent_module.get_approval_threshold_usd()
  tool_context = _mock_tool_context()
  tool_context.tool_confirmation = ToolConfirmation(confirmed=False)

  result = agent_module.reimburse(
      purpose="new laptop", amount=threshold + 1, tool_context=tool_context
  )

  assert result["status"] == "rejected"


def test_confirmed_approval_allows_large_reimbursement(agent_module) -> None:
  """The gate is not a black hole: an explicitly confirmed call still works."""
  threshold = agent_module.get_approval_threshold_usd()
  tool_context = _mock_tool_context()
  tool_context.tool_confirmation = ToolConfirmation(confirmed=True)

  result = agent_module.reimburse(
      purpose="new laptop", amount=threshold + 1, tool_context=tool_context
  )

  assert result["status"] == "ok"


def test_small_amount_executes_without_any_confirmation(agent_module) -> None:
  threshold = agent_module.get_approval_threshold_usd()
  tool_context = _mock_tool_context()

  result = agent_module.reimburse(
      purpose="lunch", amount=threshold - 1, tool_context=tool_context
  )

  assert result["status"] == "ok"
  assert not tool_context.actions.requested_tool_confirmations


def test_deploy_time_env_var_moves_threshold_but_stays_server_side(
    monkeypatch,
) -> None:
  """Only a deployer's env var can move the threshold -- never a call arg."""
  monkeypatch.setenv("REIMBURSEMENT_APPROVAL_THRESHOLD_USD", "10")
  agent_mod = _import_agent_module(_SAMPLE_DIR)
  assert agent_mod.get_approval_threshold_usd() == 10.0

  tool_context = _mock_tool_context()
  result = agent_mod.reimburse(
      purpose="coffee", amount=20, tool_context=tool_context
  )
  assert result["status"] != "ok"
