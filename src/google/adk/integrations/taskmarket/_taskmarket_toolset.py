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

"""Requester-facing Taskmarket tools.

The integration deliberately keeps public reads in Python and delegates the
funding write to the first-party ``taskmarket`` CLI. The CLI owns the wallet
keystore and signing flow; ADK never receives or stores wallet credentials.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from decimal import Decimal
from decimal import InvalidOperation
from decimal import ROUND_CEILING
import hashlib
import json
import math
import shutil
import subprocess
from typing import Any

import httpx
from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.function_tool import FunctionTool

DEFAULT_API_URL = "https://api.taskmarket.dev"
DEFAULT_CLI = "taskmarket"
BASE_CHAIN_ID = 8453
USDC_CONTRACT = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
PLATFORM_FEE_BPS = Decimal("750")
RELAY_FEE_USDC = Decimal("0.001")
USDC_QUANTUM = Decimal("0.000001")
PREVIEW_TTL = timedelta(minutes=15)
SUPPORTED_MODES = frozenset({"bounty", "claim", "pitch", "benchmark"})


@dataclass(frozen=True)
class _PendingPreview:
  request: dict[str, Any]
  deadline: datetime
  expires_at: datetime
  maximum_spend: Decimal


@dataclass(frozen=True)
class _CliResult:
  succeeded: bool
  data: Any
  error: str | None = None
  ambiguous: bool = False


class TaskMarketToolset(BaseToolset):
  """Expose a safe Taskmarket requester workflow to an ADK agent.

  Read tools use Taskmarket's public HTTP API. Task creation is a separate
  confirmation-gated tool: callers must first invoke ``preview_task`` and then
  pass the unchanged confirmation token with ``confirm=True``. Before the
  first-party CLI write, the integration checks its Base/USDC configuration and
  available balance. A failed or timed-out write is never retried automatically.

  Install the first-party CLI separately with
  ``npm install -g @lucid-agents/taskmarket``. Its keystore remains outside
  ADK and is never accepted as a constructor argument.
  """

  def __init__(
      self,
      *,
      api_url: str = DEFAULT_API_URL,
      cli_path: str = DEFAULT_CLI,
      timeout: float = 15.0,
      tool_filter: ToolPredicate | list[str] | None = None,
      tool_name_prefix: str | None = None,
      clock: Callable[[], datetime] | None = None,
  ):
    """Initialize the Taskmarket toolset.

    Args:
      api_url: Taskmarket API origin. Defaults to production.
      cli_path: Executable name or absolute path for the first-party CLI.
      timeout: Network and CLI timeout in seconds.
      tool_filter: Optional ADK tool filter.
      tool_name_prefix: Optional prefix applied by ``BaseToolset``.
      clock: Optional UTC clock, primarily useful for deterministic tests.
    """
    if not api_url.strip():
      raise ValueError("api_url must not be empty")
    if not math.isfinite(timeout) or timeout <= 0:
      raise ValueError("timeout must be a positive finite number")

    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
    )
    self.api_url = api_url.rstrip("/")
    self.cli_path = cli_path
    self.timeout = timeout
    self._clock = clock or (lambda: datetime.now(timezone.utc))
    self._pending_previews: dict[str, _PendingPreview] = {}
    self._tools = [
        FunctionTool(self.list_tasks),
        FunctionTool(self.get_task),
        FunctionTool(self.list_submissions),
        FunctionTool(self.preview_task),
        FunctionTool(self.create_task, require_confirmation=True),
    ]

  @override
  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Return the Taskmarket tools selected for the current context."""
    return [
        tool
        for tool in self._tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  async def list_tasks(
      self,
      status: str = "open",
      mode: str | None = None,
      tags: list[str] | None = None,
      limit: int = 20,
  ) -> Any:
    """List live Taskmarket tasks matching the optional filters."""
    if limit < 1 or limit > 100:
      return {"error": "limit must be between 1 and 100"}
    params: dict[str, str | int] = {"status": status, "limit": limit}
    if mode:
      params["mode"] = mode
    if tags:
      params["tags"] = ",".join(tag.strip() for tag in tags if tag.strip())
    return await self._get_json("/api/tasks", params=params)

  async def get_task(self, task_id: str) -> Any:
    """Retrieve the live status and details of a Taskmarket task."""
    validation_error = self._validate_task_id(task_id)
    if validation_error:
      return {"error": validation_error}
    return await self._get_json(f"/api/tasks/{task_id}")

  async def list_submissions(self, task_id: str) -> Any:
    """List submissions for human review without accepting or rejecting any."""
    validation_error = self._validate_task_id(task_id)
    if validation_error:
      return {"error": validation_error}
    return await self._get_json(f"/api/tasks/{task_id}/submissions")

  async def preview_task(
      self,
      description: str,
      reward_usdc: str,
      duration_hours: float,
      mode: str = "bounty",
      tags: list[str] | None = None,
  ) -> dict[str, Any]:
    """Prepare an exact, reviewable Taskmarket creation record.

    This method is read-only. It computes the deadline and a conservative
    maximum spend so the user can review the full request before any CLI write.
    The returned confirmation token is a digest of the exact record, not a
    wallet credential.
    """
    try:
      request = self._normalise_request(
          description=description,
          reward_usdc=reward_usdc,
          duration_hours=duration_hours,
          mode=mode,
          tags=tags,
      )
    except ValueError as exc:
      return {"error": str(exc)}

    now = self._utc_now()
    try:
      deadline = now + timedelta(hours=float(request["durationHours"]))
    except (OverflowError, ValueError) as exc:
      return {"error": f"duration_hours is too large: {exc}"}
    if deadline <= now:
      return {"error": "duration_hours must produce a future deadline"}

    deadline_text = _format_datetime(deadline)
    request["deadline"] = deadline_text
    maximum_spend = _maximum_spend(Decimal(request["rewardUsdc"]))
    request["maximumSpendUsdc"] = _format_usdc(maximum_spend)
    token = _confirmation_digest(request)
    expires_at = min(deadline, now + PREVIEW_TTL)
    self._pending_previews[token] = _PendingPreview(
        request=request,
        deadline=deadline,
        expires_at=expires_at,
        maximum_spend=maximum_spend,
    )

    return {
        **request,
        "network": "Base",
        "chainId": BASE_CHAIN_ID,
        "currency": "USDC",
        "usdcContract": USDC_CONTRACT,
        "confirmationToken": token,
        "expiresAt": _format_datetime(expires_at),
    }

  async def create_task(
      self,
      description: str,
      reward_usdc: str,
      duration_hours: float,
      confirmation_token: str,
      confirm: bool = False,
      mode: str = "bounty",
      tags: list[str] | None = None,
  ) -> dict[str, Any]:
    """Create a Taskmarket task only after preview and explicit confirmation.

    ``require_confirmation=True`` is set on the ADK tool wrapper. The boolean
    argument is an additional guard for direct callers and makes the intended
    authorization explicit in the public function schema.
    """
    if not confirm:
      return {
          "error": (
              "Creation is confirmation-gated. Review preview_task first, "
              "then call create_task with confirm=true."
          ),
          "retry": False,
      }
    if not confirmation_token:
      return {
          "error": "A confirmation_token from preview_task is required.",
          "retry": False,
      }

    pending = self._pending_previews.get(confirmation_token)
    if pending is None:
      return {
          "error": "The preview is missing or has already been used.",
          "retry": False,
      }
    now = self._utc_now()
    if now >= pending.expires_at or now >= pending.deadline:
      self._pending_previews.pop(confirmation_token, None)
      return {
          "error": "The preview expired. Run preview_task again.",
          "retry": False,
      }

    try:
      request = self._normalise_request(
          description=description,
          reward_usdc=reward_usdc,
          duration_hours=duration_hours,
          mode=mode,
          tags=tags,
      )
    except ValueError as exc:
      return {"error": str(exc), "retry": False}

    if request != {
        key: pending.request[key] for key in request if key in pending.request
    }:
      return {
          "error": (
              "The creation arguments differ from the reviewed preview. "
              "Run preview_task again and review the new record."
          ),
          "retry": False,
      }

    preflight = await asyncio.to_thread(
        self._preflight_cli, pending.maximum_spend
    )
    if not preflight.succeeded:
      return {
          "error": preflight.error or "Taskmarket preflight failed.",
          "retry": False,
          "status": "blocked",
      }

    remaining_hours = (pending.deadline - now).total_seconds() / 3600
    if remaining_hours <= 0:
      self._pending_previews.pop(confirmation_token, None)
      return {
          "error": "The reviewed deadline has passed. Run preview_task again.",
          "retry": False,
      }
    cli_args = [
        "task",
        "create",
        "--description",
        pending.request["description"],
        "--reward",
        pending.request["rewardUsdc"],
        "--duration",
        f"{remaining_hours:.9f}",
        "--mode",
        pending.request["mode"],
    ]
    if pending.request["tags"]:
      cli_args.extend(["--tags", ",".join(pending.request["tags"])])

    result = await asyncio.to_thread(self._run_cli, cli_args, True)
    if not result.succeeded:
      return {
          "error": (
              result.error
              or "Task creation failed; inspect live Taskmarket status before retrying."
          ),
          "retry": False,
          "status": "unknown" if result.ambiguous else "failed",
      }
    task_id = _task_id_from_cli_result(result.data)
    if not task_id:
      return {
          "error": (
              "The CLI returned no task ID. Inspect Taskmarket live status "
              "before retrying."
          ),
          "retry": False,
          "status": "unknown",
      }
    self._pending_previews.pop(confirmation_token, None)
    return {
        "taskId": task_id,
        "taskUrl": f"{self.api_url}/api/tasks/{task_id}",
        "status": "created",
        "retry": False,
    }

  async def _get_json(
      self,
      path: str,
      *,
      params: dict[str, str | int] | None = None,
  ) -> Any:
    try:
      async with httpx.AsyncClient(
          base_url=self.api_url,
          timeout=self.timeout,
          follow_redirects=False,
      ) as client:
        response = await client.get(path, params=params)
      response.raise_for_status()
      return response.json()
    except httpx.HTTPStatusError as exc:
      return {
          "error": (
              f"Taskmarket read failed with HTTP {exc.response.status_code}."
          ),
          "retry": True,
      }
    except (httpx.HTTPError, ValueError) as exc:
      return {"error": f"Taskmarket read failed: {exc}", "retry": True}

  def _preflight_cli(self, maximum_spend: Decimal) -> _CliResult:
    """Check wallet configuration and balance before any paid CLI command."""
    if shutil.which(self.cli_path) is None:
      return _CliResult(
          succeeded=False,
          data=None,
          error=(
              "The first-party taskmarket CLI was not found. Install "
              "@lucid-agents/taskmarket and retry."
          ),
      )

    deposit = self._run_cli(["deposit"], False)
    if not deposit.succeeded or not isinstance(deposit.data, dict):
      return _CliResult(
          succeeded=False,
          data=None,
          error=deposit.error or "Could not verify Taskmarket wallet network.",
      )
    expected = {
        "network": "Base",
        "chainId": BASE_CHAIN_ID,
        "currency": "USDC",
        "usdcContract": USDC_CONTRACT,
    }
    if any(deposit.data.get(key) != value for key, value in expected.items()):
      return _CliResult(
          succeeded=False,
          data=None,
          error="Taskmarket wallet is not configured for Base USDC.",
      )

    stats = self._run_cli(["stats"], False)
    if not stats.succeeded or not isinstance(stats.data, dict):
      return _CliResult(
          succeeded=False,
          data=None,
          error=stats.error or "Could not verify the available USDC balance.",
      )
    try:
      balance = Decimal(str(stats.data["balanceUsdc"]))
    except (KeyError, InvalidOperation, TypeError, ValueError):
      return _CliResult(
          succeeded=False,
          data=None,
          error="Taskmarket CLI returned an unreadable USDC balance.",
      )
    if not balance.is_finite() or balance < maximum_spend:
      return _CliResult(
          succeeded=False,
          data=None,
          error=(
              "Insufficient USDC balance for the reviewed maximum spend "
              f"({ _format_usdc(maximum_spend) } USDC)."
          ),
      )
    return _CliResult(succeeded=True, data=stats.data)

  def _run_cli(self, args: list[str], is_write: bool) -> _CliResult:
    """Run one first-party CLI command without shell interpolation or retries."""
    try:
      completed = subprocess.run(
          [self.cli_path, *args],
          capture_output=True,
          check=False,
          text=True,
          timeout=self.timeout,
      )
    except FileNotFoundError:
      return _CliResult(
          succeeded=False,
          data=None,
          error="The first-party taskmarket CLI was not found.",
      )
    except subprocess.TimeoutExpired:
      return _CliResult(
          succeeded=False,
          data=None,
          error=(
              "Taskmarket CLI timed out. Inspect live status before retrying; "
              "the command was not retried."
          ),
          ambiguous=is_write,
      )

    parsed: Any = None
    stdout = completed.stdout.strip()
    if stdout:
      try:
        parsed = json.loads(stdout)
      except json.JSONDecodeError:
        parsed = None
    if completed.returncode == 0 and isinstance(parsed, dict):
      if parsed.get("ok") is False:
        return _CliResult(
            succeeded=False,
            data=None,
            error="Taskmarket CLI rejected the command.",
            ambiguous=is_write,
        )
      return _CliResult(
          succeeded=True,
          data=parsed.get("data", parsed),
      )
    return _CliResult(
        succeeded=False,
        data=None,
        error="Taskmarket CLI rejected the command.",
        ambiguous=is_write,
    )

  def _normalise_request(
      self,
      *,
      description: str,
      reward_usdc: str,
      duration_hours: float,
      mode: str,
      tags: list[str] | None,
  ) -> dict[str, Any]:
    if not isinstance(description, str) or not description.strip():
      raise ValueError("description must not be empty")
    if mode not in SUPPORTED_MODES:
      raise ValueError(
          f"mode must be one of: {', '.join(sorted(SUPPORTED_MODES))}"
      )
    try:
      reward = Decimal(str(reward_usdc).strip())
    except (InvalidOperation, ValueError):
      raise ValueError("reward_usdc must be a positive USDC amount") from None
    if not reward.is_finite() or reward <= 0:
      raise ValueError("reward_usdc must be a positive USDC amount")
    exponent = reward.as_tuple().exponent
    if isinstance(exponent, str) or exponent < -6:
      raise ValueError("reward_usdc supports at most 6 decimal places")
    reward = reward.quantize(USDC_QUANTUM)

    try:
      duration = Decimal(str(duration_hours))
    except (InvalidOperation, ValueError):
      raise ValueError(
          "duration_hours must be a positive finite number"
      ) from None
    if not duration.is_finite() or duration <= 0:
      raise ValueError("duration_hours must be a positive finite number")
    clean_tags = [tag.strip() for tag in tags or [] if tag.strip()]
    return {
        "description": description.strip(),
        "rewardUsdc": _format_usdc(reward),
        "durationHours": _format_decimal(duration),
        "mode": mode,
        "tags": clean_tags,
    }

  def _utc_now(self) -> datetime:
    now = self._clock()
    if now.tzinfo is None:
      return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)

  @staticmethod
  def _validate_task_id(task_id: str) -> str | None:
    if not isinstance(task_id, str) or not task_id.startswith("0x"):
      return "task_id must be a 0x-prefixed Taskmarket task ID"
    return None


def _format_usdc(value: Decimal) -> str:
  return f"{value.quantize(USDC_QUANTUM):.6f}"


def _format_decimal(value: Decimal) -> str:
  text = format(value, "f")
  if "." in text:
    text = text.rstrip("0").rstrip(".")
  return text or "0"


def _format_datetime(value: datetime) -> str:
  return (
      value.astimezone(timezone.utc)
      .isoformat(timespec="seconds")
      .replace("+00:00", "Z")
  )


def _maximum_spend(reward: Decimal) -> Decimal:
  fee = reward * PLATFORM_FEE_BPS / Decimal("10000")
  return (reward + fee + RELAY_FEE_USDC).quantize(
      USDC_QUANTUM, rounding=ROUND_CEILING
  )


def _confirmation_digest(request: dict[str, Any]) -> str:
  encoded = json.dumps(
      request,
      ensure_ascii=False,
      separators=(",", ":"),
      sort_keys=True,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _task_id_from_cli_result(data: Any) -> str | None:
  if not isinstance(data, dict):
    return None
  task_id = data.get("taskId")
  if isinstance(task_id, str) and task_id.startswith("0x"):
    return task_id
  return None
