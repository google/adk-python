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

from datetime import datetime
from datetime import timezone
from types import SimpleNamespace

from google.adk.integrations.taskmarket import TaskMarketToolset
import httpx


async def test_create_requires_explicit_confirmation_before_any_preflight():
  toolset = TaskMarketToolset()

  result = await toolset.create_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
      confirmation_token="not-used",
  )

  assert result == {
      "error": (
          "Creation is confirmation-gated. Review preview_task first, "
          "then call create_task with confirm=true."
      ),
      "retry": False,
  }


async def test_confirmed_create_uses_the_reviewed_preview_once(monkeypatch):
  now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
  toolset = TaskMarketToolset(clock=lambda: now)
  preview = await toolset.preview_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
      mode="bounty",
      tags=["integration"],
  )
  cli_calls = []

  def fake_preflight(maximum_spend):
    assert str(maximum_spend) == "0.538500"
    return SimpleNamespace(succeeded=True, data={}, error=None)

  def fake_run_cli(args, is_write):
    cli_calls.append((args, is_write))
    return SimpleNamespace(
        succeeded=True,
        data={"taskId": "0x1234"},
        error=None,
        ambiguous=False,
    )

  monkeypatch.setattr(toolset, "_preflight_cli", fake_preflight)
  monkeypatch.setattr(toolset, "_run_cli", fake_run_cli)

  result = await toolset.create_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
      mode="bounty",
      tags=["integration"],
      confirmation_token=preview["confirmationToken"],
      confirm=True,
  )

  assert result == {
      "taskId": "0x1234",
      "taskUrl": "https://api.taskmarket.dev/api/tasks/0x1234",
      "status": "created",
      "retry": False,
  }
  assert len(cli_calls) == 1
  assert cli_calls[0][1] is True
  assert cli_calls[0][0][:2] == ["task", "create"]

  second_result = await toolset.create_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
      mode="bounty",
      tags=["integration"],
      confirmation_token=preview["confirmationToken"],
      confirm=True,
  )
  assert second_result["retry"] is False
  assert "missing or has already been used" in second_result["error"]


async def test_confirmed_create_rejects_arguments_that_changed_after_preview(
    monkeypatch,
):
  toolset = TaskMarketToolset(
      clock=lambda: datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
  )
  preview = await toolset.preview_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
  )

  def fail_if_called(_maximum_spend):
    raise AssertionError("a changed request must not reach preflight")

  monkeypatch.setattr(toolset, "_preflight_cli", fail_if_called)
  result = await toolset.create_task(
      description="Publish an unrelated integration PR",
      reward_usdc="0.5",
      duration_hours=24,
      confirmation_token=preview["confirmationToken"],
      confirm=True,
  )

  assert result["retry"] is False
  assert "differ from the reviewed preview" in result["error"]


async def test_create_stops_when_wallet_preflight_fails(monkeypatch):
  toolset = TaskMarketToolset(
      clock=lambda: datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
  )
  preview = await toolset.preview_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
  )
  cli_calls = []

  monkeypatch.setattr(
      toolset,
      "_preflight_cli",
      lambda _maximum_spend: SimpleNamespace(
          succeeded=False,
          data=None,
          error="Insufficient USDC balance.",
      ),
  )
  monkeypatch.setattr(
      toolset,
      "_run_cli",
      lambda *args: cli_calls.append(args),
  )

  result = await toolset.create_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
      confirmation_token=preview["confirmationToken"],
      confirm=True,
  )

  assert result == {
      "error": "Insufficient USDC balance.",
      "retry": False,
      "status": "blocked",
  }
  assert cli_calls == []


async def test_create_does_not_retry_an_ambiguous_cli_write(monkeypatch):
  toolset = TaskMarketToolset(
      clock=lambda: datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
  )
  preview = await toolset.preview_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
  )
  write_calls = []

  monkeypatch.setattr(
      toolset,
      "_preflight_cli",
      lambda _maximum_spend: SimpleNamespace(
          succeeded=True,
          data={},
          error=None,
      ),
  )

  def ambiguous_write(args, is_write):
    write_calls.append((args, is_write))
    return SimpleNamespace(
        succeeded=False,
        data=None,
        error="Taskmarket CLI timed out.",
        ambiguous=True,
    )

  monkeypatch.setattr(toolset, "_run_cli", ambiguous_write)
  result = await toolset.create_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
      confirmation_token=preview["confirmationToken"],
      confirm=True,
  )

  assert result == {
      "error": "Taskmarket CLI timed out.",
      "retry": False,
      "status": "unknown",
  }
  assert len(write_calls) == 1
  assert write_calls[0][1] is True


async def test_read_tools_use_the_public_api_and_return_live_json(monkeypatch):
  requests = []

  def handler(request):
    requests.append(request)
    return httpx.Response(
        200,
        json={"tasks": [{"id": "0x1234", "status": "open"}]},
    )

  transport = httpx.MockTransport(handler)
  real_async_client = httpx.AsyncClient

  def client_factory(**kwargs):
    return real_async_client(transport=transport, **kwargs)

  monkeypatch.setattr(httpx, "AsyncClient", client_factory)
  toolset = TaskMarketToolset()

  result = await toolset.list_tasks(
      status="open",
      mode="bounty",
      tags=["integration", "taskmarket"],
      limit=3,
  )

  assert result == {"tasks": [{"id": "0x1234", "status": "open"}]}
  assert len(requests) == 1
  assert str(requests[0].url) == (
      "https://api.taskmarket.dev/api/tasks?status=open&limit=3&"
      "mode=bounty&tags=integration%2Ctaskmarket"
  )


async def test_create_checks_base_usdc_and_balance_before_the_cli_write(
    monkeypatch,
):
  toolset = TaskMarketToolset(
      clock=lambda: datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
  )
  preview = await toolset.preview_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
  )
  calls = []

  monkeypatch.setattr(
      "google.adk.integrations.taskmarket._taskmarket_toolset.shutil.which",
      lambda _path: "/usr/local/bin/taskmarket",
  )

  def fake_run_cli(args, is_write):
    calls.append((args, is_write))
    if args == ["deposit"]:
      return SimpleNamespace(
          succeeded=True,
          data={
              "network": "Base",
              "chainId": 8453,
              "currency": "USDC",
              "usdcContract": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
          },
          error=None,
      )
    if args == ["stats"]:
      return SimpleNamespace(
          succeeded=True,
          data={"balanceUsdc": "0.538500"},
          error=None,
      )
    return SimpleNamespace(
        succeeded=True,
        data={"taskId": "0x5678"},
        error=None,
        ambiguous=False,
    )

  monkeypatch.setattr(toolset, "_run_cli", fake_run_cli)
  result = await toolset.create_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
      confirmation_token=preview["confirmationToken"],
      confirm=True,
  )

  assert result["taskId"] == "0x5678"
  assert calls[0] == (["deposit"], False)
  assert calls[1] == (["stats"], False)
  assert calls[2][0][:2] == ["task", "create"]
  assert calls[2][1] is True


async def test_preview_shows_the_exact_requester_authorization_record():
  toolset = TaskMarketToolset()

  preview = await toolset.preview_task(
      description="Review an integration PR",
      reward_usdc="0.5",
      duration_hours=24,
      mode="bounty",
      tags=["integration"],
  )

  assert preview["description"] == "Review an integration PR"
  assert preview["rewardUsdc"] == "0.500000"
  assert preview["durationHours"] == "24"
  assert preview["mode"] == "bounty"
  assert preview["tags"] == ["integration"]
  assert preview["network"] == "Base"
  assert preview["chainId"] == 8453
  assert preview["currency"] == "USDC"
  assert preview["usdcContract"] == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
  assert preview["maximumSpendUsdc"] == "0.538500"
  assert preview["confirmationToken"]
  assert preview["deadline"]
