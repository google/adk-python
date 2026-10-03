# Copyright 2026 Probity contributors
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

"""Exercise admission, retained failure and external policy boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location(
    "publication_sample",
    Path(__file__).resolve().parents[3]
    / "contributing/samples/plugins/plugin_report_publication/publish_reference.py",
)
assert spec is not None and spec.loader is not None
sample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample)


@pytest.fixture
def selected(tmp_path: Path) -> dict[str, Any]:
  packet = tmp_path / "packet"
  packet.mkdir()
  policy = tmp_path / "policy.json"
  policy.write_bytes(b'{"selected":"outside"}')
  return dict(
      packet=packet,
      policy=policy,
      policy_sha=hashlib.sha256(policy.read_bytes()).hexdigest(),
      interpreter=Path("/selected/bin/python"),
      reader=Path("/selected/bin/probity-adk-read"),
      output=tmp_path / "published",
  )


def good(**changes: Any) -> bytes:
  record = dict(
      profile="probity-google-adk-ticket-v0",
      publicationDecision="admitted",
      reason="complete-selected-reference-population",
      returncode=0,
      reportSha256="a" * 64,
  )
  record.update(changes)
  return json.dumps(record).encode()


def run_result(raw: bytes, code: int = 0) -> subprocess.CompletedProcess[bytes]:
  return subprocess.CompletedProcess([], code, raw, b"actual stderr")


def test_two_actual_child_receipts(
    selected: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
  commands = []

  def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    commands.append((argv, kwargs))
    return run_result(good())

  monkeypatch.setattr(sample.subprocess, "run", run)
  report = sample.publish(**selected)
  assert report["readerInvocations"] == 2
  assert report["outsideAdoption"] is False
  assert len(commands) == 2
  assert commands[0][0][1:4] == ["-I", "-B", "-c"]
  assert commands[0][1]["timeout"] == 75
  assert (selected["output"] / "selected-policy.json").read_bytes() == selected[
      "policy"
  ].read_bytes()
  assert (selected["output"] / "gate-1.stderr").read_bytes() == b"actual stderr"
  assert json.loads(
      (selected["output"] / "gate-1.status.json").read_bytes()
  ) == {
      "timeout": False,
      "returncode": 0,
  }
  assert (selected["output"] / "publication-receipt.json").is_file()


@pytest.mark.parametrize(
    "changes",
    [
        {"profile": "foreign"},
        {"publicationDecision": "refused"},
        {"reason": "different"},
        {"returncode": False},
        {"returncode": 1},
        {"extra": "field"},
        {"reportSha256": "g" * 64},
        {"reportSha256": "a" * 63},
        {"reportSha256": 123},
    ],
)
def test_typed_selected_decision(
    changes: dict[str, Any],
    selected: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
      sample.subprocess, "run", lambda *a, **k: run_result(good(**changes))
  )
  with pytest.raises(ValueError):
    sample.publish(**selected)
  assert not (selected["output"] / "publication-receipt.json").exists()


@pytest.mark.parametrize("raw", [b"[]", b"null", b"{", b'"admitted"'])
def test_invalid_child_json(
    raw: bytes, selected: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(sample.subprocess, "run", lambda *a, **k: run_result(raw))
  with pytest.raises((ValueError, TypeError)):
    sample.publish(**selected)
  assert not (selected["output"] / "publication-receipt.json").exists()


@pytest.mark.parametrize("failure_at", [0, 1])
def test_failed_reader_never_publishes(
    selected: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure_at: int
) -> None:
  calls = 0

  def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    nonlocal calls
    result = run_result(good(), 1 if calls == failure_at else 0)
    calls += 1
    return result

  monkeypatch.setattr(sample.subprocess, "run", run)
  with pytest.raises(ValueError, match="installed_gate_refused"):
    sample.publish(**selected)
  assert not (selected["output"] / "publication-receipt.json").exists()
  assert (selected["output"] / f"gate-{failure_at}.stdout").is_file()
  assert json.loads(
      (selected["output"] / f"gate-{failure_at}.status.json").read_bytes()
  ) == {"timeout": False, "returncode": 1}


def test_timeout_retains_partial_receipt(
    selected: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
  def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    raise subprocess.TimeoutExpired(
        argv, 75, output=b"partial", stderr=b"error"
    )

  monkeypatch.setattr(sample.subprocess, "run", run)
  with pytest.raises(ValueError, match="installed_gate_timeout"):
    sample.publish(**selected)
  assert (selected["output"] / "gate-0.stdout").read_bytes() == b"partial"
  assert json.loads(
      (selected["output"] / "gate-0.status.json").read_bytes()
  ) == {
      "timeout": True,
      "returncode": None,
  }
  assert not (selected["output"] / "publication-receipt.json").exists()


def test_literal_repeat_changes_refused(
    selected: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
  results = iter([good(), good() + b"\n"])
  monkeypatch.setattr(
      sample.subprocess,
      "run",
      lambda *args, **kwargs: run_result(next(results)),
  )
  with pytest.raises(ValueError, match="repeated_decision_changed"):
    sample.publish(**selected)
  assert not (selected["output"] / "publication-receipt.json").exists()


def test_policy_digest_checked_before_launch(
    selected: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(
      sample.subprocess, "run", lambda *args, **kwargs: pytest.fail("launched")
  )
  selected["policy"].write_bytes(b"changed")
  with pytest.raises(ValueError, match="selected_policy_digest"):
    sample.publish(**selected)
  assert not selected["output"].exists()


@pytest.mark.parametrize("field", ["interpreter", "reader"])
def test_relative_executable_refused(
    selected: dict[str, Any], field: str
) -> None:
  selected[field] = Path("relative")
  with pytest.raises(ValueError, match="selected_installation_paths"):
    sample.publish(**selected)


def test_receipt_inside_candidate_refused(selected: dict[str, Any]) -> None:
  selected["output"] = selected["packet"] / "receipt"
  with pytest.raises(ValueError, match="receipt_inside_packet"):
    sample.publish(**selected)


def test_environment_strips_python_and_credentials(
    selected: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  for key in [
      "PYTHONPATH",
      "PIP_CONFIG_FILE",
      "PROVIDER_API_KEY",
      "AUTH_TOKEN",
  ]:
    monkeypatch.setenv(key, "inherited")
  monkeypatch.setenv("SELECTED_PUBLIC_VALUE", "kept")
  environments = []

  def invoke(*a: Any, **k: Any) -> subprocess.CompletedProcess[bytes]:
    environments.append(k["env"])
    return run_result(good())

  monkeypatch.setattr(sample.subprocess, "run", invoke)
  sample.publish(**selected)
  environment = environments[0]
  assert environments[0] == environments[1]
  assert not any(
      key in environment
      for key in [
          "PYTHONPATH",
          "PIP_CONFIG_FILE",
          "PROVIDER_API_KEY",
          "AUTH_TOKEN",
      ]
  )
  assert environment["SELECTED_PUBLIC_VALUE"] == "kept"
  assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
