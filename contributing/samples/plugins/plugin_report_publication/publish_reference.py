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

"""Publish only after two installed consumers admit a selected ADK reference.

The packet is the twelve-case Probity synthetic reference, not arbitrary ADK
tracing. The operator independently selects the interpreter, installed reader,
policy bytes and output path before invoking this sample. The installation is
trusted: -B prevents cache writes, but does not reject pre-existing bytecode.
Authenticate package source and refuse unselected caches before this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def _encode(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _environment() -> dict[str, str]:
  """Remove inherited Python configuration and provider credentials."""
  environment = {
      key: value
      for key, value in os.environ.items()
      if not key.startswith(("PYTHON", "PIP"))
      and not any(
          word in key.upper()
          for word in ("API_KEY", "TOKEN", "SECRET", "CREDENTIAL", "PASSWORD")
      )
  }
  environment["PYTHONDONTWRITEBYTECODE"] = "1"
  return environment


def _invoke(argv: list[str], output: Path, index: int) -> bytes:
  """Retain actual child bytes, including partial timeout output."""
  try:
    result = subprocess.run(
        argv,
        cwd=output,
        env=_environment(),
        capture_output=True,
        timeout=75,
        check=False,
    )
  except subprocess.TimeoutExpired as error:
    (output / f"gate-{index}.stdout").write_bytes(error.stdout or b"")
    (output / f"gate-{index}.stderr").write_bytes(error.stderr or b"")
    (output / f"gate-{index}.status.json").write_bytes(
        _encode({"timeout": True, "returncode": None})
    )
    raise ValueError("installed_gate_timeout") from error
  (output / f"gate-{index}.stdout").write_bytes(result.stdout)
  (output / f"gate-{index}.stderr").write_bytes(result.stderr)
  (output / f"gate-{index}.status.json").write_bytes(
      _encode({"timeout": False, "returncode": result.returncode})
  )
  if result.returncode != 0 or len(result.stdout) > 4096:
    raise ValueError("installed_gate_refused")
  return result.stdout


def _decision(raw: bytes) -> dict[str, Any]:
  """Require the selected installed gate's literal successful decision."""
  report = json.loads(raw)
  expected = {
      "profile": "probity-google-adk-ticket-v0",
      "publicationDecision": "admitted",
      "reason": "complete-selected-reference-population",
      "returncode": 0,
  }
  if not isinstance(report, dict) or set(report) != {*expected, "reportSha256"}:
    raise ValueError("installed_gate_schema")
  if _encode({key: report[key] for key in expected}) != _encode(expected):
    raise ValueError("installed_gate_refused")
  digest = report["reportSha256"]
  if not isinstance(digest, str) or len(digest) != 64:
    raise ValueError("reader_report_digest")
  if any(char not in "0123456789abcdef" for char in digest):
    raise ValueError("reader_report_digest")
  return report


def _freeze_policy(
    packet: Path,
    policy: Path,
    policy_sha: str,
    interpreter: Path,
    reader: Path,
    output: Path,
) -> tuple[Path, Path]:
  """Freeze checked host policy outside the packet before any invocation."""
  packet, policy = packet.resolve(), policy.resolve()
  raw = policy.read_bytes()
  if hashlib.sha256(raw).hexdigest() != policy_sha:
    raise ValueError("selected_policy_digest")
  if not interpreter.is_absolute() or not reader.is_absolute():
    raise ValueError("selected_installation_paths")
  if output.resolve().is_relative_to(packet):
    raise ValueError("receipt_inside_packet")
  output.mkdir(parents=True, exist_ok=False)
  frozen = output.resolve() / "selected-policy.json"
  frozen.write_bytes(raw)
  return packet, frozen


def publish(
    *,
    packet: Path,
    policy: Path,
    policy_sha: str,
    interpreter: Path,
    reader: Path,
    output: Path,
) -> dict[str, Any]:
  """Retain both gate receipts and refuse changes or failed admission."""
  packet, frozen = _freeze_policy(
      packet, policy, policy_sha, interpreter, reader, output
  )
  reports = []
  for index in range(2):
    argv = [
        str(interpreter),
        "-I",
        "-B",
        "-c",
        "from probity_adk.gate import main; main()",
        str(packet),
        "--reader",
        str(reader),
        "--policy",
        str(frozen),
        "--policy-sha256",
        policy_sha,
        "--output",
        str(output.resolve() / f"gate-{index}"),
    ]
    reports.append(_invoke(argv, output, index))
  if reports[0] != reports[1]:
    raise ValueError("repeated_decision_changed")
  decision = _decision(reports[0])
  receipt = {
      "profile": "probity-google-adk-ticket-v0",
      "publicationDecision": "admitted",
      "policySha256": policy_sha,
      "readerReportSha256": decision["reportSha256"],
      "readerInvocations": 2,
      "modelQuality": "not-evaluated",
      "outsideAdoption": False,
  }
  (output / "publication-receipt.json").write_bytes(_encode(receipt))
  return receipt


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("packet", type=Path)
  parser.add_argument("--policy", type=Path, required=True)
  parser.add_argument("--policy-sha256", required=True)
  parser.add_argument("--python", type=Path, required=True)
  parser.add_argument("--reader", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  receipt = publish(
      packet=args.packet,
      policy=args.policy,
      policy_sha=args.policy_sha256,
      interpreter=args.python,
      reader=args.reader,
      output=args.output,
  )
  print(_encode(receipt).decode())


if __name__ == "__main__":
  main()
