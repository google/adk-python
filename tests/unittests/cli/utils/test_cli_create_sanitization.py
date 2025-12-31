import os
from pathlib import Path

import click
import google.adk.cli.cli_create as cli_create
import pytest


def test_sanitize_to_python_identifier() -> None:
  """Tests robust sanitization rules."""
  # Hyphens replaced
  assert (
      cli_create._sanitize_to_python_identifier("sample-agent")
      == "sample_agent"
  )
  assert (
      cli_create._sanitize_to_python_identifier("my-cool-agent")
      == "my_cool_agent"
  )

  # Other chars replaced
  assert cli_create._sanitize_to_python_identifier("agent.name") == "agent_name"
  assert cli_create._sanitize_to_python_identifier("agent name") == "agent_name"
  assert cli_create._sanitize_to_python_identifier("agent@name") == "agent_name"

  # Leading digits handling
  assert cli_create._sanitize_to_python_identifier("1agent") == "_1agent"

  # Already valid
  assert (
      cli_create._sanitize_to_python_identifier("valid_agent") == "valid_agent"
  )
  assert cli_create._sanitize_to_python_identifier("_private") == "_private"


def test_run_cmd_sanitization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  """run_cmd should sanitize agent name and warn user."""
  monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

  # Mock prompt functions to avoid input requirement
  monkeypatch.setattr(
      cli_create, "_prompt_for_model", lambda: "gemini-2.5-flash"
  )
  monkeypatch.setattr(
      cli_create,
      "_prompt_to_choose_backend",
      lambda a, b, c: ("key", None, None),
  )
  monkeypatch.setattr(cli_create, "_prompt_to_choose_type", lambda: "code")

  # Mock generated files to avoid actual IO work and just check name
  monkeypatch.setattr(cli_create, "_generate_files", lambda *a, **k: None)

  messages = []
  monkeypatch.setattr(click, "secho", lambda msg, **k: messages.append(msg))

  cli_create.run_cmd(
      agent_name="bad-name",
      model="gemini-2.5-flash",
      google_api_key="key",
      google_cloud_project=None,
      google_cloud_region=None,
      type="code",
  )

  # Verify warning
  # The actual message is formatted with a break in line 299-300 of cli_create.py
  # We should check for substrings or reconstruct expectations
  expected_part = "Renaming to 'bad_name'"
  assert any(expected_part in msg for msg in messages), f"Messages: {messages}"
