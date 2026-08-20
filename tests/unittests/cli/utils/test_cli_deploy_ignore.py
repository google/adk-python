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

"""Tests for ignore file support in cli_deploy."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from unittest import mock

import click
import pytest

import src.google.adk.cli.cli_deploy as cli_deploy


@pytest.fixture(autouse=True)
def _mute_click(monkeypatch: pytest.MonkeyPatch) -> None:
  """Suppress click.echo to keep test output clean."""
  monkeypatch.setattr(click, "echo", lambda *_a, **_k: None)
  monkeypatch.setattr(click, "secho", lambda *_a, **_k: None)


def test_to_cloud_run_respects_ignore_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  """Test that to_cloud_run respects .gitignore and .gcloudignore."""
  agent_dir = tmp_path / "agent"
  agent_dir.mkdir()
  (agent_dir / "agent.py").write_text("# agent")
  (agent_dir / "__init__.py").write_text("")
  (agent_dir / "ignored_by_git.txt").write_text("ignored")
  (agent_dir / "ignored_by_gcloud.txt").write_text("ignored")
  (agent_dir / "ignored_rooted.txt").write_text("ignored")
  (agent_dir / "not_ignored.txt").write_text("keep")

  # Use a root-anchored pattern (leading slash) to ensure it is honored.
  (agent_dir / ".gitignore").write_text(
      "ignored_by_git.txt\n/ignored_rooted.txt\n"
  )
  (agent_dir / ".gcloudignore").write_text("ignored_by_gcloud.txt\n")

  temp_deploy_dir = tmp_path / "temp_deploy"

  # Mock subprocess.run to avoid actual gcloud call
  monkeypatch.setattr(subprocess, "run", mock.Mock())
  # Mock shutil.rmtree to keep the temp folder for verification
  monkeypatch.setattr(
      shutil,
      "rmtree",
      lambda path, **kwargs: None
      if "temp_deploy" in str(path)
      else shutil.rmtree(path, **kwargs),
  )

  cli_deploy.to_cloud_run(
      agent_folder=str(agent_dir),
      project="proj",
      region="us-central1",
      service_name="svc",
      app_name="app",
      temp_folder=str(temp_deploy_dir),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      verbosity="info",
      adk_version="1.0.0",
  )

  agent_src_path = temp_deploy_dir / "agents" / "app"

  assert (agent_src_path / "agent.py").exists()
  assert (agent_src_path / "not_ignored.txt").exists()

  # These should be ignored
  assert not (
      agent_src_path / "ignored_by_git.txt"
  ).exists(), "Should respect .gitignore"
  assert not (
      agent_src_path / "ignored_by_gcloud.txt"
  ).exists(), "Should respect .gcloudignore"
  assert not (
      agent_src_path / "ignored_rooted.txt"
  ).exists(), "Should respect root-anchored (leading slash) patterns"


def test_to_agent_engine_respects_multiple_ignore_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  """Test that to_agent_engine respects .gitignore, .gcloudignore and .ae_ignore."""
  # We need to be in the project dir for to_agent_engine
  project_dir = tmp_path / "project"
  project_dir.mkdir()
  monkeypatch.chdir(project_dir)

  agent_dir = project_dir / "my_agent"
  agent_dir.mkdir()
  (agent_dir / "agent.py").write_text("root_agent = None")
  (agent_dir / "__init__.py").write_text("from . import agent")
  (agent_dir / "ignored_by_git.txt").write_text("ignored")
  (agent_dir / "ignored_by_ae.txt").write_text("ignored")

  (agent_dir / ".gitignore").write_text("ignored_by_git.txt\n")
  (agent_dir / ".ae_ignore").write_text("ignored_by_ae.txt\n")

  # Mock vertexai.Client and other things to avoid network/complex setup. The
  # created agent engine must expose a realistic resource name so the downstream
  # console-URL formatting does not choke on a bare Mock.
  mock_client = mock.Mock()
  mock_client.agent_engines.create.return_value.api_resource.name = (
      "projects/proj/locations/us-central1/reasoningEngines/123"
  )
  monkeypatch.setattr("vertexai.Client", mock.Mock(return_value=mock_client))
  # Mock shutil.rmtree to keep the temp folder for verification
  original_rmtree = shutil.rmtree

  def mock_rmtree(path, **kwargs):
    if "_tmp" in str(path):
      return None
    return original_rmtree(path, **kwargs)

  monkeypatch.setattr(shutil, "rmtree", mock_rmtree)

  cli_deploy.to_agent_engine(
      agent_folder=str(agent_dir),
      staging_bucket="gs://test",
      adk_app="adk_app",
      # Pass project/region explicitly so the function does not fall back to
      # the interactive `gcloud auth application-default login` onboarding flow,
      # which fails on CI runners without Application Default Credentials.
      project="proj",
      region="us-central1",
      adk_version="1.0.0",
  )

  # Find the temp folder created by to_agent_engine
  temp_folders = [
      d for d in project_dir.iterdir() if d.is_dir() and "_tmp" in d.name
  ]
  assert len(temp_folders) == 1
  agent_src_path = temp_folders[0]

  copied_agent_dir = agent_src_path / "agents" / "my_agent"
  assert (copied_agent_dir / "agent.py").exists()
  assert not (
      copied_agent_dir / "ignored_by_git.txt"
  ).exists(), "Should respect .gitignore"
  assert not (
      copied_agent_dir / "ignored_by_ae.txt"
  ).exists(), "Should respect .ae_ignore"


def test_to_gke_respects_ignore_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  """Test that to_gke respects ignore files."""
  agent_dir = tmp_path / "agent"
  agent_dir.mkdir()
  (agent_dir / "agent.py").write_text("# agent")
  (agent_dir / "__init__.py").write_text("")
  (agent_dir / "ignored.txt").write_text("ignored")
  (agent_dir / ".gitignore").write_text("ignored.txt\n")

  temp_deploy_dir = tmp_path / "temp_deploy"

  # Mock subprocess.run to avoid actual gcloud call
  mock_run = mock.Mock()
  mock_run.return_value.stdout = "deployment created"
  monkeypatch.setattr(subprocess, "run", mock_run)
  # Mock shutil.rmtree to keep the temp folder for verification
  monkeypatch.setattr(
      shutil,
      "rmtree",
      lambda path, **kwargs: None
      if "temp_deploy" in str(path)
      else shutil.rmtree(path, **kwargs),
  )

  cli_deploy.to_gke(
      agent_folder=str(agent_dir),
      project="proj",
      region="us-central1",
      cluster_name="cluster",
      service_name="svc",
      app_name="app",
      temp_folder=str(temp_deploy_dir),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      adk_version="1.0.0",
  )

  agent_src_path = temp_deploy_dir / "agents" / "app"

  assert (agent_src_path / "agent.py").exists()
  assert not (
      agent_src_path / "ignored.txt"
  ).exists(), "Should respect .gitignore"
