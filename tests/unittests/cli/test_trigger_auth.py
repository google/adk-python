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

from collections.abc import Awaitable
from collections.abc import Callable
import logging
from pathlib import Path
from unittest.mock import MagicMock

import click
from fastapi import Request
from google.adk.cli import api_server as api_server_module
from google.adk.cli import cli_deploy
from google.adk.cli.api_server import ApiServer
import pytest


def _make_server(
    *,
    trigger_sources: list[str] | None = None,
    trigger_oidc_audience: str | None = None,
    trigger_oidc_service_accounts: list[str] | None = None,
    trigger_auth_verifier: (
        Callable[[Request], None | Awaitable[None]] | None
    ) = None,
) -> ApiServer:
  return ApiServer(
      agent_loader=MagicMock(),
      session_service=MagicMock(),
      memory_service=MagicMock(),
      artifact_service=MagicMock(),
      credential_service=MagicMock(),
      eval_sets_manager=MagicMock(),
      eval_set_results_manager=MagicMock(),
      agents_dir='/tmp',
      trigger_sources=trigger_sources,
      trigger_oidc_audience=trigger_oidc_audience,
      trigger_oidc_service_accounts=trigger_oidc_service_accounts,
      trigger_auth_verifier=trigger_auth_verifier,
  )


def test_trigger_sources_requires_oidc_audience_or_verifier() -> None:
  with pytest.raises(
      ValueError, match='trigger_sources requires trigger_oidc_audience'
  ):
    _make_server(trigger_sources=['pubsub'])


def test_trigger_sources_warns_with_oidc_audience_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
  caplog.set_level(logging.WARNING, logger=api_server_module.logger.name)
  server = _make_server(
      trigger_sources=['pubsub'],
      trigger_oidc_audience='my-audience',
  )
  assert server.trigger_sources == ['pubsub']
  assert server.trigger_oidc_audience == 'my-audience'
  assert (
      'trigger_oidc_audience is set without trigger_oidc_service_accounts'
      in caplog.text
  )


def test_trigger_sources_ok_with_oidc_audience_and_service_accounts(
    caplog: pytest.LogCaptureFixture,
) -> None:
  caplog.set_level(logging.WARNING, logger=api_server_module.logger.name)
  server = _make_server(
      trigger_sources=['pubsub'],
      trigger_oidc_audience='my-audience',
      trigger_oidc_service_accounts=['sa@project.iam.gserviceaccount.com'],
  )
  assert server.trigger_sources == ['pubsub']
  assert server.trigger_oidc_audience == 'my-audience'
  assert server.trigger_oidc_service_accounts == [
      'sa@project.iam.gserviceaccount.com'
  ]
  assert (
      'trigger_oidc_audience is set without trigger_oidc_service_accounts'
      not in caplog.text
  )


def test_service_accounts_without_audience_raises_when_triggers_unset() -> None:
  with pytest.raises(
      ValueError,
      match='trigger_oidc_service_accounts requires trigger_oidc_audience',
  ):
    _make_server(
        trigger_oidc_service_accounts=['sa@project.iam.gserviceaccount.com']
    )


def test_trigger_sources_ok_with_auth_verifier() -> None:
  server = _make_server(
      trigger_sources=['eventarc'],
      trigger_auth_verifier=lambda req: None,
  )
  assert server.trigger_sources == ['eventarc']
  assert server.trigger_auth_verifier is not None


def test_to_cloud_run_requires_oidc_audience_when_trigger_sources_set(
    tmp_path: Path,
) -> None:
  with pytest.raises(
      click.UsageError,
      match='--trigger_oidc_audience is required when --trigger_sources is set',
  ):
    cli_deploy.to_cloud_run(
        agent_folder=str(tmp_path),
        project='proj',
        region='us-central1',
        service_name='svc',
        app_name='app',
        temp_folder=str(tmp_path / 'tmp'),
        port=8080,
        trace_to_cloud=False,
        otel_to_cloud=False,
        with_ui=False,
        log_level='info',
        verbosity='info',
        adk_version='1.3.0',
        trigger_sources='pubsub',
    )


def test_to_agent_engine_requires_oidc_audience_when_trigger_sources_set(
    tmp_path: Path,
) -> None:
  with pytest.raises(
      click.UsageError,
      match='--trigger_oidc_audience is required when --trigger_sources is set',
  ):
    cli_deploy.to_agent_engine(
        agent_folder=str(tmp_path),
        trigger_sources='pubsub',
    )


def test_to_gke_requires_oidc_audience_when_trigger_sources_set(
    tmp_path: Path,
) -> None:
  with pytest.raises(
      click.UsageError,
      match='--trigger_oidc_audience is required when --trigger_sources is set',
  ):
    cli_deploy.to_gke(
        agent_folder=str(tmp_path),
        project='proj',
        region='us-central1',
        cluster_name='cluster',
        service_name='svc',
        app_name='app',
        temp_folder=str(tmp_path / 'tmp'),
        port=8080,
        trace_to_cloud=False,
        otel_to_cloud=False,
        with_ui=False,
        log_level='info',
        adk_version='1.3.0',
        trigger_sources='pubsub',
    )


def test_deploy_warns_when_oidc_audience_set_without_service_accounts(
    caplog: pytest.LogCaptureFixture,
) -> None:
  caplog.set_level(logging.WARNING, logger=cli_deploy.logger.name)
  cli_deploy._validate_trigger_options(
      trigger_sources='pubsub',
      trigger_oidc_audience='https://my-service.run.app',
      trigger_oidc_service_accounts=None,
  )
  assert (
      '--trigger_oidc_audience is set without --trigger_oidc_service_accounts'
      in caplog.text
  )


def test_deploy_rejects_service_accounts_without_audience() -> None:
  with pytest.raises(
      click.UsageError,
      match=(
          '--trigger_oidc_service_accounts requires --trigger_oidc_audience'
          ' to be set'
      ),
  ):
    cli_deploy._validate_trigger_options(
        trigger_sources=None,
        trigger_oidc_audience=None,
        trigger_oidc_service_accounts='sa@project.iam.gserviceaccount.com',
    )
