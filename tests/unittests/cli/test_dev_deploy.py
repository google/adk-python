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

"""Tests for the dev-only deploy endpoints in `dev_deploy`.

The deploy itself is a child `adk deploy` process, so these tests replace that
child with a short Python script that prints whatever output the case needs.
Nothing here contacts Google Cloud.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.adk.cli import _dev_deploy as dev_deploy
from google.adk.cli._dev_deploy import AgentEngineDeployRequest
from google.adk.cli._dev_deploy import CloudRunDeployRequest
from google.adk.cli._dev_deploy import GkeDeployRequest
import pydantic
import pytest

APP_NAME = 'my_agent'

# Output fragments as the real tools emit them.
AGENT_RUNTIME_OK = (
    'print("Deployed to Agent Platform:'
    ' projects/p/locations/us-central1/reasoningEngines/42")'
)
# gcloud emboldens the bracketed values; the escape codes must not survive
# into the scraped project name.
CLOUD_RUN_OK = (
    r'print("Building using Dockerfile and deploying container to Cloud Run'
    r' service [\x1b[1mmy-agent\x1b[m] in project [\x1b[1madk-demo\x1b[m]'
    r' region [\x1b[1mus-central1\x1b[m]")'
    '\n'
    r'print("Service URL: https://my-agent-abc123-uc.a.run.app")'
)


@pytest.fixture
def agents_dir(tmp_path):
  """A directory holding one agent, as `adk web` would serve."""
  (tmp_path / APP_NAME).mkdir()
  return tmp_path


@pytest.fixture
def client(agents_dir):
  app = FastAPI()
  dev_deploy.register_dev_deploy_endpoints(
      app, get_agent_dir=lambda name: str(agents_dir / name)
  )
  return TestClient(app)


@pytest.fixture(autouse=True)
def _isolate_module_state():
  """Keeps the module's process-wide state from leaking between tests."""
  dev_deploy._deploys_in_flight.clear()
  yield
  dev_deploy._deploys_in_flight.clear()


@pytest.fixture(autouse=True)
def _no_gcloud():
  """Stops the defaults lookup from shelling out to a real gcloud."""

  async def _none():
    return None

  with patch.object(dev_deploy, '_gcloud_project', _none):
    yield


@contextlib.contextmanager
def stub_child(script: str):
  """Replaces the `adk deploy` child with a script printing `script`."""

  def fake_argv(**kwargs):
    del kwargs
    return [sys.executable, '-c', script]

  with patch.object(dev_deploy, '_build_deploy_argv', fake_argv):
    yield


def result_of(response) -> dict:
  """Parses the JSON verdict from the final line of a deploy response."""
  _, marker, tail = response.text.rpartition(dev_deploy._RESULT_MARKER)
  assert marker, f'no result line in response: {response.text[:200]}'
  return json.loads(tail)


def deploy(client, target: str, body: dict, script: str) -> dict:
  with stub_child(script):
    response = client.post(f'/dev/apps/{APP_NAME}/deploy/{target}', json=body)
  assert response.status_code == 200, response.text
  return result_of(response)


# --- Request validation ------------------------------------------------------


def test_region_is_required_for_every_target():
  """Each target fails differently without a region, so all three demand it."""
  for model, extra in (
      (AgentEngineDeployRequest, {}),
      (CloudRunDeployRequest, {'serviceName': 'a'}),
      (GkeDeployRequest, {'serviceName': 'a', 'clusterName': 'c'}),
  ):
    with pytest.raises(pydantic.ValidationError):
      model(**extra)


@pytest.mark.parametrize(
    'field,value',
    [
        ('project', '--not-a-project'),
        ('project', 'has space'),
        ('region', '-us-central1'),
        ('region', 'US_CENTRAL1'),
        ('displayName', '--flag-like'),
        ('displayName', 'line\nbreak'),
        ('description', 'x' * 300),
        ('agentEngineId', 'slashes/not/allowed'),
    ],
)
def test_agent_engine_rejects_unsafe_values(field, value):
  """Request values become argv, so flag-like and control input is refused."""
  fields = {'region': 'us-central1'}
  fields[field] = value
  with pytest.raises(pydantic.ValidationError):
    AgentEngineDeployRequest(**fields)


def test_agent_engine_id_accepts_bare_id_and_resource_name():
  bare = AgentEngineDeployRequest(region='us-central1', agentEngineId='812345')
  assert bare.agent_engine_id == '812345'
  full = AgentEngineDeployRequest(
      region='us-central1',
      agentEngineId='projects/p/locations/us-central1/reasoningEngines/7',
  )
  assert full.agent_engine_id.startswith('projects/')


@pytest.mark.parametrize(
    'service_name', ['My_Agent', '-leading', 'trailing-', 'a' * 64, '']
)
def test_cloud_run_rejects_illegal_service_names(service_name):
  with pytest.raises(pydantic.ValidationError):
    CloudRunDeployRequest(region='us-central1', serviceName=service_name)


@pytest.mark.parametrize('port', [0, -1, 65536])
def test_container_targets_reject_out_of_range_ports(port):
  with pytest.raises(pydantic.ValidationError):
    CloudRunDeployRequest(region='us-central1', serviceName='a', port=port)


def test_gke_requires_a_cluster_name():
  with pytest.raises(pydantic.ValidationError):
    GkeDeployRequest(region='us-central1', serviceName='a')


def test_gke_rejects_unknown_service_type():
  with pytest.raises(pydantic.ValidationError):
    GkeDeployRequest(
        region='us-central1',
        serviceName='a',
        clusterName='c',
        serviceType='NodePort',
    )


def test_target_is_a_class_constant_not_a_request_field():
  """The target must not be settable by a client, or it could contradict the
  route it was posted to."""
  for model, expected in (
      (AgentEngineDeployRequest, 'agent_engine'),
      (CloudRunDeployRequest, 'cloud_run'),
      (GkeDeployRequest, 'gke'),
  ):
    assert model.target == expected
    assert 'target' not in model.model_fields


# --- argv construction -------------------------------------------------------


def test_argv_invokes_the_cli_in_the_server_interpreter():
  request = AgentEngineDeployRequest(region='us-central1')
  argv = dev_deploy._build_deploy_argv(
      agent_dir='/agents/my_agent', temp_folder='/tmp/stage', request=request
  )
  assert argv[:5] == [
      sys.executable,
      '-m',
      'google.adk.cli',
      'deploy',
      'agent_engine',
  ]
  # The agent folder is the CLI's positional argument and comes last.
  assert argv[-1] == '/agents/my_agent'
  assert '--temp_folder' in argv


def test_argv_carries_each_target_specific_flag():
  cases = [
      (
          AgentEngineDeployRequest(
              region='us-central1',
              project='adk-demo',
              displayName='My Agent',
              agentEngineId='812345',
          ),
          ['--display_name', 'My Agent', '--agent_engine_id', '812345'],
      ),
      (
          CloudRunDeployRequest(
              region='us-central1',
              serviceName='my-agent',
              port=8080,
              withUi=True,
          ),
          ['--service_name', 'my-agent', '--port', '8080', '--with_ui'],
      ),
      (
          GkeDeployRequest(
              region='us-central1',
              serviceName='my-agent',
              clusterName='my-cluster',
              serviceType='LoadBalancer',
          ),
          ['--cluster_name', 'my-cluster', '--service_type', 'LoadBalancer'],
      ),
  ]
  for request, expected in cases:
    argv = dev_deploy._build_deploy_argv(
        agent_dir='/a', temp_folder='/t', request=request
    )
    for token in expected:
      assert token in argv, f'{token} missing for {request.target}'


@pytest.mark.parametrize(
    'allow,expected',
    [(True, '--allow-unauthenticated'), (False, '--no-allow-unauthenticated')],
)
def test_cloud_run_always_states_its_exposure(allow, expected):
  """Left unsaid, gcloud prompts for this on a terminal the child lacks and
  answers it by default."""
  request = CloudRunDeployRequest(
      region='us-central1', serviceName='a', allowUnauthenticated=allow
  )
  argv = dev_deploy._build_deploy_argv(
      agent_dir='/a', temp_folder='/t', request=request
  )
  assert f'--provider-args={expected}' in argv


def test_staging_folder_is_absolute_and_outside_the_agents_dir():
  """cli_deploy would otherwise stage beside the agent, where the file watcher
  and /list-apps would both pick it up."""
  request = AgentEngineDeployRequest(region='us-central1')
  argv = dev_deploy._build_deploy_argv(
      agent_dir='/agents/my_agent',
      temp_folder='/tmp/adk_deploy/x',
      request=request,
  )
  temp_folder = argv[argv.index('--temp_folder') + 1]
  assert os.path.isabs(temp_folder)
  assert not temp_folder.startswith('/agents')


# --- Service name sanitizing -------------------------------------------------


@pytest.mark.parametrize(
    'folder_name,expected',
    [
        ('my_weather_agent', 'my-weather-agent'),
        ('MyAgent', 'myagent'),
        ('_leading', 'leading'),
        ('123go', 'adk-123go'),
        ('--evil', 'evil'),
    ],
)
def test_service_name_is_derived_from_the_agent_folder(folder_name, expected):
  """Agent folders are Python identifiers; Cloud Run and GKE names are not."""
  assert dev_deploy._sanitize_service_name(folder_name) == expected


def test_sanitized_service_name_fits_the_length_limit():
  slug = dev_deploy._sanitize_service_name('a' * 200)
  assert len(slug) <= dev_deploy._MAX_SERVICE_NAME_CHARS
  # And the result must itself be a legal service name.
  CloudRunDeployRequest(region='us-central1', serviceName=slug)


# --- Defaults resolution -----------------------------------------------------


async def resolve(agents_dir, dotenv=None, environ=None, gcloud=None):
  """Resolves defaults with the three sources controlled independently."""
  agent_dir = agents_dir / APP_NAME
  if dotenv is not None:
    (agent_dir / '.env').write_text(dotenv)

  async def _gcloud():
    return gcloud

  with patch.dict(dev_deploy._STARTUP_ENVIRON, environ or {}, clear=True):
    with patch.object(dev_deploy, '_gcloud_project', _gcloud):
      return await dev_deploy._resolve_defaults(
          app_name=APP_NAME, agent_dir=str(agent_dir)
      )


async def test_defaults_prefer_the_agents_dotenv(agents_dir):
  defaults = await resolve(
      agents_dir,
      dotenv=(
          'GOOGLE_CLOUD_PROJECT=from-dotenv\nGOOGLE_CLOUD_LOCATION=us-west1\n'
      ),
      environ={
          'GOOGLE_CLOUD_PROJECT': 'from-environ',
          'GOOGLE_CLOUD_LOCATION': 'eu-west4',
      },
  )
  assert (defaults.project, defaults.project_source) == (
      'from-dotenv',
      'dotenv',
  )
  assert (defaults.region, defaults.region_source) == ('us-west1', 'dotenv')


async def test_defaults_fall_back_to_the_environment(agents_dir):
  defaults = await resolve(
      agents_dir,
      environ={
          'GOOGLE_CLOUD_PROJECT': 'from-environ',
          'GOOGLE_CLOUD_LOCATION': 'eu-west4',
      },
  )
  assert (defaults.project, defaults.project_source) == (
      'from-environ',
      'environment',
  )
  assert (defaults.region, defaults.region_source) == (
      'eu-west4',
      'environment',
  )


async def test_each_field_resolves_independently(agents_dir):
  """A .env naming only one of the two must not suppress the other."""
  defaults = await resolve(
      agents_dir,
      dotenv='GOOGLE_CLOUD_LOCATION=asia-east1\n',
      environ={'GOOGLE_CLOUD_PROJECT': 'from-environ'},
  )
  assert (defaults.region, defaults.region_source) == (
      'asia-east1',
      'dotenv',
  )
  assert (defaults.project, defaults.project_source) == (
      'from-environ',
      'environment',
  )


async def test_project_falls_back_to_gcloud_but_region_does_not(agents_dir):
  defaults = await resolve(agents_dir, gcloud='from-gcloud')
  assert (defaults.project, defaults.project_source) == (
      'from-gcloud',
      'gcloud',
  )
  assert defaults.region is None
  assert defaults.region_source is None


async def test_defaults_are_empty_when_nothing_supplies_them(agents_dir):
  defaults = await resolve(agents_dir)
  assert defaults.project is None and defaults.project_source is None
  assert defaults.region is None and defaults.region_source is None


async def test_defaults_do_not_leak_other_dotenv_values(agents_dir):
  """Only the two location variables are read; the rest of .env is private."""
  defaults = await resolve(
      agents_dir,
      dotenv='GOOGLE_CLOUD_PROJECT=p\nGOOGLE_API_KEY=super-secret\n',
  )
  assert 'super-secret' not in defaults.model_dump_json()


async def test_defaults_find_a_dotenv_in_a_parent_directory(agents_dir):
  """A .env shared by a whole agents directory still applies."""
  (agents_dir / '.env').write_text('GOOGLE_CLOUD_LOCATION=asia-east1\n')
  defaults = await resolve(agents_dir)
  assert defaults.region == 'asia-east1'
  # The path is reported, because the deploy itself only ships variables from
  # the agent's own .env.
  assert defaults.env_file == str(agents_dir / '.env')


async def test_defaults_read_the_agent_engine_config(agents_dir):
  (agents_dir / APP_NAME / '.agent_engine_config.json').write_text(
      json.dumps({'display_name': 'Weather Bot', 'description': 'Forecasts.'})
  )
  defaults = await resolve(agents_dir)
  assert defaults.display_name == 'Weather Bot'
  assert defaults.description == 'Forecasts.'


async def test_defaults_name_the_agent_when_no_config_exists(agents_dir):
  defaults = await resolve(agents_dir)
  assert defaults.display_name == APP_NAME
  assert defaults.service_name == 'my-agent'


# --- Result parsing ----------------------------------------------------------


def test_agent_runtime_reports_the_resource_and_console_url(client):
  result = deploy(
      client, 'agent_engine', {'region': 'us-central1'}, AGENT_RUNTIME_OK
  )
  assert result['status'] == 'succeeded'
  assert result['resourceName'].endswith('/reasoningEngines/42')
  assert '/agent-engines/42/playground' in result['consoleUrl']
  assert result['serviceUrl'] is None


def test_agent_runtime_clean_exit_without_a_resource_is_a_failure(client):
  """cli_deploy prints a message and returns 0 on several paths that deploy
  nothing, so the exit status alone cannot be trusted."""
  result = deploy(
      client,
      'agent_engine',
      {'region': 'us-central1'},
      'print("Failed to initialize Agent Platform client.")',
  )
  assert result['status'] == 'failed'
  assert result['exitCode'] == 0
  assert 'nothing was deployed' in result['message']


def test_cloud_run_scrapes_the_service_url_and_builds_a_console_url(client):
  result = deploy(
      client,
      'cloud_run',
      {'region': 'us-central1', 'serviceName': 'my-agent'},
      CLOUD_RUN_OK,
  )
  assert result['status'] == 'succeeded'
  assert result['serviceUrl'] == 'https://my-agent-abc123-uc.a.run.app'
  # The project was never sent, so it can only have come from the log -- and
  # it must arrive without gcloud's bold escape codes attached.
  assert result['consoleUrl'].endswith('?project=adk-demo')
  assert '\x1b' not in result['consoleUrl']


def test_cloud_run_prefers_an_explicit_project_over_the_log(client):
  result = deploy(
      client,
      'cloud_run',
      {
          'region': 'us-central1',
          'serviceName': 'my-agent',
          'project': 'explicit-project',
      },
      CLOUD_RUN_OK,
  )
  assert result['consoleUrl'].endswith('?project=explicit-project')


def test_cloud_run_succeeds_even_when_no_url_appears(client):
  result = deploy(
      client,
      'cloud_run',
      {'region': 'us-central1', 'serviceName': 'my-agent'},
      'print("done")',
  )
  assert result['status'] == 'succeeded'
  assert result['serviceUrl'] is None
  assert 'gcloud run services describe' in result['message']


@pytest.mark.parametrize(
    'service_type,expected_hint',
    [
        ('ClusterIP', 'port-forward'),
        ('LoadBalancer', 'kubectl get svc'),
    ],
)
def test_gke_explains_how_to_reach_the_service(
    client, service_type, expected_hint
):
  """GKE exposes nothing addressable from here, so the result carries the
  command to reach it instead of a URL."""
  result = deploy(
      client,
      'gke',
      {
          'region': 'us-central1',
          'serviceName': 'my-agent',
          'clusterName': 'c1',
          'serviceType': service_type,
      },
      'print("Deployment to GKE finished successfully!")',
  )
  assert result['status'] == 'succeeded'
  assert result['consoleUrl'] is None
  assert expected_hint in result['message']


def test_gke_hint_uses_the_generated_services_port(client):
  """The port is a literal in cli_deploy's manifest; the hint must match it."""
  result = deploy(
      client,
      'gke',
      {
          'region': 'us-central1',
          'serviceName': 'my-agent',
          'clusterName': 'c1',
      },
      'print("ok")',
  )
  assert f'8080:{dev_deploy._GKE_SERVICE_PORT}' in result['message']


def test_gke_failure_that_exits_cleanly_is_still_a_failure(client):
  """GKE creates nothing this side can look up, so a clean exit is the only
  other evidence of success. `cli_deploy_gke` swallowed exceptions and exited
  0 until recently, which would have reported a failed deploy as succeeded."""
  result = deploy(
      client,
      'gke',
      {
          'region': 'us-central1',
          'serviceName': 'my-agent',
          'clusterName': 'c1',
      },
      'print("Deploy failed: boom")',
  )
  assert result['status'] == 'failed'
  assert result['exitCode'] == 0
  assert 'exited cleanly' in result['message']


def test_a_failing_child_is_reported_as_failed(client):
  result = deploy(
      client,
      'cloud_run',
      {'region': 'us-central1', 'serviceName': 'my-agent'},
      'import sys; print("ERROR: quota"); sys.exit(3)',
  )
  assert result['status'] == 'failed'
  assert result['exitCode'] == 3


@pytest.mark.parametrize(
    'target,body',
    [
        ('agent_engine', {'region': 'us-central1'}),
        ('cloud_run', {'region': 'us-central1', 'serviceName': 'a'}),
        (
            'gke',
            {
                'region': 'us-central1',
                'serviceName': 'a',
                'clusterName': 'c',
            },
        ),
    ],
)
def test_every_result_carries_the_same_keys(client, target, body):
  """Success and failure paths share one payload shape, so a client never has
  to guess which fields exist."""
  expected = {
      'target',
      'status',
      'exitCode',
      'logPath',
      'resourceName',
      'consoleUrl',
      'serviceUrl',
  }
  for script in ('print("ok")', 'import sys; sys.exit(1)'):
    result = deploy(client, target, body, script)
    assert expected <= set(result), f'missing {expected - set(result)}'
    assert result['target'] == target


# --- Log handling ------------------------------------------------------------


def test_log_reading_strips_ansi_escapes(tmp_path):
  log = tmp_path / 'deploy.log'
  log.write_text('service [\x1b[1mmy-agent\x1b[m] ready\n')
  assert dev_deploy._read_log(str(log)) == 'service [my-agent] ready\n'


def test_log_reading_tolerates_a_missing_file(tmp_path):
  assert dev_deploy._read_log(str(tmp_path / 'absent.log')) == ''


def test_the_deploy_log_is_kept_for_debugging(client):
  result = deploy(
      client, 'agent_engine', {'region': 'us-central1'}, AGENT_RUNTIME_OK
  )
  assert os.path.exists(result['logPath'])


# --- Endpoints ---------------------------------------------------------------


def test_routes_are_registered_under_dev(client):
  paths = {
      (tuple(sorted(r.methods)), r.path)
      for r in client.app.routes
      if 'deploy' in getattr(r, 'path', '')
  }
  assert paths == {
      (('GET',), '/dev/apps/{app_name}/deploy/defaults'),
      (('POST',), '/dev/apps/{app_name}/deploy/agent_engine'),
      (('POST',), '/dev/apps/{app_name}/deploy/cloud_run'),
      (('POST',), '/dev/apps/{app_name}/deploy/gke'),
  }
  # Every route is dev-only: a deployed image has dev_server.py removed.
  assert all(path.startswith('/dev/') for _, path in paths)


def test_unknown_agent_is_a_404(client):
  response = client.post(
      '/dev/apps/nope/deploy/agent_engine', json={'region': 'us-central1'}
  )
  assert response.status_code == 404


def test_an_invalid_body_is_rejected_before_anything_is_spawned(client):
  response = client.post(f'/dev/apps/{APP_NAME}/deploy/agent_engine', json={})
  assert response.status_code == 422
  assert not dev_deploy._deploys_in_flight


def test_a_second_deploy_of_the_same_agent_is_refused(client):
  """Two concurrent Agent Runtime deploys would create two billed instances."""
  dev_deploy._deploys_in_flight[APP_NAME] = None
  response = client.post(
      f'/dev/apps/{APP_NAME}/deploy/gke',
      json={
          'region': 'us-central1',
          'serviceName': 'a',
          'clusterName': 'c',
      },
  )
  assert response.status_code == 409


def test_the_slot_is_released_once_the_deploy_finishes(client):
  deploy(client, 'agent_engine', {'region': 'us-central1'}, 'print("ok")')
  assert not dev_deploy._deploys_in_flight


def test_defaults_endpoint_serves_every_target(client):
  response = client.get(f'/dev/apps/{APP_NAME}/deploy/defaults')
  assert response.status_code == 200
  body = response.json()
  # camelCase, like every other response the UI consumes.
  assert 'serviceName' in body and 'projectSource' in body


def test_defaults_endpoint_404s_for_an_unknown_agent(client):
  assert client.get('/dev/apps/nope/deploy/defaults').status_code == 404


# --- Child process lifecycle -------------------------------------------------


async def spawn(agents_dir, script, request=None):
  request = request or AgentEngineDeployRequest(region='us-central1')
  with stub_child(script):
    return await dev_deploy._spawn_deploy(
        app_name=APP_NAME,
        agent_dir=str(agents_dir / APP_NAME),
        request=request,
    )


async def test_the_child_runs_unbuffered(agents_dir):
  """click.echo into a file is block-buffered, which would stall the UI."""
  captured = {}
  real = asyncio.create_subprocess_exec

  async def spy(*args, **kwargs):
    captured.update(kwargs.get('env') or {})
    captured['stdin'] = kwargs.get('stdin')
    return await real(*args, **kwargs)

  with patch.object(asyncio, 'create_subprocess_exec', spy):
    process, _, waiter = await spawn(agents_dir, 'print("ok")')
  await waiter
  assert captured.get('PYTHONUNBUFFERED') == '1'
  # No stdin, so a tool that prompts fails fast instead of hanging forever.
  assert captured.get('stdin') is not None
  assert process.returncode == 0


async def test_a_disconnected_client_does_not_kill_the_deploy(
    agents_dir, tmp_path
):
  """cli_deploy creates the Agent Runtime before building its image and only
  deletes it if the later update raises, so an interrupted child can strand an
  empty instance. Losing the log is the acceptable cost; losing the deploy is
  not."""
  marker = tmp_path / 'child_finished'
  script = (
      'import time\n'
      'print("starting", flush=True)\n'
      'time.sleep(1.5)\n'
      f'open({str(marker)!r}, "w").write("done")\n'
  )
  dev_deploy._deploys_in_flight[APP_NAME] = None
  process, log_path, waiter = await spawn(agents_dir, script)
  stream = dev_deploy._tail_deploy(
      app_name=APP_NAME,
      agent_dir=str(agents_dir / APP_NAME),
      process=process,
      log_path=log_path,
      waiter=waiter,
      request=AgentEngineDeployRequest(region='us-central1'),
  )
  await stream.__anext__()
  await stream.aclose()  # the browser goes away

  assert not marker.exists(), 'child finished before the test disconnected'
  await waiter
  assert marker.exists(), 'the disconnect killed the deploy'


async def test_the_slot_is_held_until_the_child_exits_not_the_request(
    agents_dir,
):
  """Releasing on disconnect would let a duplicate deploy start while the
  first is still running."""
  script = 'import time; time.sleep(1.0)'
  dev_deploy._deploys_in_flight[APP_NAME] = None
  process, log_path, waiter = await spawn(agents_dir, script)
  stream = dev_deploy._tail_deploy(
      app_name=APP_NAME,
      agent_dir=str(agents_dir / APP_NAME),
      process=process,
      log_path=log_path,
      waiter=waiter,
      request=AgentEngineDeployRequest(region='us-central1'),
  )
  await stream.__anext__()
  await stream.aclose()

  assert APP_NAME in dev_deploy._deploys_in_flight
  await waiter
  await asyncio.sleep(0)  # let the waiter's done callback run
  assert APP_NAME not in dev_deploy._deploys_in_flight


async def test_a_child_that_cannot_start_frees_the_slot(client, agents_dir):
  """A failure to spawn must not wedge the agent into a permanent 409."""

  def explode(**kwargs):
    del kwargs
    return ['/nonexistent/interpreter', '-c', 'pass']

  with patch.object(dev_deploy, '_build_deploy_argv', explode):
    response = client.post(
        f'/dev/apps/{APP_NAME}/deploy/agent_engine',
        json={'region': 'us-central1'},
    )
  assert response.status_code == 500
  assert not dev_deploy._deploys_in_flight
