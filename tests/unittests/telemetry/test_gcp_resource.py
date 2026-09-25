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

"""Tests for the OTel resource describing where an agent runs."""

from typing import Optional
from unittest import mock

from google.adk.telemetry import _gcp_resource
from google.adk.telemetry._gcp_resource import _fetch_project
from google.adk.telemetry._gcp_resource import _get_gcp_detected_resource
from google.adk.telemetry._gcp_resource import _Project
from google.adk.telemetry._gcp_resource import _PROJECT_CACHE_SIZE
from google.adk.telemetry._gcp_resource import get_gcp_resource
import google.cloud
from opentelemetry.sdk.resources import Resource
import pytest

_GKE_ATTRIBUTES = {
    "cloud.platform": "gcp_kubernetes_engine",
    "cloud.account.id": "my-project",
    "cloud.region": "us-central1",
    "k8s.cluster.name": "my-cluster",
    "k8s.namespace.name": "default",
    "k8s.deployment.name": "my-agent",
}
_CLOUD_RUN_ATTRIBUTES = {
    "cloud.platform": "gcp_cloud_run",
    "cloud.account.id": "my-project",
    "cloud.region": "us-central1",
    "faas.name": "my-agent",
}


def _fake_get_project(name: str) -> mock.Mock:
  """Answers like Resource Manager: both identifiers, whichever was asked by."""
  project = name.removeprefix("projects/")
  fetched = mock.Mock()
  if project.isdecimal():
    fetched.project_id = f"{project}-id"
    fetched.name = f"projects/{project}"
  else:
    fetched.project_id = project
    fetched.name = f"projects/{project}-number"
  return fetched


@pytest.fixture(autouse=True)
def resource_manager():
  """Stands in for Resource Manager, so no test reaches the network.

  Yields the stub module, so a test can assert what was looked up. The memo is
  emptied around every test: it is process-wide, and would outlive one.
  """
  resourcemanager = mock.Mock()
  resourcemanager.ProjectsClient.return_value.get_project.side_effect = (
      _fake_get_project
  )

  _fetch_project.cache.clear()
  # Both, because `from google.cloud import resourcemanager_v3` reads the
  # attribute off the package when the real module has already been imported,
  # and consults `sys.modules` only when it has not.
  with (
      mock.patch.object(
          google.cloud, "resourcemanager_v3", resourcemanager, create=True
      ),
      mock.patch.dict(
          "sys.modules", {"google.cloud.resourcemanager_v3": resourcemanager}
      ),
  ):
    yield resourcemanager
  _fetch_project.cache.clear()


@pytest.fixture(autouse=True)
def off_agent_runtime(monkeypatch: pytest.MonkeyPatch):
  """Nothing here should inherit an Agent Runtime from the environment."""
  monkeypatch.delenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", raising=False)
  monkeypatch.delenv("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION", raising=False)
  monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
  monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)


@pytest.fixture(autouse=True)
def detected_platform():
  """Stands in for the GCP detector, which would probe the metadata server.

  Yields the patched function: a test for one platform sets its return value to
  the resource that platform's detector would have produced.
  """
  with mock.patch.object(
      _gcp_resource,
      "_get_gcp_detected_resource",
      autospec=True,
      return_value=Resource.get_empty(),
  ) as detected:
    yield detected


# The resource, source by source.


@pytest.mark.parametrize("project_id_in_arg", ["project_id_in_arg", None])
@pytest.mark.parametrize("project_id_on_env", ["project_id_on_env", None])
def test_get_gcp_resource(
    project_id_in_arg: Optional[str],
    project_id_on_env: Optional[str],
    monkeypatch: pytest.MonkeyPatch,
):
  """The OTel environment overrides the project the caller passed in."""
  # Arrange.
  if project_id_on_env is not None:
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES", f"gcp.project_id={project_id_on_env}"
    )

  # Act.
  otel_resource = get_gcp_resource(project_id_in_arg)

  # Assert.
  # Both are either a non-empty string or None, so the environment wins
  # whenever it is set and the argument stands in otherwise.
  expected_project_id = project_id_on_env or project_id_in_arg
  assert otel_resource is not None
  assert (
      otel_resource.attributes.get("gcp.project_id", None)
      == expected_project_id
  )


def test_get_gcp_resource_identifies_the_process():
  """Two replicas of one deployment have to stay distinguishable."""
  first = get_gcp_resource("my-project").attributes["service.instance.id"]
  second = get_gcp_resource("my-project").attributes["service.instance.id"]

  assert first != second


def test_get_gcp_resource_keeps_what_the_platform_detected(  # pylint: disable=redefined-outer-name
    detected_platform: mock.Mock,
):
  """The platform's own answer is the one that has to survive the merge."""
  detected_platform.return_value = Resource(attributes=_CLOUD_RUN_ATTRIBUTES)

  otel_resource = get_gcp_resource("my-project")

  assert otel_resource.attributes["cloud.platform"] == "gcp_cloud_run"
  assert otel_resource.attributes["faas.name"] == "my-agent"


def test_get_gcp_resource_is_not_agent_runtime_off_agent_runtime():
  """Local, GCE, GKE and Cloud Run runs are not Agent Runtime deployments."""
  otel_resource = get_gcp_resource("my-project")

  # Whatever the platform is, the GCP detector decides it -- not us.
  assert otel_resource.attributes.get("cloud.platform") != "gcp.agent_engine"
  assert "cloud.resource_id" not in otel_resource.attributes
  assert otel_resource.attributes["gcp.project_id"] == "my-project"


def test_gcp_detection_is_skipped_on_agent_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
  """It describes the infrastructure underneath, and it merges last.

  Called directly rather than through `get_gcp_resource`, whose fixture stands
  in for the detector and so cannot show that it was never consulted.

  Args:
    monkeypatch: puts this process on Agent Runtime.
  """
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "1234567890")

  assert not _get_gcp_detected_resource().attributes


def test_get_gcp_resource_describes_the_agent_runtime_deployment(
    monkeypatch: pytest.MonkeyPatch,
):
  """Agent Runtime names the deployment, not the infrastructure under it."""
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "1234567890")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

  otel_resource = get_gcp_resource("my-project")

  assert otel_resource.attributes["cloud.platform"] == "gcp.agent_engine"
  assert otel_resource.attributes["service.name"] == "1234567890"
  assert otel_resource.attributes["cloud.region"] == "us-central1"
  assert otel_resource.attributes["cloud.account.id"] == "my-project"
  # Contributed by `Resource.create`, as they were before OTLP export.
  assert otel_resource.attributes["telemetry.sdk.language"] == "python"
  assert otel_resource.attributes["telemetry.sdk.name"] == "opentelemetry"


def test_get_gcp_resource_sets_standard_cloud_resource_id(
    monkeypatch: pytest.MonkeyPatch,
):
  """The OTel-standard key is the one the Agent Engine dashboard filters on."""
  # Arrange.
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "1234567890")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

  # Act.
  otel_resource = get_gcp_resource("my-project")

  # Assert.
  # The Agent Engine dashboard filters on the OTel-standard key.
  assert otel_resource.attributes.get("cloud.resource_id") == (
      "//aiplatform.googleapis.com/projects/my-project"
      "/locations/us-central1/reasoningEngines/1234567890"
  )
  assert "cloud.resource.id" not in otel_resource.attributes


# The Agent Registry URN.


def test_get_gcp_resource_sets_main_agent_id_on_agent_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
  """The Agent Registry URN identifies the project by number, not by ID."""
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "1234567890")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

  otel_resource = get_gcp_resource("123456789")

  assert otel_resource.attributes["gen_ai.main_agent.id"] == (
      "urn:agent:projects-123456789:projects:123456789:locations:"
      "us-central1:aiplatform:reasoningEngines:1234567890"
  )
  # The rest of the resource keeps identifying the project the way it was given.
  assert otel_resource.attributes["gcp.project_id"] == "123456789"


def test_get_gcp_resource_never_puts_an_unresolved_project_in_the_urn(  # pylint: disable=redefined-outer-name
    monkeypatch: pytest.MonkeyPatch,
    resource_manager: mock.Mock,
):
  """Digits are not a number until Resource Manager says they are.

  Roughly 20k legacy projects have an all-digit ID, so reading the digits as
  the number when the lookup fails would put an ID in the URN, which joins to
  nothing while looking like it should.

  Args:
    monkeypatch: puts this process on Agent Runtime.
    resource_manager: stands in for Resource Manager, made to refuse the lookup.
  """
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "1234567890")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
  resource_manager.ProjectsClient.return_value.get_project.side_effect = (
      PermissionError("no resource manager here")
  )

  otel_resource = get_gcp_resource("123456789")

  assert "gen_ai.main_agent.id" not in otel_resource.attributes
  # Only the URN is that strict; the rest still says what it was given.
  assert otel_resource.attributes["gcp.project_id"] == "123456789"


def test_get_gcp_resource_looks_up_main_agent_id_project_number(  # pylint: disable=redefined-outer-name
    monkeypatch: pytest.MonkeyPatch,
    resource_manager: mock.Mock,
):
  """A project ID has to be converted before it can go in the URN."""
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "1234567890")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

  otel_resource = get_gcp_resource("my-project")

  assert otel_resource.attributes["gen_ai.main_agent.id"] == (
      "urn:agent:projects-my-project-number:projects:my-project-number"
      ":locations:us-central1:aiplatform:reasoningEngines:1234567890"
  )
  assert otel_resource.attributes["gcp.project_id"] == "my-project"
  resource_manager.ProjectsClient.return_value.get_project.assert_called_once_with(
      name="projects/my-project"
  )


@pytest.mark.parametrize(
    "attributes,expected",
    [
        pytest.param(
            _GKE_ATTRIBUTES,
            "urn:agent:projects-my-project-number:projects:my-project-number"
            ":locations:us-central1:container:clusters:my-cluster"
            ":k8s:namespaces:default:apps:deployments:my-agent",
            id="gke_regional_cluster",
        ),
        pytest.param(
            # A zonal cluster reports its zone instead of its region.
            {
                **_GKE_ATTRIBUTES,
                "cloud.region": None,
                "cloud.availability_zone": "us-central1-c",
            },
            "urn:agent:projects-my-project-number:projects:my-project-number"
            ":zones:us-central1-c:container:clusters:my-cluster"
            ":k8s:namespaces:default:apps:deployments:my-agent",
            id="gke_zonal_cluster",
        ),
        pytest.param(
            _CLOUD_RUN_ATTRIBUTES,
            "urn:agent:projects-my-project-number:projects:my-project-number"
            ":locations:us-central1:run:services:my-agent",
            id="cloud_run_service",
        ),
    ],
)
def test_get_gcp_resource_sets_main_agent_id_off_agent_runtime(  # pylint: disable=redefined-outer-name
    detected_platform: mock.Mock,
    attributes: dict[str, Optional[str]],
    expected: str,
):
  """GKE and Cloud Run agents are named the way Agent Registry names them."""
  detected_platform.return_value = Resource(
      attributes={k: v for k, v in attributes.items() if v is not None}
  )

  otel_resource = get_gcp_resource("my-project")

  assert otel_resource.attributes["gen_ai.main_agent.id"] == expected


def test_get_gcp_resource_reads_gke_namespace_and_deployment_off_the_pod(  # pylint: disable=redefined-outer-name
    monkeypatch: pytest.MonkeyPatch,
    detected_platform: mock.Mock,
    tmp_path,
):
  """Neither is a resource attribute the GCP detector can produce."""
  # A pod learns its namespace from its service account token, and its own name
  # from HOSTNAME: `<deployment>-<replicaset hash>-<pod hash>`.
  namespace_path = tmp_path / "namespace"
  _ = namespace_path.write_text("agents\n")
  monkeypatch.setattr(_gcp_resource, "_K8S_NAMESPACE_PATH", str(namespace_path))
  monkeypatch.setenv("HOSTNAME", "my-agent-7d8f9c5b4-xk2p9")
  detected_platform.return_value = Resource(
      attributes={
          k: v
          for k, v in _GKE_ATTRIBUTES.items()
          if k not in ("k8s.namespace.name", "k8s.deployment.name")
      }
  )

  otel_resource = get_gcp_resource("my-project")

  assert otel_resource.attributes["gen_ai.main_agent.id"] == (
      "urn:agent:projects-my-project-number:projects:my-project-number"
      ":locations:us-central1:container:clusters:my-cluster"
      ":k8s:namespaces:agents:apps:deployments:my-agent"
  )


@pytest.mark.parametrize(
    "project,location",
    [
        pytest.param(None, "us-central1", id="no_project"),
        pytest.param("", "us-central1", id="empty_project"),
        pytest.param("my-project", None, id="missing_location"),
    ],
)
def test_get_gcp_resource_omits_an_incomplete_urn_on_agent_runtime(
    monkeypatch: pytest.MonkeyPatch,
    project: Optional[str],
    location: Optional[str],
):
  """A URN missing a segment joins to the wrong agent, so it is not emitted."""
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "1234567890")
  if location:
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", location)

  otel_resource = get_gcp_resource(project)

  assert "gen_ai.main_agent.id" not in otel_resource.attributes


@pytest.mark.parametrize(
    "attributes",
    [
        pytest.param(
            {**_GKE_ATTRIBUTES, "k8s.cluster.name": None},
            id="gke_without_a_cluster",
        ),
        pytest.param(
            {**_GKE_ATTRIBUTES, "k8s.deployment.name": None},
            id="gke_without_a_deployment",
        ),
        pytest.param(
            {**_CLOUD_RUN_ATTRIBUTES, "cloud.region": None},
            id="cloud_run_without_a_region",
        ),
        pytest.param(
            {**_CLOUD_RUN_ATTRIBUTES, "faas.name": None},
            id="cloud_run_without_a_service",
        ),
    ],
)
def test_get_gcp_resource_omits_an_incomplete_urn_off_agent_runtime(  # pylint: disable=redefined-outer-name
    monkeypatch: pytest.MonkeyPatch,
    detected_platform: mock.Mock,
    attributes: dict[str, Optional[str]],
):
  """A URN missing a segment joins to the wrong agent, so it is not emitted."""
  # Nothing to fall back on: the pod name is not one a Deployment would give.
  monkeypatch.setenv("HOSTNAME", "not-a-pod")
  monkeypatch.setattr(_gcp_resource, "_K8S_NAMESPACE_PATH", "/nonexistent")
  detected_platform.return_value = Resource(
      attributes={k: v for k, v in attributes.items() if v is not None}
  )

  otel_resource = get_gcp_resource("my-project")

  assert "gen_ai.main_agent.id" not in otel_resource.attributes


def test_get_gcp_resource_omits_main_agent_id_off_a_named_platform(  # pylint: disable=redefined-outer-name
    detected_platform: mock.Mock,
):
  """Agent Registry names no agent on plain GCE, so there is no URN at all."""
  detected_platform.return_value = Resource(
      attributes={
          **_CLOUD_RUN_ATTRIBUTES,
          "cloud.platform": "gcp_compute_engine",
      }
  )

  otel_resource = get_gcp_resource("my-project")

  assert "gen_ai.main_agent.id" not in otel_resource.attributes


def test_get_gcp_resource_omits_the_urn_without_a_resolvable_project_number(  # pylint: disable=redefined-outer-name
    resource_manager: mock.Mock,
    detected_platform: mock.Mock,
):
  """A Resource Manager failure costs the whole URN, not just the number."""
  resource_manager.ProjectsClient.side_effect = RuntimeError("boom")
  detected_platform.return_value = Resource(attributes=_CLOUD_RUN_ATTRIBUTES)

  otel_resource = get_gcp_resource("my-project")

  assert "gen_ai.main_agent.id" not in otel_resource.attributes


def test_get_gcp_resource_keeps_a_main_agent_id_the_platform_set(  # pylint: disable=redefined-outer-name
    monkeypatch: pytest.MonkeyPatch,
    detected_platform: mock.Mock,
):
  """The runtime knows what it deployed better than we can reconstruct it."""
  monkeypatch.setenv(
      "OTEL_RESOURCE_ATTRIBUTES", "gen_ai.main_agent.id=urn:agent:stated"
  )
  detected_platform.return_value = Resource(attributes=_CLOUD_RUN_ATTRIBUTES)

  otel_resource = get_gcp_resource("my-project")

  assert otel_resource.attributes["gen_ai.main_agent.id"] == "urn:agent:stated"


# The project identifiers.


@pytest.mark.parametrize(
    "project,expected",
    [
        pytest.param(
            # Digits are not a number until the lookup says so: an all-digit
            # project ID would otherwise reach the URN as one.
            "123456789",
            _Project(id="", number=""),
            id="digits_are_not_taken_for_a_number_when_the_lookup_fails",
        ),
        pytest.param(
            "my-project",
            _Project(id="my-project", number=""),
            id="id_kept_when_lookup_fails",
        ),
    ],
)
def test_fetch_project_returns_what_it_was_given_when_the_lookup_fails(  # pylint: disable=redefined-outer-name
    resource_manager: mock.Mock,
    project: str,
    expected: _Project,
):
  """Losing one identifier must not cost us an ID we were handed."""
  resource_manager.ProjectsClient.side_effect = RuntimeError("boom")

  assert _fetch_project(project) == expected


def test_fetch_project_does_not_memoize_a_failed_lookup(  # pylint: disable=redefined-outer-name
    resource_manager: mock.Mock,
):
  """A blip at startup must not cost the identifier for the whole process."""
  resource_manager.ProjectsClient.side_effect = RuntimeError("boom")
  assert not _fetch_project("my-project").number

  resource_manager.ProjectsClient.side_effect = None

  assert _fetch_project("my-project").number == "my-project-number"


@pytest.mark.parametrize("project", [None, ""], ids=["none", "empty"])
def test_fetch_project_does_not_look_up_an_absent_project(  # pylint: disable=redefined-outer-name
    resource_manager: mock.Mock,
    project: Optional[str],
):
  """`projects/` is not a lookup worth making."""
  assert _fetch_project(project) == _Project(id="", number="")

  resource_manager.ProjectsClient.assert_not_called()


def test_fetch_project_looks_up_a_project_only_once(  # pylint: disable=redefined-outer-name
    resource_manager: mock.Mock,
):
  """One lookup answers both callers: one wants the ID, the other the number."""
  assert _fetch_project("my-project").id == "my-project"
  assert _fetch_project("my-project").number == "my-project-number"

  resource_manager.ProjectsClient.return_value.get_project.assert_called_once_with(
      name="projects/my-project"
  )


def test_fetch_project_bounds_what_it_memoizes():
  """The memo is a cache, not a record of every project ever looked up."""
  for index in range(_PROJECT_CACHE_SIZE + 1):
    _ = _fetch_project(f"project-{index}")

  assert len(_fetch_project.cache) == _PROJECT_CACHE_SIZE
  # The oldest went, the newest stayed.
  assert "project-0" not in _fetch_project.cache
  assert f"project-{_PROJECT_CACHE_SIZE}" in _fetch_project.cache


class _RacedCache(dict):
  """A cache another thread evicts from between picking a key and popping it.

  Picking the oldest key and popping it are two steps, so two threads crossing
  the size limit together pick the same key and the second pop finds it gone.
  Iterating hands back the key it has just dropped, which is what the thread
  that lost that race sees.
  """

  def __iter__(self):
    oldest = next(super().__iter__())
    _ = self.pop(oldest, None)
    return iter([oldest])


def test_fetch_project_survives_a_concurrent_eviction(
    monkeypatch: pytest.MonkeyPatch,
):
  """Losing the race to evict must not take the lookup down with it."""
  monkeypatch.setattr(
      _fetch_project,
      "cache",
      _RacedCache({
          f"project-{index}": _Project(id="", number="1")
          for index in range(_PROJECT_CACHE_SIZE)
      }),
  )

  assert _fetch_project("late-project").number == "late-project-number"


def test_fetch_project_uses_the_credentials_it_was_given(  # pylint: disable=redefined-outer-name
    resource_manager: mock.Mock,
):
  """The lookup has to run as whoever the caller exports telemetry as."""
  credentials = mock.Mock(name="credentials")

  _ = _fetch_project("my-project", credentials)

  resource_manager.ProjectsClient.assert_called_once_with(
      credentials=credentials
  )
